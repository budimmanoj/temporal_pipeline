# llm_ner_validator.py  — Arch 2: Stage 1.5
#
# The BiLSTM+CRF NER model misses temporal expressions in two common ways:
#
#   1. FULL MISS   — span not extracted at all
#                    e.g. "the following morning" → nothing returned
#
#   2. PARTIAL SPAN — relational context stripped
#                    e.g. "two days later" → only "two days" returned,
#                    losing the cross-dep signal (handled partially by
#                    _sentence_cross_dep, but not always)
#
# This stage runs ONE LLM call per sentence (only for sentences where
# the NER output looks suspect) to:
#   a) Verify every temporal expression has been captured.
#   b) Inject any missed spans with their correct entity type.
#   c) Flag partial spans (e.g. "two days" when sentence has "two days later")
#      so the rule engine can re-evaluate them with the full span.
#
# DESIGN CONSTRAINTS:
#   - Only adds new entities; never removes or modifies NER-found ones.
#   - LLM output is schema-validated; bad responses are silently dropped
#     (no poisoning of the entity list).
#   - Falls back gracefully if Ollama is unavailable.

import re
import json

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ── Entity types the pipeline handles ────────────────────────
VALID_TYPES = {"DATE", "TIME", "DURATION", "SET", "EVENT"}

# ── Patterns that strongly suggest a missed cross-dep span ───
# Used to decide which sentences need LLM validation at all.
_CROSS_DEP_HINTS = re.compile(
    r'\b('
    r'the following (morning|afternoon|evening|night|day|week|month)|'
    r'the next (morning|afternoon|evening|day|week|month)|'
    r'shortly after|soon after|immediately after|'
    r'later that (morning|evening|night|day)|'
    r'that (morning|evening|night|afternoon)|'
    r'\d+\s+(hour|day|week|month|year)s?\s+(later|after|before|following)|'
    r'(two|three|four|five|six|several|few|a couple of)\s+'
    r'(hour|day|week|month|year)s?\s+(later|after|following)|'
    r'within\s+\w+\s+(hour|day|week|month|year)s?\s+of|'
    r'since (then|that)|'
    r'by (then|that time|the time)|'
    r'at that (point|time|moment)'
    r')',
    re.IGNORECASE,
)

# Known TimeML temporal surface patterns for quick candidate detection
_TEMPORAL_SURFACE = re.compile(
    r'\b('
    r'yesterday|today|tomorrow|'
    r'last\s+\w+|next\s+\w+|this\s+\w+|'
    r'the following \w+|the next \w+|'
    r'\d+\s+(second|minute|hour|day|week|month|year)s?|'
    r'(a|an|one|two|three|four|five|six|seven|eight|nine|ten|several|few|couple of)\s+'
    r'(second|minute|hour|day|week|month|year)s?|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
    r'january|february|march|april|may|june|july|august|'
    r'september|october|november|december|'
    r'morning|afternoon|evening|night|midnight|noon|'
    r'daily|weekly|monthly|annually|hourly|'
    r'every\s+\w+|once\s+a\s+\w+|twice\s+a\s+\w+'
    r')',
    re.IGNORECASE,
)


# ── Few-shot prompt ───────────────────────────────────────────

_FEW_SHOT = """
=== VALIDATION EXAMPLE 1 ===
Sentence (idx=0): "The patient was admitted last night."
NER found: [{"text": "last night", "type": "TIME"}]

All temporal expressions: ["last night"]
Missing: []
Output: []

=== VALIDATION EXAMPLE 2 ===
Sentence (idx=1): "Surgery was performed the following morning and recovery started two days later."
NER found: [{"text": "two days", "type": "DURATION"}]

All temporal expressions: ["the following morning", "two days later"]
Missing: ["the following morning"]
Partial: [{"ner_text": "two days", "full_text": "two days later", "type": "DURATION"}]

Output:
[
  {"text": "the following morning", "type": "TIME", "sentence_idx": 1, "issue": "full_miss"},
  {"text": "two days later", "type": "DURATION", "sentence_idx": 1, "issue": "partial_span",
   "replaces": "two days"}
]

=== VALIDATION EXAMPLE 3 ===
Sentence (idx=2): "Rescue teams arrived within two hours of the explosion."
NER found: [{"text": "two hours", "type": "DURATION"}]

All temporal expressions: ["within two hours of the explosion"]
Partial: [{"ner_text": "two hours", "full_text": "within two hours of the explosion", "type": "DURATION"}]

Output:
[
  {"text": "within two hours of the explosion", "type": "DURATION", "sentence_idx": 2,
   "issue": "partial_span", "replaces": "two hours"}
]

=== VALIDATION EXAMPLE 4 ===
Sentence (idx=0): "The conference started last Monday."
NER found: [{"text": "last Monday", "type": "DATE"}]

All temporal expressions: ["last Monday"]
Missing: []
Output: []
"""


# ── Main validator ────────────────────────────────────────────

def validate_ner_output(sentences: list,
                        raw_entities: list,
                        llm_model: str = "mistral") -> list:
    """
    Stage 1.5: LLM validation of NER output.

    For each sentence that has cross-dep hints or looks under-extracted,
    ask the LLM to list ALL temporal expressions and flag misses/partials.

    Returns a NEW raw entity list (original NER entities + injected ones).
    Entity order is preserved; injected entities are appended after the
    NER entities for the same sentence.

    If Ollama is unavailable, returns raw_entities unchanged.
    """
    if not OLLAMA_AVAILABLE:
        print("[Stage 1.5 — NER Validator]  Ollama unavailable — skipping.")
        return raw_entities

    # Group existing NER entities by sentence
    ner_by_sent: dict[int, list] = {i: [] for i in range(len(sentences))}
    for e in raw_entities:
        ner_by_sent[e["sentence_idx"]].append(e)

    # Decide which sentences need validation
    sentences_to_check = []
    for idx, sent in enumerate(sentences):
        found = ner_by_sent[idx]
        # Always check if cross-dep hint found
        if _CROSS_DEP_HINTS.search(sent):
            sentences_to_check.append(idx)
            continue
        # Check if there seem to be temporal expressions the NER missed
        candidate_count = len(_TEMPORAL_SURFACE.findall(sent))
        if candidate_count > len(found):
            sentences_to_check.append(idx)

    if not sentences_to_check:
        print("[Stage 1.5 — NER Validator]  All sentences look complete — skipping LLM.")
        return raw_entities

    print(f"[Stage 1.5 — NER Validator]  Checking {len(sentences_to_check)} sentence(s): "
          f"{sentences_to_check}")

    injected: list[dict] = []

    for sidx in sentences_to_check:
        sent  = sentences[sidx]
        found = ner_by_sent[sidx]
        found_json = json.dumps(
            [{"text": e["text"], "type": e["type"]} for e in found], indent=2
        )

        prompt = f"""You are a TimeML temporal expression validator.

Your job:
1. Read the sentence carefully.
2. List ALL temporal expressions present (DATE, TIME, DURATION, SET, EVENT).
3. Compare against what the NER already found.
4. Return ONLY the missing or partially-captured expressions.

Rules:
- Include relational context in the span:
    WRONG: "two days"      (when sentence says "two days later")
    RIGHT: "two days later"
    WRONG: "two hours"     (when sentence says "within two hours of the explosion")
    RIGHT: "within two hours of the explosion"
- "issue" must be one of: "full_miss" (NER missed it entirely) or
  "partial_span" (NER found it but stripped context).
- For "partial_span", include "replaces" with the exact NER text to replace.
- If nothing is missing, return an empty array: []
- Output ONLY a valid JSON array. No markdown, no explanation.

Each object must have:
  text         (str)   — the full temporal expression span
  type         (str)   — DATE | TIME | DURATION | SET | EVENT
  sentence_idx (int)   — always {sidx}
  issue        (str)   — "full_miss" or "partial_span"
  replaces     (str)   — only for "partial_span": the NER text being corrected

{_FEW_SHOT}

=== NOW VALIDATE ===
Sentence (idx={sidx}): "{sent}"
NER found: {found_json}

Output:"""

        try:
            res = ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = res["message"]["content"].strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            results = json.loads(raw)

            if not isinstance(results, list):
                print(f"   [S{sidx}] Validator returned non-list — skipping")
                continue

            for item in results:
                if not _validate_item(item, sidx):
                    print(f"   [S{sidx}] Invalid validator item: {item} — skipping")
                    continue

                issue = item["issue"]
                text  = item["text"].strip()
                typ   = item["type"].upper()

                if issue == "partial_span":
                    # Replace the partial NER entry with the full span
                    replaces = item.get("replaces", "").strip()
                    replaced = False
                    for e in raw_entities:
                        if (e["sentence_idx"] == sidx and
                                e["text"].lower() == replaces.lower()):
                            old = e["text"]
                            e["text"] = text
                            e["type"] = typ
                            print(f"   [S{sidx}] PARTIAL FIX  '{old}' → '{text}'  [{typ}]")
                            replaced = True
                            break
                    if not replaced:
                        # replaces didn't match exactly — treat as full_miss
                        issue = "full_miss"

                if issue == "full_miss":
                    # Check not already present (avoid duplicates)
                    already = any(
                        e["sentence_idx"] == sidx and
                        e["text"].lower() == text.lower()
                        for e in raw_entities + injected
                    )
                    if not already:
                        injected.append({
                            "sentence_idx": sidx,
                            "text":         text,
                            "type":         typ,
                        })
                        print(f"   [S{sidx}] INJECTED     '{text}'  [{typ}]")

        except json.JSONDecodeError as ex:
            print(f"   [S{sidx}] JSON parse error: {ex} — skipping")
        except Exception as ex:
            print(f"   [S{sidx}] LLM error: {ex} — skipping")

    final = raw_entities + injected
    n_added = len(injected)
    n_fixed = 0
    # Count partial fixes (text was changed in-place)
    print(f"\n[Stage 1.5 — NER Validator]  "
          f"+{n_added} injected, partial-span fixes applied in-place")
    return final


# ── Schema validation for LLM output items ───────────────────

def _validate_item(item: dict, expected_sidx: int) -> bool:
    """Return True if the validator output item is well-formed."""
    if not isinstance(item, dict):
        return False
    if "text" not in item or not isinstance(item["text"], str) or not item["text"].strip():
        return False
    if "type" not in item or item["type"].upper() not in VALID_TYPES:
        return False
    if "issue" not in item or item["issue"] not in ("full_miss", "partial_span"):
        return False
    if item.get("sentence_idx") != expected_sidx:
        return False
    if item["issue"] == "partial_span" and not item.get("replaces"):
        return False
    return True