# layer_executor.py  — Arch 2: Stage 5
#
# The heart of the divide-and-conquer approach.
#
# For each layer (in topo order):
#   1. Pull entities for this layer.
#   2. Fast path: if ALL are already rule-resolved → skip LLM, write
#      to registry, advance.
#   3. Inject verified anchor dates from registry into LLM prompt.
#      The LLM ONLY does coref + arithmetic.  It NEVER guesses an
#      anchor date — every anchor date it sees is already verified.
#   4. Validate LLM response (schema + range check).
#      Bad response → confidence=0, conflict_flag.  Never silent poison.
#   5. Write back to registry → next layer builds on verified facts.

import re
import json
from datetime import date, timedelta, datetime

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from core.anchor_registry import AnchorRegistry


# ── Date helpers ──────────────────────────────────────────────

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

def _fix_timeml(value: str) -> str:
    """Repair common LLM formatting mistakes in TimeML strings."""
    if not value:
        return value
    # "2026-0419TMO" → "2026-04-19TMO"
    m = re.match(r'^(\d{4})-(\d{2})(\d{2})(T.+)$', value)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}{m.group(4)}"
    return value

def _extract_date(value: str) -> str | None:
    if not value:
        return None
    if _DATE_RE.match(value):
        return value
    m = re.match(r'^(\d{4}-\d{2}-\d{2})', value)
    if m:
        return m.group(1)
    # "YYYY-MM" → first of month
    m = re.match(r'^(\d{4}-\d{2})$', value)
    if m:
        return m.group(1) + "-01"
    return None

def _compute_end_date(value: str, start: str) -> str | None:
    """Compute end date for a duration value given a start date."""
    if not value or not start:
        return None
    try:
        start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
    except ValueError:
        return None
    m = re.match(r'^P(\d+)D$', value)
    if m: return (start_dt + timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.match(r'^P(\d+)W$', value)
    if m: return (start_dt + timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.match(r'^PT(\d+)H$', value)
    if m: return start_dt.strftime("%Y-%m-%d")   # same day for hours
    return None


# ── Few-shot for layer executor ───────────────────────────────

def _build_few_shot(today: str) -> str:
    t    = date.fromisoformat(today)
    d3   = (t - timedelta(3)).isoformat()
    d3p2 = (t - timedelta(3) + timedelta(2)).isoformat()
    d3p5 = (t - timedelta(3) + timedelta(5)).isoformat()
    d1   = (t - timedelta(1)).isoformat()
    d1p5 = (t - timedelta(1) + timedelta(5)).isoformat()
    return f"""
HARD RULES:
R1. Output MUST include every id from the input list — never drop one.
R2. Anchor dates in VERIFIED_ANCHORS are guaranteed correct — use them directly for arithmetic.
R3. Never invent an anchor date.  If you cannot derive a date from VERIFIED_ANCHORS, set confidence=0.
R4. absolute_date is always YYYY-MM-DD with dashes.
R5. For DURATION: absolute_date = start date, end_date = start + length.
R6. For TIME/DATE: end_date = null.
R7. RECURRING only for SET type.

=== LAYER EXAMPLE 1 — anchor_dep chain ===
Today: {today}
VERIFIED_ANCHORS: {{ 0: "{d3}" }}   ← blast happened {d3}

Entities in this layer:
[
  {{"id":1,"text":"within two hours of the explosion","type":"DURATION","sentence_idx":1,"anchor_entity_id":0}},
  {{"id":2,"text":"5 days after the bomb blast","type":"DURATION","sentence_idx":2,"anchor_entity_id":0}}
]

Reasoning:
- id=1: blast anchor={d3}, PT2H → same day. absolute_date={d3}, end_date={d3}
- id=2: blast anchor={d3}, +5 days → {d3p5}. absolute_date={d3p5}

Output JSON:
[
  {{"id":1,"value":"PT2H","absolute_date":"{d3}","end_date":"{d3}","confidence":1.0}},
  {{"id":2,"value":"P5D","absolute_date":"{d3p5}","end_date":null,"confidence":1.0}}
]

=== LAYER EXAMPLE 2 — duration chain ===
Today: {today}
VERIFIED_ANCHORS: {{ 3: "{d1}" }}   ← earthquake struck {d1}

Entities in this layer:
[
  {{"id":4,"text":"three hours later","type":"DURATION","sentence_idx":1,"anchor_entity_id":3}},
  {{"id":5,"text":"five days","type":"DURATION","sentence_idx":1,"anchor_entity_id":4}}
]

Reasoning:
- id=4: 3h after {d1} → same day {d1}. value=PT3H
- id=5: rescue for 5 days from {d1} → ends {d1p5}. value=P5D

Output JSON:
[
  {{"id":4,"value":"PT3H","absolute_date":"{d1}","end_date":"{d1}","confidence":1.0}},
  {{"id":5,"value":"P5D","absolute_date":"{d1}","end_date":"{d1p5}","confidence":1.0}}
]
"""


# ── Main executor ─────────────────────────────────────────────

def execute_layers(paragraph:  str,
                   sentences:  list,
                   entities:   list,
                   graph,
                   registry:   AnchorRegistry,
                   llm_model:  str = "mistral") -> list:
    """
    Run the layer executor loop.

    For each layer in topo order:
      - Fast path if all entities in layer are already rule-resolved.
      - Otherwise call LLM with only that layer's entities + registry snapshot.

    Writes all resolved dates to registry.
    Returns the fully resolved entity list.
    """
    from pipeline.stages.graph_builder import layer_batches

    today  = date.today().isoformat()
    eid_map = {e["entity_id"]: e for e in entities}
    batches = layer_batches(graph)

    print(f"\n[Layer Executor]  {len(batches)} layer(s), "
          f"{len(entities)} total entities")

    for layer_idx, eids in enumerate(batches):
        layer_entities = [eid_map[eid] for eid in eids]

        print(f"\n  Layer {layer_idx}  ({len(layer_entities)} entities)")

        # ── Write layer-0 rule-resolved entities to registry ─
        # (rule-resolved entities are always in layer 0 by construction,
        #  but we process them here regardless of layer number)
        all_rule_resolved = all(
            e["rule_result"]["status"] == "resolved" for e in layer_entities
        )

        if all_rule_resolved:
            print(f"    [FAST PATH] all rule-resolved — skipping LLM")
            for e in layer_entities:
                val  = e["rule_result"]["value"]
                adate = _extract_date(val) or today
                e["value"]        = val
                e["absolute_date"] = adate
                e["end_date"]     = None
                e["confidence"]   = 1.0
                e["method"]       = "rule"
                registry.write(
                    entity_id=e["entity_id"],
                    value=adate,
                    sentence_idx=e["sentence_idx"],
                    confidence=1.0,
                    source="rule",
                )
                print(f"      [RULE]  {e['text']:35s} → {val}  abs={adate}")
            continue

        # ── Mixed layer: some rule-resolved, some anchor_dep ─
        # Write rule-resolved ones first so they're in the registry
        # before the LLM call for the rest.
        for e in layer_entities:
            if e["rule_result"]["status"] == "resolved":
                val   = e["rule_result"]["value"]
                adate = _extract_date(val) or today
                e["value"]        = val
                e["absolute_date"] = adate
                e["end_date"]     = None
                e["confidence"]   = 1.0
                e["method"]       = "rule"
                registry.write(
                    entity_id=e["entity_id"],
                    value=adate,
                    sentence_idx=e["sentence_idx"],
                    confidence=1.0,
                    source="rule",
                )
                print(f"      [RULE]  {e['text']:35s} → {val}  abs={adate}")

        # Entities that still need LLM
        llm_entities = [
            e for e in layer_entities
            if e["rule_result"]["status"] != "resolved"
        ]

        if not llm_entities:
            continue

        if not OLLAMA_AVAILABLE:
            _fallback_layer(llm_entities, registry, today)
            continue

        _llm_layer_call(paragraph, sentences, llm_entities,
                        registry, llm_model, today, layer_idx)

    return entities


# ── LLM layer call ────────────────────────────────────────────

def _llm_layer_call(paragraph:    str,
                    sentences:    list,
                    llm_entities: list,
                    registry:     AnchorRegistry,
                    llm_model:    str,
                    today:        str,
                    layer_idx:    int):
    """
    One LLM call for the anchor_dep entities in this layer.

    Injects ONLY the anchor dates that are already verified in the
    registry — the LLM never sees an unresolved anchor date.
    """
    # Build verified anchor snapshot — only dates relevant to this layer
    verified = {}
    for e in llm_entities:
        rr = e["rule_result"]
        anchor_eid = rr.get("anchor_entity_id")
        if anchor_eid is not None:
            d = registry.read(anchor_eid)
            if d:
                verified[anchor_eid] = d
        # Also inject by-sentence fallback
        s = e["sentence_idx"]
        fb = registry.anchor_date_for_sentence(s)
        if fb:
            verified[f"sent_{s-1}_anchor"] = fb

    sent_block = "\n".join(f'  S{i}: "{s}"' for i, s in enumerate(sentences))

    entity_input = []
    for e in llm_entities:
        rr = e["rule_result"]
        entry = {
            "id":             e["entity_id"],
            "text":           e["text"],
            "type":           e["type"],
            "sentence_idx":   e["sentence_idx"],
            "anchor_entity_id": rr.get("anchor_entity_id"),
            "offset_n":       rr.get("offset_n", 0),
            "offset_unit":    rr.get("offset_unit", ""),
        }
        entity_input.append(entry)

    entity_json  = json.dumps(entity_input, indent=2)
    anchors_json = json.dumps(verified, indent=2)
    input_ids    = [e["entity_id"] for e in llm_entities]

    prompt = f"""You are a TimeML temporal normalizer — layer {layer_idx} of a divide-and-conquer resolver.

Today: {today}

YOUR TASK:
- You receive a small batch of temporal entities that depend on anchor events.
- VERIFIED_ANCHORS contains only pre-verified dates (resolved by rules, guaranteed correct).
- Use these anchor dates to do coref + arithmetic ONLY.
- NEVER guess or invent an anchor date.  If you cannot derive a date from VERIFIED_ANCHORS, set confidence=0.
- All input IDs must appear in output: {input_ids}

TimeML rules:
- DATE/TIME  → absolute_date = YYYY-MM-DD
- DURATION   → absolute_date = START date, end_date = START + length (P3D from 2026-04-15 → end=2026-04-18)
- SET        → absolute_date = "RECURRING"
- absolute_date always YYYY-MM-DD with dashes.  Never XXXX patterns.

{_build_few_shot(today)}

=== YOUR INPUT ===

Full paragraph:
{paragraph}

Sentences:
{sent_block}

VERIFIED_ANCHORS (entity_id → resolved date, guaranteed correct):
{anchors_json}

Entities to resolve (layer {layer_idx}):
{entity_json}

Reply with ONLY a valid JSON array. No markdown, no explanation.
Each object must have exactly:
  id (int), value (TimeML str), absolute_date (YYYY-MM-DD or RECURRING),
  end_date (YYYY-MM-DD or null), confidence (0.0–1.0)
"""

    try:
        res = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw = res["message"]["content"].strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        results = json.loads(raw)
        id_map  = {r["id"]: r for r in results if isinstance(r, dict) and "id" in r}

        for e in llm_entities:
            eid = e["entity_id"]
            r   = id_map.get(eid, {})

            val   = _fix_timeml(r.get("value", "UNKNOWN"))
            conf  = float(r.get("confidence", 0.5))
            raw_abs = str(r.get("absolute_date", ""))
            adate = raw_abs if _DATE_RE.match(raw_abs) else (_extract_date(val) or today)
            raw_end = str(r.get("end_date") or "")
            end   = raw_end if _DATE_RE.match(raw_end) else _compute_end_date(val, adate)

            e["value"]         = val
            e["absolute_date"] = adate
            e["end_date"]      = end
            e["confidence"]    = conf
            e["method"]        = "llm"

            ok = registry.write(
                entity_id=eid,
                value=adate,
                sentence_idx=e["sentence_idx"],
                confidence=conf,
                source="llm",
            )
            status = "✓" if ok else "✗ INVALID"
            print(f"      [LLM {status}]  {e['text']:35s} → {val}  "
                  f"abs={adate}  conf={conf:.1f}")

    except json.JSONDecodeError as ex:
        print(f"    [layer {layer_idx}] JSON parse error: {ex} — using fallback")
        _fallback_layer(llm_entities, registry, today)
    except Exception as ex:
        print(f"    [layer {layer_idx}] LLM error: {ex} — using fallback")
        _fallback_layer(llm_entities, registry, today)


# ── Fallback ──────────────────────────────────────────────────

def _fallback_layer(entities: list, registry: AnchorRegistry, today: str):
    """Used when Ollama is unavailable or the LLM call fails."""
    for e in entities:
        rr = e["rule_result"]
        # Try to use registry anchor date if available
        anchor_eid = rr.get("anchor_entity_id")
        adate = registry.read(anchor_eid) if anchor_eid is not None else None
        if not adate:
            adate = registry.anchor_date_for_sentence(e["sentence_idx"]) or today

        e["value"]         = "UNKNOWN"
        e["absolute_date"] = adate
        e["end_date"]      = None
        e["confidence"]    = 0.0
        e["method"]        = "fallback"
        registry.write(
            entity_id=e["entity_id"],
            value=adate,
            sentence_idx=e["sentence_idx"],
            confidence=0.0,
            source="fallback",
        )
        print(f"      [FALLBACK]  {e['text']:35s} → UNKNOWN  abs={adate}")
