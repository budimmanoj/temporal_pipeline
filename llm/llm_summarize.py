# llm_summarize.py
#
# STAGE 4 — News Summarizer
#
# Compact, news-wire style bullet summary ordered chronologically.
# RECURRING (SET) entities listed at end as schedule notes.
#
# Runaway-generation protection:
#   - max_tokens=400 on LLM call
#   - Post-process: extract only lines starting with "-"
#   - If zero bullet lines found → fallback

import re
import json
from datetime import date

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def _extract_bullets(text: str) -> str:
    """
    Extract only genuine news bullet lines from LLM output.
    Rejects instruction-style runaway lines.
    """
    INSTRUCTION_PREFIXES = re.compile(
        r'^\s*[-•]\s*(Identify|Include|Exclude|Do NOT|Do not|Ensure|Avoid|'
        r'Incor|Note|List|Use only|Return|Output|Format|Write|Generate|'
        r'Provide|Make sure|Remember|Each bullet)\b',
        re.IGNORECASE
    )
    GARBLED_INSTRUCTION = re.compile(r'^\s*[-•]\s*[A-Za-z]{2,6}\d', re.IGNORECASE)

    lines = text.splitlines()
    bullets = []
    for l in lines:
        if not re.match(r'^\s*[-•]', l):
            continue
        if INSTRUCTION_PREFIXES.match(l):
            continue
        if GARBLED_INSTRUCTION.match(l):
            continue
        bullets.append(l.rstrip())
    return "\n".join(bullets)


FEW_SHOT_SUMMARIZE = """
=== SUMMARY EXAMPLE 1 ===
Paragraph: "A major fire broke out in the warehouse three days ago. Rescue teams arrived two hours later. The investigation began the following morning and lasted four days."

Timeline:
[
  {"sentence":"A major fire broke out in the warehouse three days ago.","absolute_date":"2026-04-15"},
  {"sentence":"Rescue teams arrived two hours later.","absolute_date":"2026-04-15","end_date":"2026-04-15"},
  {"sentence":"The investigation began the following morning and lasted four days.","absolute_date":"2026-04-16","end_date":"2026-04-20"}
]

Output (ONLY bullet lines — nothing else):
- Major fire broke out in warehouse (2026-04-15).
- Rescue teams arrived two hours later.
- Investigation began the following morning (2026-04-16), concluded 2026-04-20.

=== SUMMARY EXAMPLE 2 ===
Paragraph: "The event starts tomorrow morning at 9 AM and continues for three days."

Timeline:
[
  {"sentence":"The event starts tomorrow morning at 9 AM and continues for three days.","absolute_date":"2026-04-19","end_date":"2026-04-22"}
]

Output:
- Event starts tomorrow morning at 9 AM (2026-04-19), runs until 2026-04-22.

=== SUMMARY EXAMPLE 3 ===
Paragraph: "The earthquake struck yesterday at midnight. Rescue operations started three hours later and continued for five days."

Timeline:
[
  {"sentence":"The earthquake struck yesterday at midnight.","absolute_date":"2026-04-17"},
  {"sentence":"Rescue operations started three hours later and continued for five days.","absolute_date":"2026-04-17","end_date":"2026-04-22"}
]

Output:
- Earthquake struck yesterday at midnight (2026-04-17).
- Rescue operations started three hours later, continued until 2026-04-22.

=== SUMMARY EXAMPLE 4 ===
Paragraph: "The system runs every Monday morning and performs backup for two hours."

Timeline:
[
  {"sentence":"The system runs every Monday morning and performs backup for two hours.","absolute_date":"RECURRING"}
]

Output:
- System runs every Monday morning with a two-hour backup (recurring schedule).

=== SUMMARY EXAMPLE 5 ===
Paragraph: "The conference started last Monday. Two days later, a workshop was conducted. After another three days, the final report was submitted."

Timeline:
[
  {"sentence":"The conference started last Monday.","absolute_date":"2026-04-13"},
  {"sentence":"Two days later, a workshop was conducted.","absolute_date":"2026-04-15"},
  {"sentence":"After another three days, the final report was submitted.","absolute_date":"2026-04-18"}
]

Output:
- Conference started last Monday (2026-04-13).
- Workshop conducted two days later (2026-04-15).
- Final report submitted three days after workshop (2026-04-18).
"""


def summarize(paragraph: str,
              sentences: list,
              normalized_entities: list,
              llm_model: str = "mistral") -> dict:
    today = date.today().isoformat()

    # Separate recurring vs dated
    recurring = [e for e in normalized_entities if e.get("absolute_date") == "RECURRING"]
    dated     = [e for e in normalized_entities if e.get("absolute_date") != "RECURRING"]
    dated_sorted = sorted(dated, key=lambda e: e.get("absolute_date", today) or today)
    all_sorted   = dated_sorted + recurring

    # Group by sentence, preserving chronological order
    seen = {}
    for e in all_sorted:
        sidx = e.get("sentence_idx", 0)
        if sidx not in seen:
            sent = sentences[sidx] if sidx < len(sentences) else ""
            seen[sidx] = {
                "sentence"     : sent,
                "absolute_date": e.get("absolute_date", today),
                "entities"     : []
            }
        seen[sidx]["entities"].append({
            "text"    : e.get("text", ""),
            "value"   : e.get("value", ""),
            "end_date": e.get("end_date"),
        })

    timeline_input = [seen[k] for k in sorted(seen)]

    # Compact timeline for LLM (sentence + start + end for durations)
    compact_timeline = []
    for item in timeline_input:
        entry = {
            "sentence"      : item["sentence"],
            "absolute_date" : item["absolute_date"],
        }
        # Add end_date if any entity in this sentence has one
        ends = [e.get("end_date") for e in item.get("entities", []) if e.get("end_date")]
        if ends:
            entry["end_date"] = max(ends)  # latest end date for the sentence
        compact_timeline.append(entry)
    timeline_json = json.dumps(compact_timeline, indent=2)

    if not OLLAMA_AVAILABLE:
        return {"summary": _fallback_summary(timeline_input), "timeline": all_sorted}

    prompt = f"""You are a news wire summarizer. Today: {today}

RULES:
1. One bullet per sentence. Group all events from the same sentence into ONE bullet.
2. Show start date in parentheses ONLY when original text was vague ("three days ago", "the following morning").
3. If timeline entry has end_date, append "until YYYY-MM-DD" or "concluded YYYY-MM-DD" at the end of the bullet.
4. For same-day durations (PT2H, PT3H) keep original wording — no end_date annotation needed.
5. Chronological order, oldest first. RECURRING entries go last.
6. ONLY output bullet lines starting with "-". NO headers, NO explanations, NO extra text.
7. STOP immediately after the last bullet.

{FEW_SHOT_SUMMARIZE}

=== NOW SUMMARIZE ===

Today: {today}

Paragraph:
{paragraph}

Timeline:
{timeline_json}

Output ONLY bullet lines starting with "-" (then STOP):"""

    try:
        res = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 200}
        )
        raw_output = res["message"]["content"].strip()
        bullets    = _extract_bullets(raw_output)

        if not bullets:
            print("[summarize] No bullet lines found in LLM output — using fallback.")
            bullets = _fallback_summary(timeline_input)

        return {"summary": bullets, "timeline": all_sorted}

    except Exception as ex:
        print(f"[summarize] LLM error: {ex}. Returning fallback summary.")
        return {"summary": _fallback_summary(timeline_input), "timeline": all_sorted}


def _fallback_summary(timeline_input: list) -> str:
    lines = []
    for item in timeline_input:
        sent     = item.get("sentence", "").strip()
        abs_date = item.get("absolute_date", "")
        if abs_date == "RECURRING":
            lines.append(f"- {sent} (recurring)")
            continue
        # Show date hint only for vague expressions
        date_hint = ""
        for ent in item.get("entities", []):
            txt = ent.get("text", "")
            if abs_date and not re.search(r'\d{4}', txt):
                date_hint = f" ({abs_date})"
                break
        lines.append(f"- {sent}{date_hint}")
    return "\n".join(lines)