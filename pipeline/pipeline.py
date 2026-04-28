# pipeline.py  — Arch 2: Universal rule-first divide-and-conquer
#
# ┌──────────────────────────────────────────────────────────────────┐
# │  Stage 1 — NER  (BERT + BiLSTM + CRF)                           │
# │    sentence-by-sentence extraction → raw entity list             │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 1.5 — LLM NER Validator                                   │
# │    Checks every sentence for missed/partial temporal spans.      │
# │    Injects full_miss entities; fixes partial_span strips.        │
# │    Only fires for sentences with cross-dep hints or count gap.   │
# │    Falls back gracefully if Ollama unavailable.                  │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 2 — Universal rule engine                                 │
# │    EVERY entity goes through rule_normalize().                   │
# │    Returns one of:                                               │
# │      resolved   → TimeML value, no LLM needed                   │
# │      anchor_dep → structured tag {anchor_tag, offset_n, unit}   │
# │      vague      → truly ambiguous, LLM gets it with conf=0      │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 3 — Graph builder                                         │
# │    Builds DAG from anchor_dep edges.                             │
# │    Detects cycles before any LLM call is made.                  │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 4 — Topological sort                                      │
# │    Assigns layer numbers. Groups into LayerBatch lists.          │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 5 — Layer executor  ↻                                     │
# │    For each layer in order:                                      │
# │      Fast path: skip LLM if all rule-resolved.                   │
# │      LLM path: coref + arithmetic ONLY on verified anchor dates. │
# │      LLM never sees an unresolved anchor date.                   │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 6 — Conflict resolver                                     │
# │    Merge, deduplicate, flag conflicts by entity_id.              │
# ├──────────────────────────────────────────────────────────────────┤
# │  Stage 7 — Summariser                                            │
# │    Sort by date, emit final chronological summary.               │
# └──────────────────────────────────────────────────────────────────┘

import re
import json
import argparse
from datetime import date

from core.predict             import load_model, predict_sentence, extract_entities
from pipeline.stages.rule_normalize    import rule_normalize, _n as _word_to_n
from pipeline.stages.graph_builder     import build_graph, layer_batches
from core.anchor_registry     import AnchorRegistry
from pipeline.stages.layer_executor    import execute_layers
from pipeline.stages.conflict_resolver import resolve_conflicts
from llm.llm_summarize        import summarize
from llm.llm_ner_validator    import validate_ner_output


# ── Sentence-context cross-dep detector ──────────────────────

_CTX_TRAILING = re.compile(
    r'\b(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten'
    r'|few|several|couple|another)\s+'
    r'(?P<u>hour|day|week|month|year)s?\s+'
    r'(?:later|after\b|before\b|following|since\b|of the)',
    re.IGNORECASE,
)

_CTX_LEADING = re.compile(
    r'\b(?:after|within|before|following)\s+(?:another\s+)?'
    r'(?P<n>another|\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten'
    r'|few|several|couple)\s+'
    r'(?P<u>hour|day|week|month|year)s?\b',
    re.IGNORECASE,
)


def _sentence_cross_dep(span: str, sentence: str) -> dict | None:
    for pat in (_CTX_TRAILING, _CTX_LEADING):
        m = pat.search(sentence)
        if m:
            n_str = m.group("n").lower()
            if n_str == "another":
                n_val = 1.0
            else:
                n_val = _word_to_n(n_str)
            return {
                "status":      "anchor_dep",
                "anchor_tag":  "event",
                "offset_n":    n_val,
                "offset_unit": m.group("u").lower(),
            }
    return None


# ── Utilities ─────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Stage 1: NER ──────────────────────────────────────────────

def stage1_ner(sentences: list) -> list:
    raw = []
    for idx, sent in enumerate(sentences):
        for ent in extract_entities(predict_sentence(sent)):
            raw.append({
                "sentence_idx": idx,
                "text":         ent["text"],
                "type":         ent["type"],
            })
    print(f"\n[Stage 1 — NER]  {len(raw)} raw entities across {len(sentences)} sentences")
    for e in raw:
        print(f"   S{e['sentence_idx']}  [{e['type']:8s}]  {e['text']}")
    return raw


# ── Stage 1.5: LLM NER Validator ─────────────────────────────

def stage1_5_validate(sentences: list, raw: list, llm_model: str) -> list:
    return validate_ner_output(sentences, raw, llm_model=llm_model)


# ── Stage 2: Universal rule engine ───────────────────────────

def stage2_rule_pass(raw_entities: list, sentences: list) -> list:
    print(f"\n[Stage 2 — Universal Rule Engine]")
    entities = []
    for i, e in enumerate(raw_entities):
        rr = rule_normalize(e["text"], e["type"])

        if rr["status"] == "resolved" and e["type"] in ("DURATION", "DATE"):
            sent = sentences[e["sentence_idx"]] if e["sentence_idx"] < len(sentences) else ""
            ctx = _sentence_cross_dep(e["text"], sent)
            if ctx:
                rr = ctx
        entity = {
            "entity_id":   i,
            "sentence_idx": e["sentence_idx"],
            "text":        e["text"],
            "type":        e["type"],
            "rule_result": rr,
            "layer":       0,
            "value":       None,
            "absolute_date": None,
            "end_date":    None,
            "confidence":  0.0,
            "method":      None,
            "conflict_flag": False,
        }
        entities.append(entity)
        status = rr["status"]
        detail = rr.get("value", "") if status == "resolved" else \
                 f"anchor={rr.get('anchor_tag','?')} +{rr.get('offset_n',0)}{rr.get('offset_unit','')}" \
                 if status == "anchor_dep" else "vague"
        print(f"   S{e['sentence_idx']}  [{e['type']:8s}]  {e['text']:35s}  "
              f"[{status}]  {detail}")
    return entities


# ── Stage 3: Graph builder ────────────────────────────────────

def stage3_graph(entities: list):
    print(f"\n[Stage 3 — Graph Builder]")
    g = build_graph(entities)
    if not g.cycle_free:
        print(f"  ⚠  Cycle detected in nodes: {g.cycle_nodes}")
        print(f"  Cycle nodes assigned to layer 999 — will be treated as vague.")
    else:
        print(f"  DAG is cycle-free.")
    print(f"  Layer distribution: { {l: sum(1 for e in entities if e['layer']==l) for l in g.topo_order} }")
    return g


# ── Stage 4: Topo sort ────────────────────────────────────────

def stage4_topo_sort(g) -> list:
    batches = layer_batches(g)
    print(f"\n[Stage 4 — Topological Sort]  {len(batches)} layer(s)")
    for i, batch in enumerate(batches):
        print(f"  Layer {i}: entity_ids={batch}")
    return batches


# ── Stage 5: Layer executor ───────────────────────────────────

def stage5_execute(paragraph: str,
                   sentences: list,
                   entities:  list,
                   graph,
                   llm_model: str) -> tuple:
    registry = AnchorRegistry()
    resolved = execute_layers(paragraph, sentences, entities, graph,
                              registry, llm_model)
    return resolved, registry


# ── Stage 6: Conflict resolver ────────────────────────────────

def stage6_conflicts(entities: list) -> list:
    return resolve_conflicts(entities)


# ── Stage 7: Summariser ───────────────────────────────────────

def stage7_summarize(paragraph: str,
                     sentences: list,
                     entities:  list,
                     llm_model: str) -> dict:
    print(f"\n[Stage 7 — Summariser]  Generating chronological summary...")
    return summarize(paragraph, sentences, entities, llm_model)


# ── Pretty-print resolved entities ───────────────────────────

def _print_resolved(entities: list):
    print(f"\n[Resolved Entities]")
    print(f"  {'S':<4} {'TYPE':<10} {'M':<8} {'TEXT':<35} {'VALUE':<24} {'DATE':<12} {'END'}")
    print(f"  {'─'*4} {'─'*10} {'─'*8} {'─'*35} {'─'*24} {'─'*12} {'─'*10}")
    for e in entities:
        sidx   = e.get("sentence_idx", "?")
        etype  = e.get("type", "?")
        method = e.get("method", "?")
        text   = e.get("text", "")[:34]
        value  = e.get("value") or "?"
        start  = e.get("absolute_date") or "?"
        end    = e.get("end_date") or "—"
        flag   = " ⚠" if e.get("conflict_flag") else ""
        print(f"  S{sidx:<3} {etype:<10} {method:<8} {text:<35} {value:<24} {start:<12} {end}{flag}")


# ── Full pipeline ─────────────────────────────────────────────

def run_pipeline(paragraph: str, llm_model: str = "mistral") -> dict:
    print(f"\n{'═'*65}")
    print(f"  PARAGRAPH")
    print(f"{'═'*65}")
    print(f"  {paragraph[:200]}{'...' if len(paragraph) > 200 else ''}")

    sentences = split_sentences(paragraph)

    raw = stage1_ner(sentences)
    if not raw:
        print("\n[Pipeline] No temporal entities found by NER.")
        return {"paragraph": paragraph, "sentences": sentences,
                "entities": [], "summary": "No temporal expressions found.",
                "timeline": []}

    raw = stage1_5_validate(sentences, raw, llm_model)
    entities = stage2_rule_pass(raw, sentences)
    graph = stage3_graph(entities)
    stage4_topo_sort(graph)
    entities, registry = stage5_execute(paragraph, sentences, entities, graph, llm_model)
    entities = stage6_conflicts(entities)

    if not entities:
        print("\n[Pipeline] No entities after conflict resolution.")
        return {"paragraph": paragraph, "sentences": sentences,
                "entities": [], "summary": "No temporal expressions resolved.",
                "timeline": []}

    _print_resolved(entities)
    result = stage7_summarize(paragraph, sentences, entities, llm_model)

    print(f"\n{'═'*65}")
    print(f"  NEWS SUMMARY  (chronological)")
    print(f"{'═'*65}")
    print(result["summary"])
    print(f"{'═'*65}\n")

    return {
        "paragraph": paragraph,
        "sentences": sentences,
        "entities":  entities,
        "summary":   result["summary"],
        "timeline":  result["timeline"],
    }


# ── Demo paragraphs ───────────────────────────────────────────

DEMOS = [
    (
        "Cross-dependency: blast → rescue → investigation",
        "A bomb blast occurred three days ago. Rescue workers were deployed within "
        "two hours of the explosion. The investigation will start after 5 days of "
        "the bomb blast. Three suspects have been detained so far.",
    ),
    (
        "Wildfire: sequential events",
        "The wildfire started on Monday morning near the northern district. "
        "Firefighters arrived three hours later. By Wednesday the fire had spread "
        "over 500 acres. It was fully contained five days after it began. "
        "Restoration work is expected to last at least six months.",
    ),
    (
        "Political crisis: mixed rule + cross-dep",
        "Protests erupted last Tuesday after the election results were announced. "
        "Three days of unrest followed. The government declared a curfew at midnight "
        "on Thursday. International mediators arrived 48 hours after the curfew. "
        "Talks are expected to conclude within the next two weeks.",
    ),
    (
        "Flood: dense temporal chain",
        "Heavy rainfall began on Sunday evening and continued for 36 hours. "
        "By Monday night, 200 families had been evacuated. The water level started "
        "receding two days after the rain stopped. Full restoration is expected to "
        "take at least three weeks.",
    ),
    (
        "Warehouse fire: full chain",
        "A major fire broke out in the warehouse three days ago. Rescue teams arrived "
        "two hours later and controlled the situation by midnight. The investigation "
        "began the following morning and continued for four days. A detailed report "
        "was released two days after the investigation ended.",
    ),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="News Temporal Extraction Pipeline — Arch 2 (Universal rule-first)"
    )
    parser.add_argument("--text",  type=str, default=None)
    parser.add_argument("--model", type=str, default="tempeval3_ner_final.pt")
    parser.add_argument("--llm",   type=str, default="mistral")
    parser.add_argument("--json",  action="store_true")
    parser.add_argument("--demo",  type=int, default=None, help="0-4: run a specific demo")
    args = parser.parse_args()

    load_model(args.model)

    if args.text:
        r = run_pipeline(args.text, llm_model=args.llm)
        if args.json:
            print(json.dumps(
                {k: v for k, v in r.items() if k != "sentences"}, indent=2))

    elif args.demo is not None:
        idx = args.demo % len(DEMOS)
        title, para = DEMOS[idx]
        print(f"\n{'#'*65}\n  DEMO {idx}: {title}\n{'#'*65}")
        run_pipeline(para, llm_model=args.llm)

    else:
        for i, (title, para) in enumerate(DEMOS):
            print(f"\n\n{'#'*65}\n  DEMO {i}: {title}\n{'#'*65}")
            run_pipeline(para, llm_model=args.llm)
