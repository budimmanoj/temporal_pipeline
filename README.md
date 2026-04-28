# News Temporal Extraction + Summarization  (Arch 2)

A modular, rule-first pipeline for extracting and normalizing temporal expressions from news text. Every entity passes through a deterministic rule engine before any LLM call is made. The LLM only performs coreference resolution and date arithmetic on **verified** anchor dates — it never guesses.

---

## Project Structure

```
temporal_pipeline/          ← project root (run all commands from here)
├── app.py                  ← interactive terminal entry point
├── requirements.txt
├── README.md
│
├── core/                   ← NER model + shared state
│   ├── __init__.py
│   ├── model.py            ← BERT + BiLSTM + CRF architecture
│   ├── predict.py          ← inference: load_model, predict_sentence, extract_entities
│   └── anchor_registry.py  ← shared mutable registry (entity_id → YYYY-MM-DD)
│
├── pipeline/               ← orchestration
│   ├── __init__.py
│   ├── pipeline.py         ← run_pipeline(), all stage functions, DEMOS
│   └── stages/             ← one file per pipeline stage
│       ├── __init__.py
│       ├── rule_normalize.py    ← Stage 2: TimeML rule engine
│       ├── graph_builder.py     ← Stage 3+4: DAG construction + topological sort
│       ├── layer_executor.py    ← Stage 5: per-layer LLM / fast-path loop
│       └── conflict_resolver.py ← Stage 6: merge, dedup, conflict flagging
│
└── llm/                    ← LLM-specific modules
    ├── __init__.py
    ├── llm_ner_validator.py ← Stage 1.5: NER gap recovery via LLM
    └── llm_summarize.py     ← Stage 7: chronological bullet summary
```

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `core/predict.py` | BERT+BiLSTM+CRF NER — extracts raw temporal spans |
| 1.5 | `llm/llm_ner_validator.py` | LLM cross-check for missed/partial spans |
| 2 | `pipeline/stages/rule_normalize.py` | Universal TimeML rule engine — every entity |
| 3 | `pipeline/stages/graph_builder.py` | Build dependency DAG, detect cycles |
| 4 | `pipeline/stages/graph_builder.py` | Topological sort → layer assignments |
| 5 | `pipeline/stages/layer_executor.py` | Per-layer: fast path (rule) or LLM arithmetic |
| 6 | `pipeline/stages/conflict_resolver.py` | Merge, deduplicate, flag conflicts |
| 7 | `llm/llm_summarize.py` | Chronological bullet summary |

### Key Design Principles

- **Rule-first**: every entity goes through `rule_normalize()` before any LLM is called.
- **No anchor guessing**: the LLM receives only dates that are already verified in the `AnchorRegistry`.
- **Layer isolation**: each dependency layer is resolved sequentially; earlier layers write to the registry before later layers read from it.
- **Graceful degradation**: all LLM stages fall back cleanly if Ollama is unavailable.

---

## Installation

```bash
pip install -r requirements.txt
```

Ollama is required for LLM stages (Stage 1.5, 5, and 7). Install from [ollama.ai](https://ollama.ai) and pull a model:

```bash
ollama pull mistral
```

A trained NER checkpoint (`tempeval3_ner_final.pt`) must be present in the project root.

---

## Usage

### Interactive terminal

```bash
python app.py
python app.py --model tempeval3_ner_final.pt --llm mistral
```

**Commands inside the terminal:**

| Command | Action |
|---------|--------|
| `<paragraph>` | Run the full pipeline on typed text |
| `demo 0` – `demo 4` | Run a built-in demo paragraph |
| `json` | Toggle JSON output mode on/off |
| `quit` | Exit |

### Run a single paragraph from the command line

```bash
python -m pipeline.pipeline --text "The earthquake struck yesterday. Rescue teams arrived three hours later."
python -m pipeline.pipeline --text "..." --llm mistral --json
```

### Run a built-in demo

```bash
python -m pipeline.pipeline --demo 0   # blast → rescue → investigation
python -m pipeline.pipeline --demo 1   # wildfire
python -m pipeline.pipeline --demo 2   # political crisis
python -m pipeline.pipeline --demo 3   # flood
python -m pipeline.pipeline --demo 4   # warehouse fire
```

### Use as a library

```python
from core.predict      import load_model
from pipeline.pipeline import run_pipeline

load_model("tempeval3_ner_final.pt")

result = run_pipeline(
    "The wildfire started Monday. Firefighters arrived three hours later.",
    llm_model="mistral",
)

print(result["summary"])
for entity in result["entities"]:
    print(entity["text"], "→", entity["absolute_date"])
```

---

## Output Format

`run_pipeline()` returns a dict with:

```python
{
    "paragraph":  str,          # original input
    "sentences":  list[str],    # split sentences
    "entities":   list[dict],   # resolved temporal entities
    "summary":    str,          # chronological bullet summary
    "timeline":   list[dict],   # entities sorted by absolute_date
}
```

Each entity dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | int | Unique identifier |
| `text` | str | Original surface form |
| `type` | str | `DATE`, `TIME`, `DURATION`, `SET`, `EVENT` |
| `absolute_date` | str | `YYYY-MM-DD` or `RECURRING` |
| `end_date` | str\|None | End of duration, or `None` |
| `value` | str | TimeML normalized value |
| `confidence` | float | 0.0 – 1.0 |
| `method` | str | `rule`, `llm`, or `fallback` |
| `conflict_flag` | bool | `True` if conflicting dates were found |

---

## Built-in Demos

| Index | Title |
|-------|-------|
| 0 | Cross-dependency: blast → rescue → investigation |
| 1 | Wildfire: sequential events |
| 2 | Political crisis: mixed rule + cross-dep |
| 3 | Flood: dense temporal chain |
| 4 | Warehouse fire: full chain |

---

## Module Reference

### `core.predict`
- `load_model(path)` — load `.pt` checkpoint
- `predict_sentence(text)` — returns `[(token, label), ...]`
- `extract_entities(token_labels)` — groups BIO tags into entity dicts

### `core.anchor_registry.AnchorRegistry`
- `write(entity_id, value, sentence_idx, confidence, source)` — validated write
- `read(entity_id)` — returns resolved date or `None`
- `anchor_date_for_sentence(sentence_idx)` — nearest resolved date before sentence N

### `pipeline.pipeline`
- `run_pipeline(paragraph, llm_model)` — full 7-stage pipeline
- `DEMOS` — list of `(title, paragraph)` tuples

### `pipeline.stages.rule_normalize`
- `rule_normalize(text, entity_type)` — returns `{"status": ..., "value": ...}`

### `pipeline.stages.graph_builder`
- `build_graph(entities)` → `DependencyGraph`
- `layer_batches(graph)` → `list[list[entity_id]]`

### `pipeline.stages.layer_executor`
- `execute_layers(paragraph, sentences, entities, graph, registry, llm_model)` → resolved entities

### `pipeline.stages.conflict_resolver`
- `resolve_conflicts(entities)` → deduplicated, conflict-flagged entity list

### `llm.llm_ner_validator`
- `validate_ner_output(sentences, raw_entities, llm_model)` → augmented entity list

### `llm.llm_summarize`
- `summarize(paragraph, sentences, normalized_entities, llm_model)` → `{"summary": str, "timeline": list}`
