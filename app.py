# app.py — interactive terminal  (Arch 2 — universal rule-first)
import argparse
import json
from core.predict      import load_model
from pipeline.pipeline import run_pipeline, DEMOS

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   News Temporal Extraction + Summarization  (Arch 2)             ║
║                                                                  ║
║   Universal rule pass on every entity (Stage 2)                  ║
║   LLM does coref + arithmetic on verified anchors only           ║
║   No anchor date guessing — ever                                  ║
╚══════════════════════════════════════════════════════════════════╝
Commands:
  <paragraph>   run full pipeline
  demo 0-4      built-in demo examples
  json          toggle JSON output mode
  quit          exit
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tempeval3_ner_final.pt")
    parser.add_argument("--llm",   default="mistral")
    args = parser.parse_args()

    print(BANNER)
    print(f"  NER model : {args.model}")
    print(f"  LLM       : {args.llm}\n")

    load_model(args.model)
    json_mode = False

    while True:
        try:
            user_input = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!"); break

        if not user_input:
            continue
        low = user_input.lower()

        if low in ("quit", "exit", "q"):
            print("Bye!"); break

        elif low == "json":
            json_mode = not json_mode
            print(f"  JSON mode: {'ON' if json_mode else 'OFF'}")

        elif low.startswith("demo"):
            parts = low.split()
            idx = int(parts[1]) % len(DEMOS) if len(parts) > 1 and parts[1].isdigit() else 0
            r = run_pipeline(DEMOS[idx][1], llm_model=args.llm)
            if json_mode:
                print(json.dumps(
                    {k: v for k, v in r.items()
                     if k in ("entities", "summary", "timeline")}, indent=2))
        else:
            r = run_pipeline(user_input, llm_model=args.llm)
            if json_mode:
                print(json.dumps(
                    {k: v for k, v in r.items()
                     if k in ("entities", "summary", "timeline")}, indent=2))


if __name__ == "__main__":
    main()
