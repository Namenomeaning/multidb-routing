"""Generate LLM domain summaries for each DB instance. Cache to augmented-schemas.jsonl."""

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.openrouter import openrouter_chat

MULTI_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "multi"
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
OUTPUT_PATH = MULTI_DIR / "augmented-schemas.jsonl"


def main():
    prompt_tpl = (PROMPT_DIR / "augment-schema.txt").read_text()

    with open(MULTI_DIR / "instances.jsonl") as f:
        instances = [json.loads(line) for line in f]

    existing = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["instance_id"]] = rec
        print(f"Loaded {len(existing)} cached augmentations")

    results = []
    new_calls = 0

    for inst in tqdm(instances, desc="Augmenting"):
        iid = inst["instance_id"]

        if iid in existing:
            results.append(existing[iid])
            continue

        prompt = prompt_tpl.format(
            instance_id=iid,
            engine=inst["engine"],
            schema_text=inst["schema_text"][:3000],
        )

        raw = openrouter_chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        new_calls += 1

        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
        except json.JSONDecodeError:
            data = {"domain_description": raw[:500], "domain_tags": ["other"], "key_entities": []}

        aug_text = inst["schema_text"]
        aug_text += f"\n\nDomain: {data.get('domain_description', '')}"
        entities = data.get("key_entities", [])
        if entities:
            aug_text += f"\nKey entities: {', '.join(entities)}"

        rec = {
            "instance_id": iid,
            "engine": inst["engine"],
            "db_id": inst["db_id"],
            "domain_description": data.get("domain_description", ""),
            "domain_tags": data.get("domain_tags", []),
            "key_entities": entities,
            "augmented_schema_text": aug_text,
        }
        results.append(rec)

        if new_calls % 10 == 0:
            _save(results)

    _save(results)
    print(f"\nDone. {new_calls} new LLM calls. {len(results)} total instances.")


def _save(results):
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
