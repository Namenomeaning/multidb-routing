"""E3c decisive table: clean vs obfuscated, raw vs card, on the SAME query slice.

Embeds the stratified query slice ONCE, ranks against all 4 index matrices
(raw-clean, card-clean, raw-obf, card-obf). Queries are CLEAN in both conditions
(they use real domain words; only the INDEXED schema/card is obfuscated).

Leakage gate (BENCHMARK.md §5; memory semantic-beats-raw-obfuscated):
  Δ_clean = card_clean − raw_clean   (the gain we want to defend)
  Δ_obf   = card_obf  − raw_obf       (does the advantage survive name masking?)
  raw drop vs card drop               (raw should collapse harder if it rode on names)
Gain is real query-time enrichment iff Δ_obf stays > 0 and significant.

Usage:
  python e3c_eval.py --clean <route_dir> --obf <route_dir_obf> [--per-db-cap 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval_eval import load, stratified, embed_all, ranks_for, metrics, mcnemar  # noqa: E402

KEYS = ["R@1", "R@5", "mAP", "DBmacro_R@1", "DBmacro_R@5"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--obf", required=True)
    ap.add_argument("--per-db-cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=260611)
    args = ap.parse_args()

    clean, obf = Path(args.clean), Path(args.obf)
    man = json.loads((clean / "index" / "manifest.json").read_text())
    man_o = json.loads((obf / "index" / "manifest.json").read_text())
    assert man["instance_ids"] == man_o["instance_ids"], "clean/obf index id order mismatch"
    ids = man["instance_ids"]

    rc = np.load(clean / "index" / "raw.npy");  cc = np.load(clean / "index" / "card.npy")
    ro = np.load(obf / "index" / "raw.npy");     co = np.load(obf / "index" / "card.npy")

    qs = stratified(load(clean / "splits" / "test.jsonl"), args.per_db_cap, args.seed)
    gts = [q["instance_id"] for q in qs]
    print(f"{clean.name}: {len(ids)} DBs, {len(qs)} queries (cap {args.per_db_cap}/DB), embed={man['embedding_model']}")
    qv = embed_all([q["question"] for q in qs])

    R = {
        "raw_clean":  ranks_for(qv, rc, ids, gts),
        "card_clean": ranks_for(qv, cc, ids, gts),
        "raw_obf":    ranks_for(qv, ro, ids, gts),
        "card_obf":   ranks_for(qv, co, ids, gts),
    }
    M = {k: metrics(v, gts) for k, v in R.items()}

    print(f"\n{'metric':14s} {'raw_clean':>10s} {'card_clean':>11s} {'raw_obf':>10s} {'card_obf':>10s}")
    for k in KEYS:
        print(f"  {k:12s} {M['raw_clean'][k]:>10.4f} {M['card_clean'][k]:>11.4f} {M['raw_obf'][k]:>10.4f} {M['card_obf'][k]:>10.4f}")

    print("\n-- leakage gate (headline metric per set) --")
    for k in ["R@5", "DBmacro_R@5", "R@1"]:
        d_clean = M["card_clean"][k] - M["raw_clean"][k]
        d_obf = M["card_obf"][k] - M["raw_obf"][k]
        raw_drop = M["raw_clean"][k] - M["raw_obf"][k]
        card_drop = M["card_clean"][k] - M["card_obf"][k]
        print(f"  {k:12s} Δ_clean={d_clean:+.4f}  Δ_obf={d_obf:+.4f}  | raw_drop={raw_drop:+.4f}  card_drop={card_drop:+.4f}")

    print(f"\nMcNemar card_obf vs raw_obf @5: {mcnemar(R['raw_obf'], R['card_obf'], 5)}")
    print(f"McNemar card_obf vs raw_obf @1: {mcnemar(R['raw_obf'], R['card_obf'], 1)}")


if __name__ == "__main__":
    main()
