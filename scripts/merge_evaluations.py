"""Merge ``evaluate.py --only`` runs into one ``metrics.json`` / ``per_scenario.npz``.

    uv run python scripts/merge_evaluations.py OUT_DIR PART_DIR [PART_DIR ...]

Every part must have flown the same scenarios (same ``--n``, ``--n-sweep`` and
condition seeds; checked on the stored scenario fields), so the merged arrays stay
paired across controllers. Later parts win on name clashes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    out, parts = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    metrics = [json.loads((p / "metrics.json").read_text(encoding="utf-8")) for p in parts]
    raws = [dict(np.load(p / "per_scenario.npz")) for p in parts]
    merged, raw = metrics[0], dict(raws[0])
    for m, r in zip(metrics[1:], raws[1:], strict=True):
        for k, v in r.items():
            same = k not in raw or np.array_equal(np.asarray(raw[k]), np.asarray(v), equal_nan=True)
            if "/_scenario/" in k and not same:
                raise SystemExit(f"scenarios differ: {k}")
            raw[k] = v
        for cond, row in m["table"].items():
            merged["table"][cond].update({k: v for k, v in row.items() if k != "label"})
        for sweep, sw in m["sweeps"].items():
            for r0, r1 in zip(merged["sweeps"][sweep]["rows"], sw["rows"], strict=True):
                assert r0["value"] == r1["value"]
                r0.update({k: v for k, v in r1.items() if k != "value"})
        for key in ("training_distribution_return", "training_distribution_return_per_seed",
                    "seeds_per_controller"):  # fmt: skip
            merged["meta"].setdefault(key, {}).update(m["meta"].get(key) or {})
        for dist, rows in (m["meta"].get("heldout_return") or {}).items():
            merged["meta"].setdefault("heldout_return", {}).setdefault(dist, {}).update(rows)
    merged["meta"]["merged_from"] = [str(p) for p in parts]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
    np.savez_compressed(out / "per_scenario.npz", **raw)
    print(f"merged {len(parts)} parts: {len(merged['table']['nominal']) - 1} controllers")


if __name__ == "__main__":
    main()
