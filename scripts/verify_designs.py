"""Check the model-based designs between their 63 design points.

    uv run python scripts/verify_designs.py

Re-trims the aircraft on a 13 x 5 x 5 grid (325 points, 262 off the design
grid) and writes ``results/verification.json``; see :mod:`morphjax.verify`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from morphjax import pipeline
from morphjax.verify import check_designs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    t0 = time.time()
    setup = pipeline.build(pipeline.load_weights(ROOT / "results" / "lqr_weights.json"))
    checks = check_designs(setup)
    rows = {name: c._asdict() for name, c in checks.items()}
    out = ROOT / "results" / "verification.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{'design':6s} {'stable':>10s} {'worst Re':>9s} {'min delay':>10s} {'certified':>10s}")
    for name, c in checks.items():
        cert = "-" if c.certified is None else f"{c.certified}/{c.points}"
        print(f"{name:6s} {c.stable:>4d}/{c.points:<5d} {c.worst_real_eig:9.3f} "
              f"{c.min_delay_ms:8.0f}ms {cert:>10s}")  # fmt: skip
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
