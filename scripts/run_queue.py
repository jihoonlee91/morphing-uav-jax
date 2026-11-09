"""Run a queue of experiment commands on a few CPU slots, one log file per job.

    uv run python scripts/run_queue.py jobs.txt --slots 4 --logs results/tmlr/logs

``jobs.txt`` has one command per line (arguments after ``python``, e.g.
``scripts/train_policy.py --method ppo --seed 3``); blank lines and ``#`` comments
are skipped. The logical CPUs are split evenly between the slots and every job is
pinned to its slot's CPUs (Windows process affinity, inherited by the job), so
several JAX processes share the machine without oversubscribing it. A job whose
log already ends with ``JOB OK`` is skipped, so the queue can be restarted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = """
import ctypes, runpy, sys
mask = int(sys.argv[1])
if sys.platform == "win32":
    k = ctypes.windll.kernel32
    k.SetProcessAffinityMask(k.GetCurrentProcess(), ctypes.c_size_t(mask))
sys.argv = sys.argv[2:]
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0] if "/" in sys.argv[0] else ".")
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--logs", type=Path, default=ROOT / "results" / "tmlr" / "logs")
    args = ap.parse_args()
    lines = [ln.strip() for ln in args.jobs.read_text(encoding="utf-8").splitlines()]
    jobs = [ln for ln in lines if ln and not ln.startswith("#")]
    args.logs.mkdir(parents=True, exist_ok=True)
    ncpu = os.cpu_count() or 1
    per = max(1, ncpu // args.slots)
    masks = [sum(1 << c for c in range(i * per, (i + 1) * per)) for i in range(args.slots)]
    lock = threading.Lock()
    queue = list(enumerate(jobs))
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(per))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def name_of(cmd: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in cmd.split(" ", 1)[-1])[:120]

    def worker(slot: int) -> None:
        while True:
            with lock:
                if not queue:
                    return
                i, cmd = queue.pop(0)
            log = args.logs / f"{i:03d}_{name_of(cmd)}.log"
            if log.exists() and log.read_text(encoding="utf-8", errors="replace").rstrip().endswith(
                "JOB OK"
            ):
                continue
            t0, argv = time.time(), cmd.split()
            with log.open("w", encoding="utf-8") as fh:
                fh.write(f"# slot {slot} mask {masks[slot]:#x}: {cmd}\n")
                fh.flush()
                rc = subprocess.call(
                    [sys.executable, "-c", PIN, str(masks[slot]), *argv],
                    cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env, creationflags=flags,
                )  # fmt: skip
                fh.write(
                    f"\n# {time.time() - t0:.0f} s\n{'JOB OK' if rc == 0 else f'JOB FAILED {rc}'}\n"
                )
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] slot {slot} rc={rc} {time.time() - t0:6.0f}s {cmd}", flush=True)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(args.slots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("queue done")


if __name__ == "__main__":
    main()
