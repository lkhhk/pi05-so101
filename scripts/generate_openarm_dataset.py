"""Collect a small, spatially split OpenArm scripted dataset.

Run from the OpenArm checkout with the MuJoCo venv. The outputs use privileged
scripted grasp/lift assistance, and are not real robot demonstrations.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path, help="New output directory")
    p.add_argument("--collector", type=Path, default=Path("collect_openarm_unscrew.py"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    positions = [-0.012, -0.004, 0.004, 0.012]
    # Fixed before collecting any results. Each x coordinate appears in both
    # splits; this is a small interpolation check, not a robust OOD evaluation.
    test_indices = {(0, 1), (1, 3), (2, 0), (3, 2)}
    records = []
    manifest = args.output / "manifest.json"
    for ix, x in enumerate(positions):
        for iy, y in enumerate(positions):
            index = ix * len(positions) + iy
            split = "eval" if (ix, iy) in test_indices else "train"
            output = args.output / split / f"episode_{index:02}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(args.collector), "--output", str(output),
                       f"--offset-x={x}", f"--offset-y={y}", "--seed", str(1000 + index)]
            print(f"[{index + 1}/16] {split} offset=({x:.3f},{y:.3f})", flush=True)
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=240,
                                         env={**os.environ, "MUJOCO_GL": "egl"})
                ok = process.returncode == 0 and output.is_file()
                log = (process.stdout + process.stderr)[-4000:]
            except subprocess.TimeoutExpired:
                ok, log = False, "collector timed out after 240 seconds"
            records.append({"index": index, "split": split, "offset_xy_m": [x, y],
                            "seed": 1000 + index, "path": str(output.relative_to(args.output)),
                            "success": ok, "log": log})
            manifest.write_text(json.dumps({"source": "OpenArm scripted assisted controller",
                                             "records": records}, indent=2), encoding="utf-8")
            print(f"  {'OK' if ok else 'FAILED'} {log.strip()[-300:]}", flush=True)
    print("completed", sum(r["success"] and r["split"] == "train" for r in records),
          "train,", sum(r["success"] and r["split"] == "eval" for r in records), "eval", flush=True)
    if any(not r["success"] for r in records):
        raise SystemExit("Some demonstrations failed; inspect manifest before conversion")


if __name__ == "__main__":
    main()
