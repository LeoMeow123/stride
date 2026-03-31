# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""
Master orchestrator: runs the full T-maze pipeline from raw data to figures.

Stages:
  1-3:   tmaze/run_tmaze.py          — decisions, events, metrics, learning cycles
  4-6:   gait/run_gait_extraction.py — stride detection + filtering
  7-9:   gait/run_gait.py            — trial tagging, LMM, heatmaps
  10:    regression/run_regression.py — WT vs Tau classification

Each stage checks for output file existence before running (resumable).

Usage:
    uv run python automated_pipeline/run_all.py
    uv run python automated_pipeline/run_all.py --batch gopro_sept2025
    uv run python automated_pipeline/run_all.py --stages 7-9    # only LMM
    uv run python automated_pipeline/run_all.py --stages 1-3,7-9
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
STAGES = {
    "1-3": {
        "name": "T-Maze Behavioral Analysis",
        "script": PIPELINE_DIR / "tmaze" / "run_tmaze.py",
    },
    "4-6": {
        "name": "Gait Extraction & Filtering",
        "script": PIPELINE_DIR / "gait" / "run_gait_extraction.py",
    },
    "7-9": {
        "name": "Gait LMM Analysis",
        "script": PIPELINE_DIR / "gait" / "run_gait.py",
    },
    "10": {
        "name": "WT vs Tau Classification",
        "script": PIPELINE_DIR / "regression" / "run_regression.py",
    },
}


def run_stage(stage_key: str, stage_info: dict, config: str, output: str,
              batch: str | None = None) -> bool:
    """Run a single pipeline stage via subprocess."""
    script = stage_info["script"]
    name = stage_info["name"]

    print(f"\n{'#' * 60}")
    print(f"# Stage {stage_key}: {name}")
    print(f"# Script: {script.name}")
    print(f"{'#' * 60}")

    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    cmd = [sys.executable, str(script), "--config", config, "--output", output]
    if batch and stage_key != "10":  # regression doesn't take --batch
        cmd.extend(["--batch", batch])

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PIPELINE_DIR.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="T-Maze Pipeline Orchestrator")
    parser.add_argument("--config", default="automated_pipeline/config.yaml")
    parser.add_argument("--output", default="automated_pipeline/output")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated stage ranges to run (e.g., '1-3,7-9'). Default: all")
    args = parser.parse_args()

    print("=" * 60)
    print("T-MAZE AUTOMATED PIPELINE — FULL RUN")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    if args.batch:
        print(f"Batch:  {args.batch}")
    print("=" * 60)

    # Determine which stages to run
    if args.stages:
        stage_keys = [s.strip() for s in args.stages.split(",")]
    else:
        stage_keys = list(STAGES.keys())

    # Validate
    for sk in stage_keys:
        if sk not in STAGES:
            print(f"ERROR: Unknown stage '{sk}'. Available: {', '.join(STAGES.keys())}")
            sys.exit(1)

    t_total = time.time()
    results = {}

    for sk in stage_keys:
        ok = run_stage(sk, STAGES[sk], args.config, args.output, args.batch)
        results[sk] = ok
        if not ok:
            print(f"\nStage {sk} failed. Stopping pipeline.")
            break

    # Summary
    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for sk, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  Stage {sk:5s} ({STAGES[sk]['name']:35s}): {status}")
    print(f"\nTotal time: {elapsed_total:.1f}s")

    if all(results.values()):
        print("\nAll stages completed successfully.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
