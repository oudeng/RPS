#!/usr/bin/env python3
"""utility/benchmark_runtime_end2end.py

End-to-end runtime benchmark for RPS agents using the existing tournament CLI (RPS_main.py).

This script generates temporary 2-agent seat files (<agent> vs <opponent>) and measures
wall-clock time for running RPS_main.py over a chosen number of rounds and *explicit* seeds.

Why explicit seeds?
-------------------
RPS_main.py expects a comma-separated list of seeds (e.g. "1,2,3,5,8,13,21,34,55,89").
Earlier versions of this benchmark script used an integer (e.g. --seeds 3) and passed it
through, which unintentionally ran *one* seed (seed=3) rather than 3 independent runs.

We therefore align this script with the rest of the repository: --seeds takes an explicit
comma-separated list.

Metric
------
We report an approximate cost per decision (ms/decision):

  ms_per_decision = 1000 * elapsed_seconds / (2 * rounds * n_seeds)

because each round produces one action decision for each of the 2 agents.

Notes / caveats
---------------
- This measures *end-to-end* time, including Python overhead and any I/O performed by RPS_main.py.
- It is best used for *relative* comparisons under fixed machine and configuration.
- For publication-quality numbers, run on the same reference hardware used for experiments.

Example
-------
  python utility/benchmark_runtime_end2end.py \
      --agents RNN_v2,LSTM_v2,A3C_v2,Tr_v2,B_v1,M_v1 \
      --rounds 500 \
      --seeds 1,2,3,5,8,13,21,34,55,89 \
      --output-base outputs/runtime_benchmark \
      --extra-args "--batch-size 64 --batch-freq 64"
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from typing import List

import pandas as pd


DEFAULT_SEEDS = "1,2,3,5,8,13,21,34,55,89"


def parse_agents(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_seeds(s: str) -> List[int]:
    # Accept comma-separated integers, with optional whitespace.
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds must be a non-empty comma-separated list, e.g. '1,2,3'.")
    seeds: List[int] = []
    for p in parts:
        try:
            seeds.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid seed '{p}'. Seeds must be integers.") from e
    return seeds


def write_seat_file(path: str, agent_name: str, opponent: str = "R") -> None:
    """Write a minimal 2-seat CSV consistent with this repo's seat schema."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("seat,agent,idxname\n")
        f.write(f"1,{agent_name},01_{agent_name}\n")
        f.write(f"2,{opponent},02_{opponent}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agents",
        default="RNN_v2,LSTM_v2,A3C_v2,Tr_v2,B_v1,M_v1",
        help=(
            "Comma-separated agent names to benchmark "
            "(default: common baselines + deep models)"
        ),
    )
    ap.add_argument("--rounds", type=int, default=500, help="Rounds per seed (default: 500)")
    ap.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help=(
            "Comma-separated seed list (default: %(default)s). "
            "Example: --seeds 1,2,3,5,8,13,21,34,55,89"
        ),
    )
    ap.add_argument(
        "--output-base",
        default="outputs/runtime_benchmark",
        help="Base output directory (default: outputs/runtime_benchmark)",
    )
    ap.add_argument("--python", default="python", help="Python executable to invoke (default: python)")
    ap.add_argument("--rps-main", default="RPS_main.py", help="Path to RPS_main.py (default: RPS_main.py)")
    ap.add_argument("--extra-args", default="", help="Extra CLI args passed verbatim to RPS_main.py")
    ap.add_argument("--opponent", default="R", help="Opponent agent (default: R)")
    args = ap.parse_args()

    agents = parse_agents(args.agents)
    seeds_list = parse_seeds(args.seeds)
    seeds_arg = ",".join(str(x) for x in seeds_list)
    n_seeds = len(seeds_list)

    os.makedirs(args.output_base, exist_ok=True)

    rows = []
    for agent in agents:
        seat_path = os.path.join(args.output_base, f"seat_{agent}_vs_{args.opponent}.csv")
        out_dir = os.path.join(args.output_base, f"{agent}_vs_{args.opponent}_r{args.rounds}_s{n_seeds}")
        os.makedirs(out_dir, exist_ok=True)
        write_seat_file(seat_path, agent, args.opponent)

        cmd = [
            args.python,
            args.rps_main,
            "--seats",
            seat_path,
            "--rounds",
            str(args.rounds),
            "--seeds",
            seeds_arg,
            "--output-dir",
            out_dir,
        ]

        # Append extra args (shell-like splitting that respects quotes)
        if args.extra_args.strip():
            cmd.extend(shlex.split(args.extra_args.strip()))

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        decisions = 2 * args.rounds * n_seeds
        ms_per_decision = 1000.0 * elapsed / max(1, decisions)

        rows.append(
            {
                "agent": agent,
                "opponent": args.opponent,
                "rounds": args.rounds,
                "seeds": seeds_arg,
                "n_seeds": n_seeds,
                "elapsed_s": elapsed,
                "decisions": decisions,
                "ms_per_decision": ms_per_decision,
                "returncode": proc.returncode,
            }
        )

        # Save logs for debugging if something fails
        with open(os.path.join(out_dir, "stdout.txt"), "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        with open(os.path.join(out_dir, "stderr.txt"), "w", encoding="utf-8") as f:
            f.write(proc.stderr)

        if proc.returncode != 0:
            print(f"[WARN] {agent}: RPS_main.py exited with code {proc.returncode}. See logs in {out_dir}.")

        print(f"[OK] {agent}: {elapsed:.2f}s total, {ms_per_decision:.3f} ms/decision (n_seeds={n_seeds})")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_base, "runtime_benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Done] Wrote: {csv_path}")


if __name__ == "__main__":
    main()
