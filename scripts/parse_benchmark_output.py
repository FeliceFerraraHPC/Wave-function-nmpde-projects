#!/usr/bin/env python3
"""
==============================================================================
Benchmark Log Parser Utility
==============================================================================
Parses stdout produced by WaveBenchmark --mode bench and appends
clean, structured entries to a target CSV file.
==============================================================================
"""

import sys
import os
import re
import argparse
import datetime
import csv

def parse_benchmark_log(log_path):
    """
    Parses WaveBenchmark stdout log and returns a list of dictionaries
    representing each solver's execution results.
    """
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    results = []

    # Regex matching the solver table rows
    # Example:
    # Theta-scheme (Trilinos MPI)         16641       100           0.5123            0.00512           1.234e-05
    # Matrix-Free CG (MPI+TBB+SIMD)       16641       250           0.0812            0.00032          -2.110e-06
    pattern = re.compile(
        r"^(Theta-scheme[^\n\d]+|Matrix-Free CG[^\n\d]+|Matrix-Free DG[^\n\d]+)\s+"
        r"(\d+)\s+"                       # DoFs
        r"(\d+)\s+"                       # Steps
        r"([\d\.\-eE+]+)\s+"              # Compute time (s)
        r"([\d\.\-eE+]+)\s+"              # Avg time/step (s)
        r"([\d\.\-eE+]+)",                # Energy drift
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        name_raw = match.group(1).strip()
        dofs = int(match.group(2))
        steps = int(match.group(3))
        compute_time = float(match.group(4))
        avg_step = float(match.group(5))
        e_drift = float(match.group(6))

        if "Theta-scheme" in name_raw:
            tag = "theta"
        elif "Matrix-Free CG" in name_raw:
            tag = "cg"
        elif "Matrix-Free DG" in name_raw:
            tag = "dg"
        else:
            tag = name_raw.lower().replace(" ", "_")

        results.append({
            "solver": tag,
            "solver_name": name_raw,
            "dofs": dofs,
            "steps": steps,
            "compute_time_s": compute_time,
            "avg_time_per_step_s": avg_step,
            "energy_drift": e_drift,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse WaveBenchmark output log to CSV")
    parser.add_argument("--log", required=True, help="Path to raw output log")
    parser.add_argument("--csv", required=True, help="Path to destination CSV")
    parser.add_argument("--mode", choices=["comparison", "scaling"], default="comparison")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--refine", type=int, default=6)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--baseline-time", type=float, default=None, help="Baseline time T(1) for speedup computation")
    parser.add_argument("--baseline-dofs", type=int, default=None, help="Baseline DoFs at rank 1 for weak scaling normalization")
    parser.add_argument("--scaling-type", choices=["strong", "weak"], default="strong", help="Scaling type (strong or weak)")
    args = parser.parse_args()

    records = parse_benchmark_log(args.log)
    if not records:
        print(f"Warning: No benchmark table rows found in '{args.log}'", file=sys.stderr)
        return

    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    # If in comparison mode, find theta baseline time if present in current records
    theta_time = None
    for r in records:
        if r["solver"] == "theta":
            theta_time = r["compute_time_s"]
            break

    if args.mode == "comparison":
        fieldnames = [
            "timestamp", "dimension", "refinement", "solver", "solver_name",
            "dofs", "steps", "compute_time_s", "avg_time_per_step_s",
            "energy_drift", "speedup_vs_sparse"
        ]
        file_exists = os.path.exists(args.csv)

        with open(args.csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for r in records:
                speedup = 1.0
                if theta_time and theta_time > 0 and r["compute_time_s"] > 0:
                    speedup = round(theta_time / r["compute_time_s"], 2)

                writer.writerow({
                    "timestamp": now_iso,
                    "dimension": args.dim,
                    "refinement": args.refine,
                    "solver": r["solver"],
                    "solver_name": r["solver_name"],
                    "dofs": r["dofs"],
                    "steps": r["steps"],
                    "compute_time_s": r["compute_time_s"],
                    "avg_time_per_step_s": r["avg_time_per_step_s"],
                    "energy_drift": r["energy_drift"],
                    "speedup_vs_sparse": speedup
                })

    elif args.mode == "scaling":
        fieldnames = [
            "timestamp", "dimension", "refinement", "solver", "solver_name",
            "dofs", "ranks_or_threads", "steps", "compute_time_s",
            "avg_time_per_step_s", "energy_drift", "speedup", "parallel_efficiency"
        ]
        file_exists = os.path.exists(args.csv)

        with open(args.csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for r in records:
                speedup = None
                eff = None
                if args.baseline_time and args.baseline_time > 0 and r["compute_time_s"] > 0:
                    if args.scaling_type == "weak":
                        dof_ratio = (r["dofs"] / (args.ranks * args.baseline_dofs)) if (args.baseline_dofs and args.ranks > 0) else 1.0
                        eff = round((args.baseline_time / r["compute_time_s"]) * dof_ratio * 100.0, 1)
                        speedup = round((args.baseline_time / r["compute_time_s"]) * (args.ranks * dof_ratio), 2)
                    else:
                        speedup = round(args.baseline_time / r["compute_time_s"], 2)
                        eff = round((speedup / args.ranks) * 100.0, 1) if args.ranks > 0 else 100.0

                writer.writerow({
                    "timestamp": now_iso,
                    "dimension": args.dim,
                    "refinement": args.refine,
                    "solver": r["solver"],
                    "solver_name": r["solver_name"],
                    "dofs": r["dofs"],
                    "ranks_or_threads": args.ranks,
                    "steps": r["steps"],
                    "compute_time_s": r["compute_time_s"],
                    "avg_time_per_step_s": r["avg_time_per_step_s"],
                    "energy_drift": r["energy_drift"],
                    "speedup": speedup if speedup is not None else "",
                    "parallel_efficiency": eff if eff is not None else ""
                })


if __name__ == "__main__":
    main()
