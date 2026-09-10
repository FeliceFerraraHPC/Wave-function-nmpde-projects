#!/usr/bin/env python3
"""
==============================================================================
Wave Equation Benchmark Plotting & Analysis Utility
==============================================================================
Generates publication-quality figures for:
  1. Strong Scaling (Fixed Mesh, Sweeping Processes/Cores p = 1, 2, 4, 8...):
     - Speedup S(p) = T(1) / T(p) vs. Processes (with ideal linear reference y=x)
     - Parallel Efficiency E(p) = S(p)/p * 100% vs. Processes
     - Wall-Clock Compute Time T(p) vs. Processes
  2. Problem Size Scaling (Fixed Cores, Sweeping Mesh Refinements h -> h/2):
     - Wall-Clock Compute Time vs. Problem Size (DoFs)
     - Average Time per Timestep vs. Problem Size (DoFs)
     - Effective Throughput (DoFs * steps / second) vs. DoFs
  3. Direct Comparison (Matrix-Free CG/DG vs. Assembled Sparse Matrix Theta):
     - Wall-Clock Compute Time vs. Problem Size (DoFs)
     - Speedup Factor of Matrix-Free over Sparse Baseline

Usage:
  # Auto-detection mode
  python3 scripts/plot_results.py --csv results/local/local_scaling.csv --outdir results/local/plots

  # Explicit mode selection:
  python3 scripts/plot_results.py --csv results/local/local_scaling.csv --type strong
  python3 scripts/plot_results.py --csv results/local/local_scaling.csv --type size
  python3 scripts/plot_results.py --csv results/local/matfree_vs_sparse_comparison.csv --type comparison
==============================================================================
"""

import os
import sys
import argparse
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend for HPC / batch nodes
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def format_ranks_axis(ax, sorted_ranks):
    """Formats the process/core axis cleanly with log2 scale and explicit ticks."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(ticker.FixedLocator(sorted_ranks))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.minorticks_off()
    if len(sorted_ranks) == 1:
        r = sorted_ranks[0]
        ax.set_xlim(max(0.5, r * 0.7), r * 1.4)
    else:
        ax.set_xlim(min(sorted_ranks) * 0.75, max(sorted_ranks) * 1.3)


STYLE_MAP = {
    "theta": {"label": "Assembled Sparse (Theta)", "color": "#d95f02", "marker": "s", "ls": "--"},
    "sparse": {"label": "Assembled Sparse (Theta)", "color": "#d95f02", "marker": "s", "ls": "--"},
    "cg": {"label": "Matrix-Free CG", "color": "#1b9e77", "marker": "o", "ls": "-"},
    "matfree_cg": {"label": "Matrix-Free CG", "color": "#1b9e77", "marker": "o", "ls": "-"},
    "dg": {"label": "Matrix-Free DG", "color": "#7570b3", "marker": "^", "ls": "-."},
    "matfree_dg": {"label": "Matrix-Free DG", "color": "#7570b3", "marker": "^", "ls": "-."},
}


def get_solver_style(solver_key):
    """Returns plotting style dictionary for the given solver."""
    return STYLE_MAP.get(solver_key.lower(), {
        "label": solver_key, "color": "black", "marker": "x", "ls": "-"
    })


def load_csv_data(csv_path):
    """Loads CSV file into a list of dictionaries."""
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.", file=sys.stderr)
        return []

    rows = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ==============================================================================
# 1. Strong Scaling: Fixed Mesh, Sweeping Processes/Cores p
# ==============================================================================
def plot_strong_scaling(rows, outdir, include_dg=False):
    """
    Plots Strong Scaling:
      1. Speedup S(p) = T(1) / T(p) vs. Processes/Cores (with ideal linear reference y=x)
      2. Parallel Efficiency E(p) = S(p)/p * 100% vs. Processes/Cores
      3. Wall-Clock Compute Time T(p) vs. Processes/Cores
    """
    if not HAS_MATPLOTLIB:
        print("Error: 'matplotlib' is not installed in this Python environment.", file=sys.stderr)
        print("To enable plotting, run: pip install matplotlib", file=sys.stderr)
        return

    os.makedirs(outdir, exist_ok=True)

    by_solver = defaultdict(list)
    dim_val = 2
    for r in rows:
        try:
            solver = r.get("solver", "unknown").strip()
            if not include_dg and solver.lower() in ("dg", "matfree_dg"):
                continue
            dim_val = int(r.get("dimension", 2))
            ranks = int(r.get("ranks_or_threads", r.get("ranks", 1)))
            compute_time = float(r.get("compute_time_s", 0.0))
            time_per_step = float(r.get("avg_time_per_step_s", 0.0))
            speedup = float(r.get("speedup")) if r.get("speedup") else None
            eff = float(r.get("parallel_efficiency")) if r.get("parallel_efficiency") else None

            by_solver[solver].append({
                "ranks": ranks,
                "compute_time": compute_time,
                "time_per_step": time_per_step,
                "speedup": speedup,
                "efficiency": eff
            })
        except (ValueError, KeyError):
            continue

    all_ranks = set()
    cleaned_by_solver = {}
    for solver, pts in by_solver.items():
        # Deduplicate multiple entries for same rank (take latest)
        by_rank = {}
        for p in pts:
            by_rank[p["ranks"]] = p
        unique_pts = [by_rank[r] for r in sorted(by_rank.keys())]

        t1 = unique_pts[0]["compute_time"] if unique_pts and unique_pts[0]["ranks"] == 1 else None

        for p in unique_pts:
            all_ranks.add(p["ranks"])
            if p["ranks"] == 1:
                if p["speedup"] is None:
                    p["speedup"] = 1.0
                if p["efficiency"] is None:
                    p["efficiency"] = 100.0
            else:
                if p["speedup"] is None and t1 and t1 > 0 and p["compute_time"] > 0:
                    p["speedup"] = t1 / p["compute_time"]
                if p["efficiency"] is None and p["speedup"] is not None and p["ranks"] > 0:
                    p["efficiency"] = (p["speedup"] / p["ranks"]) * 100.0

        cleaned_by_solver[solver] = unique_pts

    sorted_ranks = sorted(list(all_ranks))
    if not sorted_ranks:
        print("Warning: No valid ranks found for strong scaling.", file=sys.stderr)
        return

    # --- Figure 1: Speedup S(p) vs Processes ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(sorted_ranks, sorted_ranks, "k--", alpha=0.7, label="Ideal Linear Speedup")

    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p["speedup"] is not None]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        speedups = [p["speedup"] for p in valid_pts]
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, speedups,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup S(p) = T(1) / T(p)", fontsize=12, fontweight="bold")
    ax.set_title(f"Strong Scaling Speedup ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    speedup_path = os.path.join(outdir, f"strong_scaling_speedup_{dim_val}d.png")
    fig.savefig(speedup_path)
    plt.close(fig)
    print(f" Saved: {speedup_path}")

    # --- Figure 2: Parallel Efficiency E(p) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.axhline(100.0, color="k", linestyle="--", alpha=0.7, label="Ideal Efficiency (100%)")

    max_eff = 100.0
    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p["efficiency"] is not None]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        effs = [p["efficiency"] for p in valid_pts]
        max_eff = max(max_eff, max(effs))
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, effs,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parallel Efficiency E(p) (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Strong Scaling Parallel Efficiency ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(115.0, max_eff * 1.1))
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    eff_path = os.path.join(outdir, f"strong_scaling_efficiency_{dim_val}d.png")
    fig.savefig(eff_path)
    plt.close(fig)
    print(f" Saved: {eff_path}")

    # --- Figure 3: Compute Time T(p) vs Processes ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p["compute_time"] > 0]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        times = [p["compute_time"] for p in valid_pts]
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, times,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

        # Ideal T(1)/p reference line
        if valid_pts[0]["ranks"] == 1:
            t1 = valid_pts[0]["compute_time"]
            ideal_times = [t1 / r for r in ranks]
            ax.plot(ranks, ideal_times, linestyle=":", color=s_info["color"], alpha=0.4)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_yscale("log")
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Compute Time T(p) (s)", fontsize=12, fontweight="bold")
    ax.set_title(f"Strong Scaling Wall-Clock Time ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    time_path = os.path.join(outdir, f"strong_scaling_time_{dim_val}d.png")
    fig.savefig(time_path)
    plt.close(fig)
    print(f" Saved: {time_path}")


# ==============================================================================
# 2. Weak Scaling: Work per Process Roughly Constant (N/p ~ const), Sweeping p
# ==============================================================================
def plot_weak_scaling(rows, outdir, include_dg=False):
    """
    Plots Weak Scaling (Work per process roughly constant N/p ~ const, sweeping p):
      1. Speedup S(p) = p * E(p)/100 vs. Processes/Cores (ideal linear reference y=x)
      2. Parallel Efficiency E(p) = T(1)/T(p) * (N(p)/(p*N(1))) * 100% vs. Processes/Cores
      3. Wall-Clock Compute Time T(p) vs. Processes/Cores (ideal: horizontal flat line T(p) = T(1))
      4. Computational Throughput (Million DoF-steps / s) vs. Processes/Cores
    """
    if not HAS_MATPLOTLIB:
        print("Error: 'matplotlib' is not installed in this Python environment.", file=sys.stderr)
        print("To enable plotting, run: pip install matplotlib", file=sys.stderr)
        return

    os.makedirs(outdir, exist_ok=True)

    by_solver = defaultdict(list)
    dim_val = 2
    for r in rows:
        try:
            solver = r.get("solver", "unknown").strip()
            if not include_dg and solver.lower() in ("dg", "matfree_dg"):
                continue
            dim_val = int(r.get("dimension", 2))
            ranks = int(r.get("ranks_or_threads", r.get("ranks", 1)))
            dofs = int(r.get("dofs", 0))
            steps = int(r.get("steps", 1))
            compute_time = float(r.get("compute_time_s", 0.0))
            time_per_step = float(r.get("avg_time_per_step_s", 0.0))
            throughput = (dofs * steps) / compute_time if compute_time > 0 else 0.0
            speedup = float(r.get("speedup")) if r.get("speedup") else None
            eff = float(r.get("parallel_efficiency")) if r.get("parallel_efficiency") else None

            by_solver[solver].append({
                "ranks": ranks,
                "dofs": dofs,
                "steps": steps,
                "compute_time": compute_time,
                "time_per_step": time_per_step,
                "throughput": throughput,
                "speedup": speedup,
                "efficiency": eff
            })
        except (ValueError, KeyError):
            continue

    all_ranks = set()
    cleaned_by_solver = {}
    for solver, pts in by_solver.items():
        # Deduplicate multiple entries for same rank (take latest)
        by_rank = {}
        for p in pts:
            by_rank[p["ranks"]] = p
        unique_pts = [by_rank[r] for r in sorted(by_rank.keys())]

        t1 = unique_pts[0]["compute_time"] if unique_pts and unique_pts[0]["ranks"] == 1 else None
        dofs1 = unique_pts[0]["dofs"] if unique_pts and unique_pts[0]["ranks"] == 1 else None

        for p in unique_pts:
            all_ranks.add(p["ranks"])
            if p["ranks"] == 1:
                if p["speedup"] is None:
                    p["speedup"] = 1.0
                if p["efficiency"] is None:
                    p["efficiency"] = 100.0
            else:
                dof_ratio = (p["dofs"] / (p["ranks"] * dofs1)) if (dofs1 and p["ranks"] > 0) else 1.0
                if p["efficiency"] is None and t1 and t1 > 0 and p["compute_time"] > 0:
                    p["efficiency"] = round((t1 / p["compute_time"]) * dof_ratio * 100.0, 1)
                if p["speedup"] is None:
                    if p["efficiency"] is not None:
                        p["speedup"] = round((p["efficiency"] / 100.0) * p["ranks"], 2)
                    elif t1 and t1 > 0 and p["compute_time"] > 0:
                        p["speedup"] = round((t1 / p["compute_time"]) * p["ranks"] * dof_ratio, 2)

        cleaned_by_solver[solver] = unique_pts

    sorted_ranks = sorted(list(all_ranks))
    if not sorted_ranks:
        print("Warning: No valid ranks found for weak scaling.", file=sys.stderr)
        return

    # --- Figure 1: Speedup S(p) vs Processes ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(sorted_ranks, sorted_ranks, "k--", alpha=0.7, label="Ideal Linear Speedup")

    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p["speedup"] is not None]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        speedups = [p["speedup"] for p in valid_pts]
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, speedups,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup S(p)", fontsize=12, fontweight="bold")
    ax.set_title(f"Weak Scaling Speedup ({dim_val}D, Work/Core ~ const)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    speedup_path = os.path.join(outdir, f"weak_scaling_speedup_{dim_val}d.png")
    fig.savefig(speedup_path)
    plt.close(fig)
    print(f" Saved: {speedup_path}")

    # --- Figure 2: Parallel Efficiency E(p) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.axhline(100.0, color="k", linestyle="--", alpha=0.7, label="Ideal Efficiency (100%)")

    max_eff = 100.0
    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p["efficiency"] is not None]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        effs = [p["efficiency"] for p in valid_pts]
        max_eff = max(max_eff, max(effs))
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, effs,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parallel Efficiency E(p) (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Weak Scaling Parallel Efficiency ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(120.0, max_eff * 1.1))
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    eff_path = os.path.join(outdir, f"weak_scaling_efficiency_{dim_val}d.png")
    fig.savefig(eff_path)
    plt.close(fig)
    print(f" Saved: {eff_path}")

    # --- Figure 3: Compute Time T(p) vs Processes ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p["compute_time"] > 0]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        times = [p["compute_time"] for p in valid_pts]
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, times,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

        # Baseline horizontal reference line T(1) for this solver (ideal flat weak scaling)
        if valid_pts[0]["ranks"] == 1:
            t1 = valid_pts[0]["compute_time"]
            ax.axhline(t1, color=s_info["color"], linestyle=":", alpha=0.45)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_yscale("log")
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Compute Time T(p) (s)", fontsize=12, fontweight="bold")
    ax.set_title(f"Weak Scaling Wall-Clock Time ({dim_val}D, Work/Core ~ const)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    time_path = os.path.join(outdir, f"weak_scaling_time_{dim_val}d.png")
    fig.savefig(time_path)
    plt.close(fig)
    print(f" Saved: {time_path}")

    # --- Figure 4: Computational Throughput (DoFs/s) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for solver_key, points in cleaned_by_solver.items():
        valid_pts = [p for p in points if p.get("throughput", 0.0) > 0]
        if not valid_pts:
            continue
        ranks = [p["ranks"] for p in valid_pts]
        throughputs = [p["throughput"] / 1e6 for p in valid_pts]  # MDoFs/s
        s_info = get_solver_style(solver_key)

        ax.plot(ranks, throughputs,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

        # Ideal linear throughput reference based on rank 1
        if valid_pts[0]["ranks"] == 1:
            tp1 = valid_pts[0]["throughput"] / 1e6
            ideal_tps = [tp1 * r for r in ranks]
            ax.plot(ranks, ideal_tps, linestyle=":", color=s_info["color"], alpha=0.45)

    format_ranks_axis(ax, sorted_ranks)
    ax.set_xlabel("MPI Processes / CPU Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (MDoF-steps / s)", fontsize=12, fontweight="bold")
    ax.set_title(f"Weak Scaling: Computational Throughput ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    tp_path = os.path.join(outdir, f"weak_scaling_throughput_{dim_val}d.png")
    fig.savefig(tp_path)
    plt.close(fig)
    print(f" Saved: {tp_path}")


# ==============================================================================
# 3. Problem Size Scaling: Fixed Cores, Sweeping Mesh Refinements
# ==============================================================================
def plot_problem_size_scaling(rows, outdir, include_dg=False):
    """
    Plots Problem Size Scaling:
      1. Wall-Clock Compute Time vs. Problem Size (DoFs)
      2. Average Time per Timestep vs. Problem Size (DoFs)
      3. Effective Throughput (DoFs * steps / s) vs. Problem Size (DoFs)
    """
    if not HAS_MATPLOTLIB:
        print("Error: 'matplotlib' is not installed in this Python environment.", file=sys.stderr)
        print("To enable plotting, run: pip install matplotlib", file=sys.stderr)
        return

    os.makedirs(outdir, exist_ok=True)

    by_solver = defaultdict(list)
    dim_val = 2
    cores_used = 1

    for r in rows:
        try:
            solver = r.get("solver", "unknown").strip()
            if not include_dg and solver.lower() in ("dg", "matfree_dg"):
                continue
            dim_val = int(r.get("dimension", 2))
            cores_used = int(r.get("ranks_or_threads", r.get("ranks", 1)))
            dofs = int(r.get("dofs", 0))
            steps = int(r.get("steps", 1))
            compute_time = float(r.get("compute_time_s", 0.0))
            time_per_step = float(r.get("avg_time_per_step_s", 0.0))
            throughput = (dofs * steps) / compute_time if compute_time > 0 else 0.0

            by_solver[solver].append({
                "dofs": dofs,
                "compute_time": compute_time,
                "time_per_step": time_per_step,
                "throughput": throughput
            })
        except (ValueError, KeyError):
            continue

    # --- Figure 1: Compute Time vs DoFs ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for solver_key, points in by_solver.items():
        points.sort(key=lambda p: p["dofs"])
        dofs = [p["dofs"] for p in points]
        times = [p["compute_time"] for p in points]
        s_info = get_solver_style(solver_key)

        ax.plot(dofs, times,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Compute Time (s)", fontsize=12, fontweight="bold")
    ax.set_title(f"Problem Size Scaling: Wall-Clock Time ({dim_val}D, {cores_used} Cores)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    time_plot_path = os.path.join(outdir, f"problem_size_scaling_time_{dim_val}d.png")
    fig.savefig(time_plot_path)
    plt.close(fig)
    print(f" Saved: {time_plot_path}")

    # --- Figure 2: Average Time per Timestep vs DoFs ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for solver_key, points in by_solver.items():
        points.sort(key=lambda p: p["dofs"])
        dofs = [p["dofs"] for p in points]
        step_times = [p["time_per_step"] * 1000.0 for p in points]
        s_info = get_solver_style(solver_key)

        ax.plot(dofs, step_times,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Time per Timestep (ms)", fontsize=12, fontweight="bold")
    ax.set_title(f"Problem Size Scaling: Per-Step Cost ({dim_val}D, {cores_used} Cores)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    step_plot_path = os.path.join(outdir, f"problem_size_scaling_step_time_{dim_val}d.png")
    fig.savefig(step_plot_path)
    plt.close(fig)
    print(f" Saved: {step_plot_path}")

    # --- Figure 3: Effective Throughput (DoFs/s) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    for solver_key, points in by_solver.items():
        points.sort(key=lambda p: p["dofs"])
        dofs = [p["dofs"] for p in points]
        throughputs = [p["throughput"] / 1e6 for p in points]  # MDoFs/s
        s_info = get_solver_style(solver_key)

        ax.plot(dofs, throughputs,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (MDoF-steps / s)", fontsize=12, fontweight="bold")
    ax.set_title(f"Problem Size Scaling: Computational Throughput ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    tp_plot_path = os.path.join(outdir, f"problem_size_scaling_throughput_{dim_val}d.png")
    fig.savefig(tp_plot_path)
    plt.close(fig)
    print(f" Saved: {tp_plot_path}")


# ==============================================================================
# 3. Direct Comparison: Matrix-Free vs. Assembled Sparse Matrix
# ==============================================================================
def plot_matfree_vs_sparse_comparison(rows, outdir, include_dg=False):
    """
    Plots direct comparison of compute time and speedup between
    Matrix-Free methods (CG, DG) and Assembled Sparse Matrix (Theta).
    """
    if not HAS_MATPLOTLIB:
        print("Error: 'matplotlib' is not installed in this Python environment.", file=sys.stderr)
        print("To enable plotting, run: pip install matplotlib", file=sys.stderr)
        return

    os.makedirs(outdir, exist_ok=True)

    by_dim = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            dim = int(r.get("dimension", 2))
            solver = r.get("solver", "unknown").strip()
            if not include_dg and solver.lower() in ("dg", "matfree_dg"):
                continue
            dofs = int(r.get("dofs", 0))
            compute_time = float(r.get("compute_time_s", 0.0))
            time_per_step = float(r.get("avg_time_per_step_s", 0.0))
            refine = int(r.get("refinement", 0))
            speedup = float(r.get("speedup_vs_sparse", 1.0)) if r.get("speedup_vs_sparse") else 1.0

            by_dim[dim][solver].append({
                "refine": refine,
                "dofs": dofs,
                "compute_time": compute_time,
                "time_per_step": time_per_step,
                "speedup": speedup
            })
        except (ValueError, KeyError):
            continue

    for dim, solvers_data in by_dim.items():
        # --- Figure 1: Total Compute Time vs DoFs ---
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        for solver_key, points in solvers_data.items():
            points = sorted(points, key=lambda p: p["dofs"])
            dofs = [p["dofs"] for p in points]
            times = [p["compute_time"] for p in points]
            s_info = get_solver_style(solver_key)

            ax.plot(dofs, times,
                    label=s_info["label"],
                    color=s_info["color"],
                    marker=s_info["marker"],
                    linestyle=s_info["ls"],
                    linewidth=2.2,
                    markersize=7)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Total Compute Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(f"Wall-Clock Compute Time vs. Problem Size ({dim}D)", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=11)
        plt.tight_layout()

        time_plot_path = os.path.join(outdir, f"wallclock_time_comparison_{dim}d.png")
        fig.savefig(time_plot_path)
        plt.close(fig)
        print(f" Saved: {time_plot_path}")

        # --- Figure 2: Average Time per Timestep vs DoFs ---
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        for solver_key, points in solvers_data.items():
            points = sorted(points, key=lambda p: p["dofs"])
            dofs = [p["dofs"] for p in points]
            step_times = [p["time_per_step"] * 1000.0 for p in points]
            s_info = get_solver_style(solver_key)

            ax.plot(dofs, step_times,
                    label=s_info["label"],
                    color=s_info["color"],
                    marker=s_info["marker"],
                    linestyle=s_info["ls"],
                    linewidth=2.2,
                    markersize=7)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Avg Time per Timestep (ms)", fontsize=12, fontweight="bold")
        ax.set_title(f"Per-Step Computational Cost vs. Problem Size ({dim}D)", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(frameon=True, fontsize=11)
        plt.tight_layout()

        step_plot_path = os.path.join(outdir, f"time_per_step_comparison_{dim}d.png")
        fig.savefig(step_plot_path)
        plt.close(fig)
        print(f" Saved: {step_plot_path}")

        # --- Figure 3: Speedup over Assembled Sparse Matrix ---
        has_speedup = any(
            any(p.get("speedup", 1.0) > 0 and p.get("speedup", 1.0) != 1.0 for p in pts)
            for s, pts in solvers_data.items() if s.lower() not in ["theta", "sparse"]
        )

        if has_speedup:
            fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Sparse Baseline (1.0x)")

            for solver_key, points in solvers_data.items():
                if solver_key.lower() in ["theta", "sparse"]:
                    continue
                points = sorted(points, key=lambda p: p["dofs"])
                dofs = [p["dofs"] for p in points]
                speedups = [p["speedup"] for p in points]
                s_info = get_solver_style(solver_key)

                ax.plot(dofs, speedups,
                        label=f"{s_info['label']} Speedup",
                        color=s_info["color"],
                        marker=s_info["marker"],
                        linestyle=s_info["ls"],
                        linewidth=2.4,
                        markersize=8)

            ax.set_xscale("log")
            ax.set_xlabel("Degrees of Freedom (DoFs)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Speedup factor (x times faster)", fontsize=12, fontweight="bold")
            ax.set_title(f"Matrix-Free Speedup over Assembled Sparse Matrix ({dim}D)", fontsize=13, fontweight="bold", pad=12)
            ax.grid(True, which="both", linestyle="--", alpha=0.6)
            ax.legend(frameon=True, fontsize=11)
            plt.tight_layout()

            speedup_plot_path = os.path.join(outdir, f"speedup_vs_sparse_{dim}d.png")
            fig.savefig(speedup_plot_path)
            plt.close(fig)
            print(f" Saved: {speedup_plot_path}")


# ==============================================================================
# Main CLI & Auto-detection Dispatcher
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot Wave Benchmark Results")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--outdir", default=None, help="Directory to save generated plots (default: alongside CSV)")
    parser.add_argument("--type", choices=["auto", "strong", "weak", "size", "comparison"], default="auto",
                        help="Analysis type: 'strong' (fixed mesh, sweeping cores), 'weak' (scaling mesh with cores), 'size' (fixed cores, sweeping refinements), 'comparison' (matfree vs sparse), or 'auto' (default: auto)")
    parser.add_argument("--include-dg", action="store_true", default=False,
                        help="Include Matrix-Free DG in scaling plots (default: False, DG is excluded)")
    args = parser.parse_args()

    csv_path = args.csv
    outdir = args.outdir if args.outdir else os.path.join(os.path.dirname(csv_path), "plots")

    rows = load_csv_data(csv_path)
    if not rows:
        print("No data loaded. Exiting.")
        sys.exit(1)

    selected_type = args.type

    # Auto-detection logic if type == 'auto'
    if selected_type == "auto":
        base_name = os.path.basename(csv_path).lower()
        if "weak" in base_name:
            selected_type = "weak"
        elif "strong" in base_name:
            selected_type = "strong"
        elif "size" in base_name:
            selected_type = "size"
        elif "comparison" in base_name or "matfree" in base_name:
            selected_type = "comparison"
        else:
            header_keys = set(rows[0].keys())
            if "ranks_or_threads" in header_keys or "ranks" in header_keys:
                ranks_set = {r.get("ranks_or_threads", r.get("ranks", 1)) for r in rows}
                refines_set = {r.get("refinement", "") for r in rows}

                if len(ranks_set) > 1 and len(refines_set) > 1:
                    # Both ranks and refinements vary -> Weak Scaling
                    selected_type = "weak"
                elif len(ranks_set) > 1 and len(refines_set) <= 1:
                    # Fixed mesh, sweeping ranks -> Strong Scaling
                    selected_type = "strong"
                elif len(refines_set) > 1 and len(ranks_set) <= 1:
                    # Fixed ranks, sweeping refinements -> Problem Size Scaling
                    selected_type = "size"
                else:
                    selected_type = "strong"
            else:
                selected_type = "comparison"

    print(f"Selected analysis type: '{selected_type}'")
    if selected_type == "strong":
        print(f"Generating Strong Scaling curves (speedup, efficiency, time vs cores)...")
        plot_strong_scaling(rows, outdir, include_dg=args.include_dg)
    elif selected_type == "weak":
        print(f"Generating Weak Scaling curves (speedup, efficiency, time vs cores)...")
        plot_weak_scaling(rows, outdir, include_dg=args.include_dg)
    elif selected_type == "size":
        print(f"Generating Problem Size Scaling curves (time, per-step cost, throughput vs DoFs)...")
        plot_problem_size_scaling(rows, outdir, include_dg=args.include_dg)
    elif selected_type == "comparison":
        print(f"Generating Direct Comparison curves (wall-clock time, per-step cost, speedup)...")
        plot_matfree_vs_sparse_comparison(rows, outdir, include_dg=args.include_dg)

    print(f"\nCompleted! Target plot directory: {outdir}\n")


if __name__ == "__main__":
    main()
