#!/usr/bin/env python3
"""
==============================================================================
Wave Equation Benchmark Plotting & Analysis Utility
==============================================================================
Generates publication-quality figures comparing:
  1. Matrix-Free (CG and DG) vs. Assembled Sparse Matrix (Theta) Wall-Clock Time
  2. Speedup of Matrix-Free over Assembled Sparse Matrix
  3. Strong Scaling (Speedup & Parallel Efficiency vs. MPI Ranks / Threads)
  4. Average Time per Timestep vs. Problem Size (DoFs)

Usage:
  python3 scripts/plot_results.py --csv results/meluxina/meluxina_matfree_vs_sparse.csv --outdir results/meluxina/plots
  python3 scripts/plot_results.py --csv results/meluxina/meluxina_strong_scaling.csv --outdir results/meluxina/plots
  python3 scripts/plot_results.py --csv results/local/matfree_vs_sparse_comparison.csv --outdir results/local/plots
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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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


def plot_matfree_vs_sparse_comparison(rows, outdir):
    """
    Plots direct comparison of compute time and speedup between
    Matrix-Free methods (CG, DG) and Assembled Sparse Matrix (Theta).
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib is not installed. Skipping plot generation.", file=sys.stderr)
        return

    os.makedirs(outdir, exist_ok=True)

    # Group data by dimension
    by_dim = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            dim = int(r.get("dimension", 2))
            solver = r.get("solver", "unknown").strip()
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
        except (ValueError, KeyError) as e:
            continue

    style_map = {
        "theta": {"label": "Assembled Sparse (Theta)", "color": "#d95f02", "marker": "s", "ls": "--"},
        "sparse": {"label": "Assembled Sparse (Theta)", "color": "#d95f02", "marker": "s", "ls": "--"},
        "cg": {"label": "Matrix-Free CG", "color": "#1b9e77", "marker": "o", "ls": "-"},
        "matfree_cg": {"label": "Matrix-Free CG", "color": "#1b9e77", "marker": "o", "ls": "-"},
        "dg": {"label": "Matrix-Free DG", "color": "#7570b3", "marker": "^", "ls": "-."},
        "matfree_dg": {"label": "Matrix-Free DG", "color": "#7570b3", "marker": "^", "ls": "-."},
    }

    for dim, solvers_data in by_dim.items():
        # --- Figure 1: Total Compute Time vs DoFs ---
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

        for solver_key, points in solvers_data.items():
            points = sorted(points, key=lambda p: p["dofs"])
            dofs = [p["dofs"] for p in points]
            times = [p["compute_time"] for p in points]

            s_info = style_map.get(solver_key.lower(), {
                "label": solver_key, "color": "black", "marker": "x", "ls": "-"
            })

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
            step_times = [p["time_per_step"] * 1000.0 for p in points]  # convert to ms

            s_info = style_map.get(solver_key.lower(), {
                "label": solver_key, "color": "black", "marker": "x", "ls": "-"
            })

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
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Sparse Matrix Baseline (1.0x)")

            for solver_key, points in solvers_data.items():
                if solver_key.lower() in ["theta", "sparse"]:
                    continue
                points = sorted(points, key=lambda p: p["dofs"])
                dofs = [p["dofs"] for p in points]
                speedups = [p["speedup"] for p in points]

                s_info = style_map.get(solver_key.lower(), {
                    "label": solver_key, "color": "blue", "marker": "o", "ls": "-"
                })

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


def plot_scaling(rows, outdir):
    """
    Plots Strong Scaling:
      1. Speedup S(p) vs MPI ranks / threads (with ideal linear reference)
      2. Parallel Efficiency E(p) vs MPI ranks / threads
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib is not installed. Skipping plot generation.", file=sys.stderr)
        return

    os.makedirs(outdir, exist_ok=True)

    # Group by solver
    by_solver = defaultdict(list)
    dim_val = 2
    for r in rows:
        try:
            solver = r.get("solver", "unknown").strip()
            dim_val = int(r.get("dimension", 2))
            ranks = int(r.get("ranks_or_threads", r.get("ranks", 1)))
            compute_time = float(r.get("compute_time_s", 0.0))
            time_per_step = float(r.get("avg_time_per_step_s", 0.0))
            speedup = float(r.get("speedup", 1.0)) if r.get("speedup") else None
            eff = float(r.get("parallel_efficiency", 100.0)) if r.get("parallel_efficiency") else None

            by_solver[solver].append({
                "ranks": ranks,
                "compute_time": compute_time,
                "time_per_step": time_per_step,
                "speedup": speedup,
                "efficiency": eff
            })
        except (ValueError, KeyError):
            continue

    style_map = {
        "theta": {"label": "Assembled Sparse (Theta)", "color": "#d95f02", "marker": "s", "ls": "--"},
        "sparse": {"label": "Assembled Sparse (Theta)", "color": "#d95f02", "marker": "s", "ls": "--"},
        "cg": {"label": "Matrix-Free CG", "color": "#1b9e77", "marker": "o", "ls": "-"},
        "matfree_cg": {"label": "Matrix-Free CG", "color": "#1b9e77", "marker": "o", "ls": "-"},
        "dg": {"label": "Matrix-Free DG", "color": "#7570b3", "marker": "^", "ls": "-."},
        "matfree_dg": {"label": "Matrix-Free DG", "color": "#7570b3", "marker": "^", "ls": "-."},
    }

    # Compute baseline T(1) if speedup wasn't pre-computed
    all_ranks = set()
    for solver, pts in by_solver.items():
        pts.sort(key=lambda p: p["ranks"])
        t1 = pts[0]["compute_time"] if pts and pts[0]["ranks"] == 1 else None
        for p in pts:
            all_ranks.add(p["ranks"])
            if p["speedup"] is None and t1 and t1 > 0:
                p["speedup"] = t1 / p["compute_time"]
            if p["efficiency"] is None and p["speedup"] is not None and p["ranks"] > 0:
                p["efficiency"] = (p["speedup"] / p["ranks"]) * 100.0

    sorted_ranks = sorted(list(all_ranks))
    if not sorted_ranks:
        return

    # --- Plot 1: Speedup S(p) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Ideal linear speedup line
    ax.plot(sorted_ranks, sorted_ranks, "k--", alpha=0.7, label="Ideal Linear Speedup")

    for solver_key, points in by_solver.items():
        ranks = [p["ranks"] for p in points if p["speedup"] is not None]
        speedups = [p["speedup"] for p in points if p["speedup"] is not None]

        s_info = style_map.get(solver_key.lower(), {
            "label": solver_key, "color": "black", "marker": "o", "ls": "-"
        })

        ax.plot(ranks, speedups,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    ax.set_xlabel("MPI Processes / Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup S(p) = T(1) / T(p)", fontsize=12, fontweight="bold")
    ax.set_title(f"Strong Scaling Speedup ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(sorted_ranks)
    ax.set_xticklabels([str(r) for r in sorted_ranks])
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    speedup_path = os.path.join(outdir, f"strong_scaling_speedup_{dim_val}d.png")
    fig.savefig(speedup_path)
    plt.close(fig)
    print(f" Saved: {speedup_path}")

    # --- Plot 2: Parallel Efficiency E(p) ---
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.axhline(100.0, color="k", linestyle="--", alpha=0.7, label="Ideal Efficiency (100%)")

    for solver_key, points in by_solver.items():
        ranks = [p["ranks"] for p in points if p["efficiency"] is not None]
        effs = [p["efficiency"] for p in points if p["efficiency"] is not None]

        s_info = style_map.get(solver_key.lower(), {
            "label": solver_key, "color": "black", "marker": "o", "ls": "-"
        })

        ax.plot(ranks, effs,
                label=s_info["label"],
                color=s_info["color"],
                marker=s_info["marker"],
                linestyle=s_info["ls"],
                linewidth=2.2,
                markersize=7)

    ax.set_xlabel("MPI Processes / Cores", fontsize=12, fontweight="bold")
    ax.set_ylabel("Parallel Efficiency E(p) (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Parallel Efficiency vs. Processes ({dim_val}D)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(sorted_ranks)
    ax.set_xticklabels([str(r) for r in sorted_ranks])
    ax.set_ylim(0, 115)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    eff_path = os.path.join(outdir, f"strong_scaling_efficiency_{dim_val}d.png")
    fig.savefig(eff_path)
    plt.close(fig)
    print(f" Saved: {eff_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Wave Benchmark Results")
    parser.add_argument("--csv", required=True, help="Path to input CSV file")
    parser.add_argument("--outdir", default=None, help="Directory to save generated plots (default: alongside CSV)")
    args = parser.parse_args()

    csv_path = args.csv
    outdir = args.outdir if args.outdir else os.path.join(os.path.dirname(csv_path), "plots")

    rows = load_csv_data(csv_path)
    if not rows:
        print("No data loaded. Exiting.")
        sys.exit(1)

    # Detect CSV type based on columns
    header_keys = set(rows[0].keys())
    if "ranks_or_threads" in header_keys or "scaling_type" in header_keys:
        print(f"Detected scaling study data. Plotting scaling curves...")
        plot_scaling(rows, outdir)
    else:
        print(f"Detected solver comparison data. Plotting comparison curves...")
        plot_matfree_vs_sparse_comparison(rows, outdir)

    print(f"\nAll plots saved to: {outdir}\n")


if __name__ == "__main__":
    main()
