#!/bin/bash
# ==============================================================================
# Local Benchmark: Matrix-Free vs. Assembled Sparse Matrix Comparison
#
# Direct comparison of wall-clock compute time between:
#   - Matrix-Free CG (FEEvaluation cell_loop, SIMD tensor-product)
#   - Matrix-Free DG (DG cell & face loops)
#   - Assembled Sparse Matrix (Theta-scheme Crank-Nicolson)
#
# Results are saved in CSV format and plotted automatically.
#
# Usage:
#   ./scripts/local/compare_matfree_vs_sparse.sh [options]
#
# Options:
#   --dim <2|3|all>         Spatial dimension (default: 2)
#   --refine-2d <"list">    2D mesh refinement levels (default: "4 5 6")
#   --refine-3d <"list">    3D mesh refinement levels (default: "2 3")
#   --time <T>              Simulation end time (default: 1.0)
#   --np <N>                Number of processes (default: auto or 1)
#   --output <path>         Output CSV file path
# ==============================================================================

set -e

# Default settings
DIM="2"
REFINES_2D="4 5 6"
REFINES_3D="2 3"
FINAL_TIME="1.0"
OUTPUT_CSV="results/local/matfree_vs_sparse_comparison.csv"
NP=""

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dim)       DIM="$2";        shift 2 ;;
        --refine-2d) REFINES_2D="$2"; shift 2 ;;
        --refine-3d) REFINES_3D="$2"; shift 2 ;;
        --time)      FINAL_TIME="$2"; shift 2 ;;
        --np)        NP="$2";         shift 2 ;;
        --output)    OUTPUT_CSV="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Detect CPU cores if NP not specified
if [ -z "$NP" ]; then
    if command -v nproc &>/dev/null; then
        NP=$(nproc)
    elif command -v sysctl &>/dev/null; then
        NP=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
    else
        NP=1
    fi
    # Use at most 4 processes for local tests by default to prevent thrashing
    if [ "$NP" -gt 4 ]; then
        NP=4
    fi
fi

echo "======================================================================"
echo " Starting Local Matrix-Free vs. Assembled Sparse Comparison"
echo " Cores / Processes : ${NP}"
echo " Simulation Time   : ${FINAL_TIME} s"
echo " Results CSV       : ${OUTPUT_CSV}"
echo "======================================================================"

# Determine launcher (mpirun or direct execution)
RUN_CMD=""
if command -v mpirun &>/dev/null && [ "$NP" -gt 1 ]; then
    RUN_CMD="mpirun -np ${NP}"
elif command -v mpiexec &>/dev/null && [ "$NP" -gt 1 ]; then
    RUN_CMD="mpiexec -n ${NP}"
fi

EXEC="./build/WaveBenchmark"

# Check if binary exists and is runnable on current architecture
if [ ! -f "${EXEC}" ] || ! "${EXEC}" --dim 2 --refine 1 --time 0.001 --solver cg &>/dev/null; then
    echo ">>> Building WaveBenchmark..."
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"${NP}"
    cd ..
fi

mkdir -p "$(dirname "${OUTPUT_CSV}")"
rm -f "${OUTPUT_CSV}"

TEMP_LOG="$(mktemp -t wave_bench_XXXXXX.log 2>/dev/null || mktemp /tmp/wave_bench_XXXXXX.log)"

run_for_dim() {
    local d="$1"
    local ref_list="$2"

    echo ""
    echo ">>> Testing ${d}D Problems across Refinements: ${ref_list}"

    for ref in ${ref_list}; do
        echo "--------------------------------------------------"
        echo " ${d}D Refinement Level: ${ref}"
        echo "--------------------------------------------------"
        rm -f "${TEMP_LOG}"

        # Run each solver (Assembled Sparse, Matrix-Free CG, Matrix-Free DG)
        for s in "theta" "cg" "dg"; do
            echo "--> Running solver: ${s} (dim=${d}, refine=${ref})..."
            ${RUN_CMD} ${EXEC} --mode bench \
                               --dim "${d}" \
                               --refine "${ref}" \
                               --time "${FINAL_TIME}" \
                               --solver "${s}" 2>&1 | tee -a "${TEMP_LOG}" || true
        done

        # Parse accumulated solver outputs into structured CSV
        python3 scripts/parse_benchmark_output.py \
            --log "${TEMP_LOG}" \
            --csv "${OUTPUT_CSV}" \
            --mode comparison \
            --dim "${d}" \
            --refine "${ref}" \
            --ranks "${NP}"
    done
}

# Run selected dimensions
if [ "${DIM}" == "2" ] || [ "${DIM}" == "all" ]; then
    run_for_dim 2 "${REFINES_2D}"
fi

if [ "${DIM}" == "3" ] || [ "${DIM}" == "all" ]; then
    run_for_dim 3 "${REFINES_3D}"
fi

rm -f "${TEMP_LOG}"

echo ""
echo "======================================================================"
echo " Comparison Completed Successfully!"
echo " Results written to: ${OUTPUT_CSV}"
echo "======================================================================"

# Generate plots
if command -v python3 &>/dev/null; then
    PLOT_DIR="$(dirname "${OUTPUT_CSV}")/plots"
    echo ">>> Generating comparison plots in: ${PLOT_DIR}..."
    python3 scripts/plot_results.py --csv "${OUTPUT_CSV}" --outdir "${PLOT_DIR}" || true
fi
