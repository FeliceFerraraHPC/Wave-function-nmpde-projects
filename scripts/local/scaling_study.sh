#!/bin/bash
# ==============================================================================
# Local Scaling Study: Strong & Problem-Size Scaling
#
# Measures:
#   1. Strong Scaling: Fixed mesh resolution, sweeping core/process counts
#      (e.g., 1, 2, 4 cores), computing Speedup S(p) and Parallel Efficiency E(p).
#   2. Problem Size Scaling: Fixed cores, sweeping mesh refinements.
#
# Compares:
#   - Matrix-Free CG
#   - Matrix-Free DG
#   - Assembled Sparse Matrix (Theta)
#
# Results are saved in CSV format and plotted automatically.
#
# Usage:
#   ./scripts/local/scaling_study.sh [options]
#
# Options:
#   --type <strong|size>    Scaling type (default: strong)
#   --dim <2|3>             Spatial dimension (default: 2)
#   --refine <N>            Refinement level for strong scaling (default: 5)
#   --refines <"list">      Refinement levels for size scaling (default: "4 5 6")
#   --cores <"list">        Core/process counts to sweep (default: "1 2 4")
#   --time <T>              Simulation end time (default: 0.5)
#   --output <path>         Output CSV file path
# ==============================================================================

set -e

# Default settings
SCALING_TYPE="strong"
DIM="2"
STRONG_REFINE="5"
SIZE_REFINES="4 5 6"
FINAL_TIME="0.5"
OUTPUT_CSV="results/local/local_scaling.csv"

# Auto-detect available cores for strong scaling list
DETECTED_CORES=1
if command -v nproc &>/dev/null; then
    DETECTED_CORES=$(nproc)
elif command -v sysctl &>/dev/null; then
    DETECTED_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
fi

CORES_LIST="1 2"
if [ "$DETECTED_CORES" -ge 4 ]; then
    CORES_LIST="1 2 4"
fi
if [ "$DETECTED_CORES" -ge 8 ]; then
    CORES_LIST="1 2 4 8"
fi

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)    SCALING_TYPE="$2";  shift 2 ;;
        --dim)     DIM="$2";           shift 2 ;;
        --refine)  STRONG_REFINE="$2"; shift 2 ;;
        --refines) SIZE_REFINES="$2";  shift 2 ;;
        --cores)   CORES_LIST="$2";    shift 2 ;;
        --time)    FINAL_TIME="$2";    shift 2 ;;
        --output)  OUTPUT_CSV="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo " Starting Local Scaling Study (${SCALING_TYPE})"
echo " Dimension         : ${DIM}D"
echo " Simulation Time   : ${FINAL_TIME} s"
if [ "${SCALING_TYPE}" == "strong" ]; then
    echo " Fixed Refinement  : ${STRONG_REFINE}"
    echo " Sweeping Cores    : ${CORES_LIST}"
else
    echo " Sweeping Refines  : ${SIZE_REFINES}"
fi
echo " Results CSV       : ${OUTPUT_CSV}"
echo "======================================================================"

EXEC="./build/WaveBenchmark"

# Check if binary exists and is runnable
if [ ! -f "${EXEC}" ] || ! "${EXEC}" --dim 2 --refine 1 --time 0.001 --solver cg &>/dev/null; then
    echo ">>> Building WaveBenchmark..."
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"${DETECTED_CORES}"
    cd ..
fi

mkdir -p "$(dirname "${OUTPUT_CSV}")"
rm -f "${OUTPUT_CSV}"

TEMP_LOG="$(mktemp -t wave_scaling_XXXXXX.log 2>/dev/null || mktemp /tmp/wave_scaling_XXXXXX.log)"

SOLVERS=("cg" "theta" "dg")

if [ "${SCALING_TYPE}" == "strong" ]; then
    # Track baseline single-core execution times for speedup calculation
    declare -A BASELINE_TIMES

    for p in ${CORES_LIST}; do
        echo ""
        echo "--------------------------------------------------"
        echo " Testing on ${p} Processes / Cores"
        echo "--------------------------------------------------"

        RUN_CMD=""
        if command -v mpirun &>/dev/null && [ "$p" -gt 1 ]; then
            RUN_CMD="mpirun -np ${p}"
        elif command -v mpiexec &>/dev/null && [ "$p" -gt 1 ]; then
            RUN_CMD="mpiexec -n ${p}"
        fi

        for s in "${SOLVERS[@]}"; do
            echo "--> Running solver: ${s} on ${p} cores..."
            ${RUN_CMD} ${EXEC} --mode bench \
                               --dim "${DIM}" \
                               --refine "${STRONG_REFINE}" \
                               --time "${FINAL_TIME}" \
                               --solver "${s}" | tee "${TEMP_LOG}"

            BASE_TIME=""
            if [ -n "${BASELINE_TIMES[${s}]}" ]; then
                BASE_TIME="${BASELINE_TIMES[${s}]}"
            fi

            # Parse and save
            python3 scripts/parse_benchmark_output.py \
                --log "${TEMP_LOG}" \
                --csv "${OUTPUT_CSV}" \
                --mode scaling \
                --dim "${DIM}" \
                --refine "${STRONG_REFINE}" \
                --ranks "${p}" \
                ${BASE_TIME:+--baseline-time "${BASE_TIME}"}

            # Record baseline for p == 1
            if [ "$p" -eq 1 ]; then
                # Extract the compute time from log
                T1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-2)}' | tail -n 1 || true)
                if [ -n "$T1" ]; then
                    BASELINE_TIMES["${s}"]="$T1"
                fi
            fi
        done
    done
else
    # Problem size scaling
    for ref in ${SIZE_REFINES}; do
        echo ""
        echo "--------------------------------------------------"
        echo " Problem Size Scaling: Refinement Level ${ref}"
        echo "--------------------------------------------------"

        RUN_CMD=""
        if command -v mpirun &>/dev/null && [ "$DETECTED_CORES" -gt 1 ]; then
            RUN_CMD="mpirun -np ${DETECTED_CORES}"
        fi

        ${RUN_CMD} ${EXEC} --mode bench \
                           --dim "${DIM}" \
                           --refine "${ref}" \
                           --time "${FINAL_TIME}" \
                           --solver all | tee "${TEMP_LOG}"

        python3 scripts/parse_benchmark_output.py \
            --log "${TEMP_LOG}" \
            --csv "${OUTPUT_CSV}" \
            --mode scaling \
            --dim "${DIM}" \
            --refine "${ref}" \
            --ranks "${DETECTED_CORES}"
    done
fi

rm -f "${TEMP_LOG}"

echo ""
echo "======================================================================"
echo " Scaling Study Finished!"
echo " Output stored in: ${OUTPUT_CSV}"
echo "======================================================================"

# Generate plots
if command -v python3 &>/dev/null; then
    PLOT_DIR="$(dirname "${OUTPUT_CSV}")/plots"
    echo ">>> Generating scaling plots in: ${PLOT_DIR}..."
    python3 scripts/plot_results.py --csv "${OUTPUT_CSV}" --outdir "${PLOT_DIR}" || true
fi
