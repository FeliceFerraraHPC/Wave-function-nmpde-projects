#!/bin/bash
# ==============================================================================
# Local Scaling Study: Strong, Weak & Problem-Size Scaling
#
# Measures:
#   1. Strong Scaling (--type strong):
#      Fixed mesh resolution, sweeping core/process counts (e.g. 1, 2, 4, 8 cores).
#      Computes Speedup S(p) = T(1)/T(p) and Parallel Efficiency E(p) = S(p)/p * 100%.
#   2. Weak Scaling (--type weak):
#      Problem size scaled proportionally to core count (N/p ~ const), sweeping cores.
#      Ideal compute time is flat: T(p) = T(1).
#      Computes Weak Parallel Efficiency E(p) = T(1)/T(p) * (N(p)/(p*N(1))) * 100%.
#   3. Problem Size Scaling (--type size):
#      Fixed cores, sweeping mesh refinements h -> h/2.
#
# Solvers compared:
#   - Matrix-Free CG
#   - Matrix-Free DG
#   - Assembled Sparse Matrix (Theta-scheme)
#
# Results are saved in CSV format (plotting can be done separately via plot_results.py).
#
# Usage:
#   ./scripts/local/scaling_study.sh [options]
#
# Options:
#   --type <strong|weak|size> Scaling type (default: strong)
#   --dim <2|3>               Spatial dimension (default: 2)
#   --refine <N>              Refinement level (strong scaling) or base refinement (weak scaling)
#   --refines <"list">        Refinement levels for size scaling (default: "4 5 6")
#   --weak-refines <"list">   Explicit refinement levels corresponding to --cores for weak scaling
#   --cores <"list">          Core/process counts to sweep (default: detected or "1 2 4")
#   --solvers <"list">        Solvers to sweep (default: "cg theta")
#   --time <T>                Simulation end time (default: 0.5)
#   --output <path>           Output CSV file path
# ==============================================================================

set -e

# Default settings
SCALING_TYPE="strong"
DIM="2"
STRONG_REFINE="5"
BASE_REFINE=""
WEAK_REFINES=""
SIZE_REFINES="4 5 6"
FINAL_TIME="0.5"
OUTPUT_CSV=""
SOLVERS_STR="cg theta"

# Auto-detect available cores for strong/weak scaling list
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
        --type)         SCALING_TYPE="$2";  shift 2 ;;
        --dim)          DIM="$2";           shift 2 ;;
        --refine)       STRONG_REFINE="$2"; BASE_REFINE="$2"; shift 2 ;;
        --base-refine)  BASE_REFINE="$2";   shift 2 ;;
        --refines)      SIZE_REFINES="$2";  shift 2 ;;
        --weak-refines) WEAK_REFINES="$2";  shift 2 ;;
        --cores)        CORES_LIST="$2";    shift 2 ;;
        --solvers)      SOLVERS_STR="$2";   shift 2 ;;
        --time)         FINAL_TIME="$2";    shift 2 ;;
        --output)       OUTPUT_CSV="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default base refinement for weak scaling if not set
if [ -z "${BASE_REFINE}" ]; then
    if [ "${DIM}" == "3" ]; then
        BASE_REFINE="3"
    else
        BASE_REFINE="4"
    fi
fi

# Set default output CSV if not explicitly passed
if [ -z "${OUTPUT_CSV}" ]; then
    OUTPUT_CSV="results/local/local_${SCALING_TYPE}_scaling.csv"
fi

# Parse solvers array
read -r -a SOLVERS <<< "${SOLVERS_STR}"

echo "======================================================================"
echo " Starting Local Scaling Study (${SCALING_TYPE})"
echo " Dimension         : ${DIM}D"
echo " Simulation Time   : ${FINAL_TIME} s"
echo " Solvers           : ${SOLVERS[*]}"
if [ "${SCALING_TYPE}" == "strong" ]; then
    echo " Fixed Refinement  : ${STRONG_REFINE}"
    echo " Sweeping Cores    : ${CORES_LIST}"
elif [ "${SCALING_TYPE}" == "weak" ]; then
    echo " Base Refinement   : ${BASE_REFINE}"
    echo " Sweeping Cores    : ${CORES_LIST}"
    if [ -n "${WEAK_REFINES}" ]; then
        echo " Custom Refinements: ${WEAK_REFINES}"
    fi
else
    echo " Sweeping Refines  : ${SIZE_REFINES}"
    echo " Fixed Cores       : ${DETECTED_CORES}"
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

if [ "${SCALING_TYPE}" == "strong" ]; then
    # Track baseline single-core execution times for speedup calculation
    declare -A BASELINE_TIMES

    for p in ${CORES_LIST}; do
        echo ""
        echo "--------------------------------------------------"
        echo " Strong Scaling: ${p} Processes / Cores (Refinement ${STRONG_REFINE})"
        echo "--------------------------------------------------"

        RUN_CMD=""
        if command -v mpirun &>/dev/null && [ "$p" -gt 1 ]; then
            RUN_CMD="mpirun -np ${p}"
        elif command -v mpiexec &>/dev/null && [ "$p" -gt 1 ]; then
            RUN_CMD="mpiexec -n ${p}"
        fi

        for s in "${SOLVERS[@]}"; do
            echo "--> Running solver: ${s} on ${p} cores..."
            rm -f "${TEMP_LOG}"
            ${RUN_CMD} ${EXEC} --mode bench \
                               --dim "${DIM}" \
                               --refine "${STRONG_REFINE}" \
                               --time "${FINAL_TIME}" \
                               --solver "${s}" 2>&1 | tee "${TEMP_LOG}" || true

            BASE_TIME=""
            if [ -n "${BASELINE_TIMES[${s}]}" ]; then
                BASE_TIME="${BASELINE_TIMES[${s}]}"
            fi

            # Parse and save
            python3 scripts/parse_benchmark_output.py \
                --log "${TEMP_LOG}" \
                --csv "${OUTPUT_CSV}" \
                --mode scaling \
                --scaling-type strong \
                --dim "${DIM}" \
                --refine "${STRONG_REFINE}" \
                --ranks "${p}" \
                ${BASE_TIME:+--baseline-time "${BASE_TIME}"}

            # Record baseline for p == 1
            if [ "$p" -eq 1 ]; then
                T1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-2)}' | tail -n 1 || true)
                if [ -n "$T1" ]; then
                    BASELINE_TIMES["${s}"]="$T1"
                fi
            fi
        done
    done

elif [ "${SCALING_TYPE}" == "weak" ]; then
    # Weak Scaling: N/p roughly constant, sweeping cores
    declare -A BASELINE_TIMES
    declare -A BASELINE_DOFS

    CORES_ARR=(${CORES_LIST})
    WEAK_REFINES_ARR=()
    if [ -n "${WEAK_REFINES}" ]; then
        WEAK_REFINES_ARR=(${WEAK_REFINES})
    fi

    for idx in "${!CORES_ARR[@]}"; do
        p="${CORES_ARR[$idx]}"

        if [ "${#WEAK_REFINES_ARR[@]}" -gt "$idx" ]; then
            ref="${WEAK_REFINES_ARR[$idx]}"
        else
            # In D dimensions, each refinement increases cells by 2^D.
            # Scale refinement with floor(log2(p) / D):
            delta_r=$(python3 -c "import math; print(int(math.floor(math.log2($p) / $DIM)))" 2>/dev/null || awk -v p="$p" -v d="$DIM" 'BEGIN{print int(log(p)/log(2)/d)}')
            ref=$((BASE_REFINE + delta_r))
        fi

        echo ""
        echo "--------------------------------------------------"
        echo " Weak Scaling: ${p} Processes / Cores (Refinement ${ref})"
        echo "--------------------------------------------------"

        RUN_CMD=""
        if command -v mpirun &>/dev/null && [ "$p" -gt 1 ]; then
            RUN_CMD="mpirun -np ${p}"
        elif command -v mpiexec &>/dev/null && [ "$p" -gt 1 ]; then
            RUN_CMD="mpiexec -n ${p}"
        fi

        for s in "${SOLVERS[@]}"; do
            echo "--> Running solver: ${s} on ${p} cores (refine ${ref})..."
            rm -f "${TEMP_LOG}"
            ${RUN_CMD} ${EXEC} --mode bench \
                               --dim "${DIM}" \
                               --refine "${ref}" \
                               --time "${FINAL_TIME}" \
                               --solver "${s}" 2>&1 | tee "${TEMP_LOG}" || true

            BASE_TIME=""
            BASE_DOF=""
            if [ -n "${BASELINE_TIMES[${s}]}" ]; then
                BASE_TIME="${BASELINE_TIMES[${s}]}"
                BASE_DOF="${BASELINE_DOFS[${s}]}"
            fi

            # Parse and save
            python3 scripts/parse_benchmark_output.py \
                --log "${TEMP_LOG}" \
                --csv "${OUTPUT_CSV}" \
                --mode scaling \
                --scaling-type weak \
                --dim "${DIM}" \
                --refine "${ref}" \
                --ranks "${p}" \
                ${BASE_TIME:+--baseline-time "${BASE_TIME}"} \
                ${BASE_DOF:+--baseline-dofs "${BASE_DOF}"}

            # Record baseline for p == 1
            if [ "$p" -eq 1 ]; then
                T1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-2)}' | tail -n 1 || true)
                D1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-4)}' | tail -n 1 || true)
                if [ -n "$T1" ]; then
                    BASELINE_TIMES["${s}"]="$T1"
                fi
                if [ -n "$D1" ]; then
                    BASELINE_DOFS["${s}"]="$D1"
                fi
            fi
        done
    done

else
    # Problem size scaling (fixed cores, sweeping mesh refinements)
    for ref in ${SIZE_REFINES}; do
        echo ""
        echo "--------------------------------------------------"
        echo " Problem Size Scaling: Refinement Level ${ref}"
        echo "--------------------------------------------------"

        RUN_CMD=""
        if command -v mpirun &>/dev/null && [ "$DETECTED_CORES" -gt 1 ]; then
            RUN_CMD="mpirun -np ${DETECTED_CORES}"
        fi

        for s in "${SOLVERS[@]}"; do
            echo "--> Running solver: ${s} (refine=${ref}, cores=${DETECTED_CORES})..."
            rm -f "${TEMP_LOG}"
            ${RUN_CMD} ${EXEC} --mode bench \
                               --dim "${DIM}" \
                               --refine "${ref}" \
                               --time "${FINAL_TIME}" \
                               --solver "${s}" 2>&1 | tee "${TEMP_LOG}" || true

            python3 scripts/parse_benchmark_output.py \
                --log "${TEMP_LOG}" \
                --csv "${OUTPUT_CSV}" \
                --mode scaling \
                --dim "${DIM}" \
                --refine "${ref}" \
                --ranks "${DETECTED_CORES}"
        done
    done
fi

rm -f "${TEMP_LOG}"

echo ""
echo "======================================================================"
echo " Scaling Study Finished!"
echo " Results saved to: ${OUTPUT_CSV}"
echo "======================================================================"
echo ""
echo "To plot the results, run in your visualization environment:"
echo "  python3 scripts/plot_results.py --csv ${OUTPUT_CSV} --type ${SCALING_TYPE}"
echo ""
