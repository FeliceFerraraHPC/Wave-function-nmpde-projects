#!/bin/bash -l
#SBATCH --job-name=wave_scaling
#SBATCH --account=p201574
#SBATCH --qos=default
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=results/meluxina/slurm_scaling_%j.out
#SBATCH --error=results/meluxina/slurm_scaling_%j.err

# ==============================================================================
# MeluXina HPC Slurm Job: Strong Scaling Study across MPI Ranks
#
# Sweeps MPI rank counts: 1, 2, 4, 8, 16, 32, 64 on a dual AMD EPYC 7H12 node.
# Measures:
#   - Wall-clock compute time T(p)
#   - Strong scaling speedup S(p) = T(1) / T(p)
#   - Parallel efficiency E(p) = S(p) / p * 100%
#
# Solvers compared:
#   - Matrix-Free CG
#   - Matrix-Free DG
#   - Assembled Sparse Matrix (Theta-scheme)
#
# Output CSV: results/meluxina/meluxina_strong_scaling.csv
#
# Submit via:
#   sbatch scripts/meluxina/slurm_scaling.sh
#   sbatch --account=p201574 --qos=default scripts/meluxina/slurm_scaling.sh
# ==============================================================================

set -e

source scripts/meluxina/load_modules.sh

echo ">>> Building project on MeluXina compute node..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
cd ..

EXEC="./build/WaveBenchmark"
OUT_DIR="results/meluxina"
mkdir -p "${OUT_DIR}"

OUTPUT_CSV="${OUT_DIR}/meluxina_strong_scaling.csv"
rm -f "${OUTPUT_CSV}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export KOKKOS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_pml='ucx'
export OMPI_MCA_btl='^uct,ofi'
export OMPI_MCA_mtl='^ofi'

DIM=2
REFINE=7          # ~260,000 DoFs: large enough to show strong scaling up to 64 ranks
FINAL_TIME="0.5"
RANKS_SWEEP=(1 2 4 8 16 32 64)
SOLVERS=("cg" "theta" "dg")

echo "======================================================================"
echo " MeluXina: Strong Scaling Study"
echo " Dimension         : ${DIM}D"
echo " Refinement Level  : ${REFINE}"
echo " Sweeping Ranks    : ${RANKS_SWEEP[*]}"
echo " Solvers           : ${SOLVERS[*]}"
echo " Simulation Time   : ${FINAL_TIME} s"
echo " Output CSV        : ${OUTPUT_CSV}"
echo "======================================================================"

TEMP_LOG="$(mktemp -t meluxina_scaling_XXXXXX.log 2>/dev/null || mktemp /tmp/meluxina_scaling_XXXXXX.log)"

declare -A BASELINE_TIMES

for p in "${RANKS_SWEEP[@]}"; do
    echo ""
    echo "=================================================="
    echo ">>> Running with ${p} MPI Ranks"
    echo "=================================================="

    for s in "${SOLVERS[@]}"; do
        echo "--> Testing solver '${s}' on ${p} MPI ranks..."
        mpirun -x OMP_NUM_THREADS -x KOKKOS_NUM_THREADS -np ${p} \
            ${EXEC} --mode bench \
                    --dim "${DIM}" \
                    --refine "${REFINE}" \
                    --time "${FINAL_TIME}" \
                    --solver "${s}" | tee "${TEMP_LOG}"

        BASE_TIME=""
        if [ -n "${BASELINE_TIMES[${s}]}" ]; then
            BASE_TIME="${BASELINE_TIMES[${s}]}"
        fi

        # Parse output into CSV
        python3 scripts/parse_benchmark_output.py \
            --log "${TEMP_LOG}" \
            --csv "${OUTPUT_CSV}" \
            --mode scaling \
            --dim "${DIM}" \
            --refine "${REFINE}" \
            --ranks "${p}" \
            ${BASE_TIME:+--baseline-time "${BASE_TIME}"}

        # Store baseline for p == 1
        if [ "$p" -eq 1 ]; then
            T1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-2)}' | tail -n 1 || true)
            if [ -n "$T1" ]; then
                BASELINE_TIMES["${s}"]="$T1"
                echo "    [Recorded ${s} 1-rank baseline: ${T1} s]"
            fi
        fi
    done
done

rm -f "${TEMP_LOG}"

echo ""
echo "======================================================================"
echo " Strong Scaling Study Completed!"
echo " Results written to: ${OUTPUT_CSV}"
echo "======================================================================"

# Generate scaling plots
if command -v python3 &>/dev/null; then
    PLOT_DIR="${OUT_DIR}/plots"
    echo ">>> Generating scaling plots in: ${PLOT_DIR}..."
    python3 scripts/plot_results.py --csv "${OUTPUT_CSV}" --outdir "${PLOT_DIR}" || true
fi
