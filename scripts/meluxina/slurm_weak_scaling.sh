#!/bin/bash -l
#SBATCH --job-name=wave_weak_scaling
#SBATCH --account=p201574
#SBATCH --qos=default
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=results/meluxina/slurm_weak_scaling_%j.out
#SBATCH --error=results/meluxina/slurm_weak_scaling_%j.err

# ==============================================================================
# MeluXina HPC Slurm Job: Weak Scaling Study across MPI Ranks
#
# Scales problem size proportionally to MPI rank count (N/p ~ const):
#   In 2D:
#     1 MPI rank   -> Refinement 5 (~16,000 DoFs)
#     4 MPI ranks  -> Refinement 6 (~66,000 DoFs)
#    16 MPI ranks  -> Refinement 7 (~260,000 DoFs)
#    64 MPI ranks  -> Refinement 8 (~1,050,000 DoFs)
#
# Solvers compared:
#   - Matrix-Free CG
#   - Assembled Sparse Matrix (Theta-scheme)
#
# Output CSV: results/meluxina/meluxina_weak_scaling.csv
#
# Submit via:
#   sbatch scripts/meluxina/slurm_weak_scaling.sh
#   sbatch --account=p201574 --qos=default scripts/meluxina/slurm_weak_scaling.sh
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

OUTPUT_CSV="${OUT_DIR}/meluxina_weak_scaling.csv"
rm -f "${OUTPUT_CSV}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export KOKKOS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_pml='ucx'
export OMPI_MCA_btl='^uct,ofi'
export OMPI_MCA_mtl='^ofi'

DIM=2
FINAL_TIME="0.5"
RANKS_SWEEP=(1 4 16 64)
REFINES_SWEEP=(5 6 7 8)
SOLVERS=("cg" "theta")

echo "======================================================================"
echo " MeluXina: Weak Scaling Study"
echo " Dimension         : ${DIM}D"
echo " Sweeping Ranks    : ${RANKS_SWEEP[*]}"
echo " Refinements       : ${REFINES_SWEEP[*]}"
echo " Solvers           : ${SOLVERS[*]}"
echo " Simulation Time   : ${FINAL_TIME} s"
echo " Output CSV        : ${OUTPUT_CSV}"
echo "======================================================================"

TEMP_LOG="$(mktemp -t meluxina_weak_scaling_XXXXXX.log 2>/dev/null || mktemp /tmp/meluxina_weak_scaling_XXXXXX.log)"

declare -A BASELINE_TIMES
declare -A BASELINE_DOFS

for idx in "${!RANKS_SWEEP[@]}"; do
    p="${RANKS_SWEEP[$idx]}"
    ref="${REFINES_SWEEP[$idx]}"

    echo ""
    echo "=================================================="
    echo ">>> Running with ${p} MPI Ranks (Refinement ${ref})"
    echo "=================================================="

    for s in "${SOLVERS[@]}"; do
        echo "--> Testing solver '${s}' on ${p} MPI ranks (refine ${ref})..."
        rm -f "${TEMP_LOG}"
        mpirun -x OMP_NUM_THREADS -x KOKKOS_NUM_THREADS -np ${p} \
            ${EXEC} --mode bench \
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

        # Parse output into CSV
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

        # Store baseline for p == 1
        if [ "$p" -eq 1 ]; then
            T1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-2)}' | tail -n 1 || true)
            D1=$(grep -E 'Theta-scheme|Matrix-Free CG|Matrix-Free DG' "${TEMP_LOG}" | awk '{print $(NF-4)}' | tail -n 1 || true)
            if [ -n "$T1" ]; then
                BASELINE_TIMES["${s}"]="$T1"
                echo "    [Recorded ${s} 1-rank baseline time: ${T1} s]"
            fi
            if [ -n "$D1" ]; then
                BASELINE_DOFS["${s}"]="$D1"
                echo "    [Recorded ${s} 1-rank baseline DoFs: ${D1}]"
            fi
        fi
    done
done

rm -f "${TEMP_LOG}"

echo ""
echo "======================================================================"
echo " Weak Scaling Study Completed!"
echo " Results written to: ${OUTPUT_CSV}"
echo "======================================================================"
echo ""
echo "To plot the results, run in your visualization environment:"
echo "  python3 scripts/plot_results.py --csv ${OUTPUT_CSV} --type weak"
echo ""
