#!/bin/bash -l
#SBATCH --job-name=wave_compare
#SBATCH --account=p201574
#SBATCH --qos=default
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=results/meluxina/slurm_compare_%j.out
#SBATCH --error=results/meluxina/slurm_compare_%j.err

# ==============================================================================
# MeluXina HPC Slurm Job: Matrix-Free vs. Assembled Sparse Matrix Comparison
#
# Direct comparison of wall-clock compute time between:
#   - Matrix-Free CG (FEEvaluation cell_loop, SIMD tensor-product)
#   - Assembled Sparse Matrix (Theta-scheme Crank-Nicolson)
#
# Sweeps mesh refinement levels in 2D and 3D.
# Output CSV: results/meluxina/meluxina_matfree_vs_sparse.csv
#
# Submit via:
#   sbatch scripts/meluxina/slurm_compare_matfree_vs_sparse.sh
#   sbatch --account=p201574 --qos=default scripts/meluxina/slurm_compare_matfree_vs_sparse.sh
# ==============================================================================

set -e

source scripts/meluxina/load_modules.sh

# Compile project
echo ">>> Building project on MeluXina compute node..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j16
cd ..

EXEC="./build/WaveBenchmark"
OUT_DIR="results/meluxina"
mkdir -p "${OUT_DIR}"

OUTPUT_CSV="${OUT_DIR}/meluxina_matfree_vs_sparse.csv"
rm -f "${OUTPUT_CSV}"

RANKS=${SLURM_NTASKS:-32}
FINAL_TIME="0.5"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export KOKKOS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMPI_MCA_pml='ucx'
export OMPI_MCA_btl='^uct,ofi'
export OMPI_MCA_mtl='^ofi'

REFINES_2D=(5 6 7)
REFINES_3D=(3 4)

echo "======================================================================"
echo " MeluXina: Direct Comparison Matrix-Free vs. Assembled Sparse Matrix"
echo " MPI Ranks         : ${RANKS}"
echo " 2D Refinements    : ${REFINES_2D[*]}"
echo " 3D Refinements    : ${REFINES_3D[*]}"
echo " Simulation Time   : ${FINAL_TIME} s"
echo " Output CSV        : ${OUTPUT_CSV}"
echo "======================================================================"

TEMP_LOG="$(mktemp -t meluxina_bench_XXXXXX.log 2>/dev/null || mktemp /tmp/meluxina_bench_XXXXXX.log)"

run_comparison() {
    local d="$1"
    shift
    local refs=("$@")

    echo ""
    echo ">>> Running ${d}D Comparison across Refinements: ${refs[*]}"

    for ref in "${refs[@]}"; do
        echo "--------------------------------------------------"
        echo " Testing ${d}D, Refinement = ${ref}"
        echo "--------------------------------------------------"

        rm -f "${TEMP_LOG}"

        # Run each solver (Assembled Sparse, Matrix-Free CG)
        for s in "theta" "cg"; do
            echo "--> Running solver '${s}' on MeluXina (dim=${d}, refine=${ref})..."
            mpirun -x OMP_NUM_THREADS -x KOKKOS_NUM_THREADS -np ${RANKS} \
                ${EXEC} --mode bench \
                        --dim "${d}" \
                        --refine "${ref}" \
                        --time "${FINAL_TIME}" \
                        --solver "${s}" 2>&1 | tee -a "${TEMP_LOG}" || true
        done

        # Parse table directly into structured CSV
        python3 scripts/parse_benchmark_output.py \
            --log "${TEMP_LOG}" \
            --csv "${OUTPUT_CSV}" \
            --mode comparison \
            --dim "${d}" \
            --refine "${ref}" \
            --ranks "${RANKS}"
    done
}

# Run 2D and 3D sweeps
run_comparison 2 "${REFINES_2D[@]}"
run_comparison 3 "${REFINES_3D[@]}"

rm -f "${TEMP_LOG}"

echo ""
echo "======================================================================"
echo " Direct Comparison Completed!"
echo " Results written to: ${OUTPUT_CSV}"
echo "======================================================================"

echo ""
echo ">>> Plotting is separate. To generate comparison plots, run:"
echo "    python3 scripts/plot_results.py --csv ${OUTPUT_CSV} --type comparison"
echo ""
