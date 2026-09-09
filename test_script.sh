#!/bin/bash

# Exit on error
set -e

# ============================================================
# Usage: ./test_script.sh [--step bench|convergence|dispersion|output|all]
#                         [--time T]    (default: 1.0)
#                         [--gamma G]   (default: 0.0)
#                         [--bc BC]     (default: dirichlet)
#                         [--wave WAVE] (default: default)
# ============================================================
STEP="all"
FINAL_TIME="1.0"
GAMMA="0.0"
BC="dirichlet"
WAVE="default"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)  STEP="$2";       shift 2 ;;
        --time)  FINAL_TIME="$2"; shift 2 ;;
        --gamma) GAMMA="$2";      shift 2 ;;
        --bc)    BC="$2";         shift 2 ;;
        --wave)  WAVE="$2";       shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Compile the project
echo "Building the project..."
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

EXEC="./WaveBenchmark"

SOLVERS=("theta" "cg" "dg")
DIMS=(2 3)

echo "=========================================="
echo " Starting Unified Testing Script (step=${STEP})"
echo "=========================================="

# ============================================================
# Step 1: Benchmarks
# ============================================================
run_benchmarks() {
    echo ">>> Step 1: Running Benchmarks"
    for d in "${DIMS[@]}"; do
        if [ "$d" -eq 3 ]; then
            REFINE=4
        else
            REFINE=6
        fi

        for s in "${SOLVERS[@]}"; do
            echo "Running benchmark for dim=${d}, solver=${s}, refine=${REFINE}, gamma=${GAMMA}, bc=${BC}, wave=${WAVE}..."
            "${EXEC}" --mode bench \
                      --dim "${d}" \
                      --solver "${s}" \
                      --refine "${REFINE}" \
                      --time "${FINAL_TIME}" \
                      --gamma "${GAMMA}" \
                      --bc "${BC}" \
                      --wave "${WAVE}"

            # Tag generated energy CSV with dimension to keep both 2D and 3D logs
            for f in energy_*.csv; do
                if [ -f "$f" ] && [[ "$f" != *_dim* ]]; then
                    mv "$f" "${f%.csv}_dim${d}.csv"
                fi
            done
        done
    done
}

# ============================================================
# Step 2: Convergence Studies (Manufactured Solution, dim=2)
# ============================================================
run_convergence() {
    echo ">>> Step 2: Running Convergence Studies (dim=2, bc=${BC})"
    for s in "${SOLVERS[@]}"; do
        echo "Running convergence study for solver=${s}..."
        "${EXEC}" --mode convergence \
                  --solver "${s}" \
                  --dim 2 \
                  --time "${FINAL_TIME}" \
                  --bc "${BC}"
    done
}

# ============================================================
# Step 3: Numerical Dispersion Analysis (dim=2)
# ============================================================
run_dispersion() {
    echo ">>> Step 3: Running Numerical Dispersion Analysis (dim=2)"
    echo "Running dispersion analysis..."
    "${EXEC}" --mode dispersion \
              --dim 2 \
              --time "${FINAL_TIME}"
}

# ============================================================
# Step 4: Visualization Output (.vtu / .pvtu)
# ============================================================
run_output() {
    echo ">>> Step 4: Generating VTU/PVTU Output"
    for d in "${DIMS[@]}"; do
        if [ "$d" -eq 3 ]; then
            REFINE=3 # Lower mesh resolution for 3D snapshots
        else
            REFINE=5
        fi

        for s in "${SOLVERS[@]}"; do
            echo "Generating visualization output for dim=${d}, solver=${s}, refine=${REFINE}, bc=${BC}, wave=${WAVE}..."
            "${EXEC}" --mode bench \
                      --dim "${d}" \
                      --solver "${s}" \
                      --refine "${REFINE}" \
                      --time "${FINAL_TIME}" \
                      --gamma "${GAMMA}" \
                      --bc "${BC}" \
                      --wave "${WAVE}" \
                      --output
        done
    done
}

# ============================================================
# Dispatch based on --step argument
# ============================================================
case "${STEP}" in
    bench)       run_benchmarks ;;
    convergence) run_convergence ;;
    dispersion)  run_dispersion ;;
    output)      run_output ;;
    all)
        run_benchmarks
        run_convergence
        run_dispersion
        run_output
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: ./test_script.sh [--step bench|convergence|dispersion|output|all] [--time T] [--gamma G]"
        cd ..
        exit 1
        ;;
esac

cd ..

echo "=========================================="
echo " Testing completed. All output files are in build/"
echo "=========================================="