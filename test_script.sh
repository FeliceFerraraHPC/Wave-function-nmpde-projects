#!/bin/bash

# Exit on error
set -e

# ============================================================
# Usage: ./test_script.sh [--step bench|convergence|output|all]
#                         [--time T]  (default: 1.0)
# ============================================================
STEP="all"
FINAL_TIME="1.0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) STEP="$2"; shift 2 ;;
        --time) FINAL_TIME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Compile the project first if needed
echo "Building the project..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

EXEC="./build/WaveBenchmark"

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
        OUT_DIR="results/benchmark/dim${d}/${s}"
        mkdir -p "${OUT_DIR}"
        echo "Running benchmark for dim=${d}, solver=${s} with refine=${REFINE}..."
        
        cat > params.prm <<EOF
set mode = bench
set dim = ${d}
set solver = ${s}
set refine = ${REFINE}
set final_time = ${FINAL_TIME}
set write_output = false
set output_frequency = 100
EOF
        
        ${EXEC} params.prm > "${OUT_DIR}/bench_output.txt"
    done
done
}

# ============================================================
# Step 2: Convergence Studies
# ============================================================
run_convergence() {
echo ">>> Step 2: Running Convergence Studies (dim=2)"
for s in "${SOLVERS[@]}"; do
    OUT_DIR="results/convergence/${s}"
    mkdir -p "${OUT_DIR}"
    echo "Running convergence for solver=${s}..."
    
    cat > params.prm <<EOF
set mode = convergence
set dim = 2
set solver = ${s}
set levels = 5-7
set final_time = ${FINAL_TIME}
set write_output = false
set output_frequency = 100
EOF
    
    ${EXEC} params.prm > "${OUT_DIR}/convergence_output.txt"
    if [ -f "convergence_${s}.csv" ]; then
        mv "convergence_${s}.csv" "${OUT_DIR}/"
    fi
done
}

# ============================================================
# Step 3: Visualization Output
# output_frequency=10 gives ~10-100 snapshots depending on solver
# ============================================================
run_output() {
echo ">>> Step 3: Generating PVTU Output"
for d in "${DIMS[@]}"; do
    if [ "$d" -eq 3 ]; then
        REFINE=3 # Use a smaller mesh for 3D output to save disk space
    else
        REFINE=5 # Slightly smaller for 2D output
    fi
    for s in "${SOLVERS[@]}"; do
        OUT_DIR="results/output/dim${d}/${s}"
        rm -rf "${OUT_DIR}"
        mkdir -p "${OUT_DIR}"
        echo "Running output generation for dim=${d}, solver=${s} with refine=${REFINE}..."
        
        # We must create params.prm in the output directory
        cat > "${OUT_DIR}/params.prm" <<EOF
set mode = bench
set dim = ${d}
set solver = ${s}
set refine = ${REFINE}
set final_time = ${FINAL_TIME}
set write_output = true
set output_frequency = 10
EOF
        
        # Change directory to OUT_DIR so files are written there
        cd "${OUT_DIR}"
        ../../../../${EXEC} params.prm > run_log.txt
        cd ../../../../
    done
done
}

# ============================================================
# Dispatch based on --step argument
# ============================================================
case "${STEP}" in
    bench)       run_benchmarks ;;
    convergence) run_convergence ;;
    output)      run_output ;;
    all)
        run_benchmarks
        run_convergence
        run_output
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: ./test_script.sh [--step bench|convergence|output|all]"
        exit 1
        ;;
esac

echo "=========================================="
echo " Testing completed. Results saved in results/"
echo "=========================================="
