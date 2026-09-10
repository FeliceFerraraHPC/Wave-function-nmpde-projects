#!/bin/bash
# ==============================================================================
# MeluXina Supercomputer Environment Configuration
#
# Configures the MeluXina User Software Environment (MUSE) 2023 stack:
#   - deal.II 9.5.2 (built with Trilinos and foss-2023a)
#   - GCC 12.3.0 & OpenMPI 4.1.5
#   - CMake
#   - SuiteSparse 7.1.0 & 5.13.0 runtime dynamic link paths
#
# Usage:
#   source scripts/meluxina/load_modules.sh
# ==============================================================================

echo ">>> Loading modules on MeluXina..."

# 0. Ensure 'module' command is available (for non-login shells / Slurm batch jobs)
if ! command -v module &>/dev/null; then
    for init_script in \
        /etc/profile.d/modules.sh \
        /etc/profile.d/z00_lmod.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash \
        /apps/USE/easybuild/software/Lmod/*/init/bash \
        /etc/profile; do
        if [ -f "${init_script}" ]; then
            source "${init_script}" 2>/dev/null || true
            if command -v module &>/dev/null; then
                break
            fi
        fi
    done
fi

# 1. Load MUSE 2023.1 stack (staging or release)
module load env/staging/2023.1 2>/dev/null || module load env/release/2023.1 2>/dev/null || true

# 2. Load deal.II with Trilinos support (pulls in GCC 12.3.0, OpenMPI, Boost, etc.)
module load deal.II/9.5.2-foss-2023a-trilinos

# 3. Load CMake
module load CMake

# 4. Export SuiteSparse library paths (needed for both deal.II and Trilinos runtime)
SS_BASE="/apps/USE/easybuild/release/2023.1/software/SuiteSparse"
SS_7="${SS_BASE}/7.1.0-foss-2023a/lib"
SS_5="${SS_BASE}/5.13.0-foss-2023a-METIS-5.1.0/lib"

if [ -d "${SS_7}" ] && [ -d "${SS_5}" ]; then
    export LD_LIBRARY_PATH="${SS_7}:${SS_5}:${LD_LIBRARY_PATH}"
fi

# 5. OpenMP and TBB runtime settings tuned for MeluXina AMD EPYC Rome nodes
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
unset KOKKOS_NUM_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 6. Slurm step overlap support for interactive sessions
export SLURM_OVERLAP=1

# 7. OpenMPI & UCX settings (fixes 'Failed to modify UD QP to INIT on mlx5_0: Operation not permitted')
export OMPI_MCA_pml='ucx'
export OMPI_MCA_btl='^uct,ofi'
export OMPI_MCA_mtl='^ofi'

echo ">>> Environment ready:"
echo "  deal.II:     9.5.2 (Trilinos)"
echo "  CMake:       $(which cmake 2>/dev/null || echo 'Not found')"
echo "  CXX:         $(which mpicxx 2>/dev/null || which g++ 2>/dev/null || echo 'Not found')"
echo "  MPI launcher:$(which mpirun 2>/dev/null || which srun 2>/dev/null || echo 'Not found')"
