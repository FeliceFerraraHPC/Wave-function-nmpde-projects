# Wave Equation Benchmarking Suite

A comprehensive C++ framework built on [deal.II](https://www.dealii.org/) for solving the time-dependent acoustic wave equation:

$$
\begin{cases}
\partial_{tt}u + \gamma \partial_t u - \Delta u = f & \text{in } \Omega \times (0,T) \\
u = g_D & \text{on } \Gamma_D \times (0,T) \\
\partial_n u = g_N & \text{on } \Gamma_N \times (0,T) \\
u(\boldsymbol{x},0) = u_0(\boldsymbol{x}) & \text{in } \Omega \\
\partial_t u(\boldsymbol{x},0) = v_0(\boldsymbol{x}) & \text{in } \Omega
\end{cases}
$$

*(where γ∂<sub>t</sub>u is an optional damping term, and the boundary ∂Ω is partitioned into Dirichlet Γ<sub>D</sub> and Neumann Γ<sub>N</sub> sections)*

This repository provides a unified benchmarking suite to directly compare different numerical schemes, focusing on computational efficiency, parallel scalability, and numerical dispersion.

## 🚀 Features
- **Solvers**:
  - `theta`: Assembled Sparse Matrix (Theta-scheme Crank-Nicolson, Trilinos MPI)
  - `cg`: Matrix-Free Continuous Galerkin (CG) with SIMD tensor-product evaluation
  - `dg`: Matrix-Free Discontinuous Galerkin (SIPG) with parallel face integrals
- **Modes**:
  - Benchmarking (`bench`)
  - Convergence Studies (`convergence`)
  - Numerical Dispersion Analysis (`dispersion`)
- **Parallelization**: Fully supports MPI + TBB + SIMD for performance.

## 📂 Repository Structure

- `src/` - Contains all C++ source files and headers:
  - `WaveSolverBase.hpp`: Abstract base class defining the standard solver interface.
  - `WaveSolverTheta.*`: Assembled Sparse Matrix (Theta-scheme) implementation.
  - `WaveSolverMatFree.*`: Matrix-Free CG implementation.
  - `WaveSolverDG.*`: Matrix-Free DG (SIPG) implementation.
  - `WaveFunctions.hpp`: Defines initial conditions and exact solutions (Gaussian packet, acoustic pulse, etc).
  - `main.cpp`: The unified executable driver.
- `docs/` - Detailed markdown documentation for the mathematical formulations and specific solver implementations.
- `report.txt` - Summary report on mathematical features and implementations.
- `test_script.sh` - Automated shell script for running standardized testing and output generation.

## 🌳 Git Branching Model

This repository is actively developed across several branches to isolate specific features and performance tests:

- **`main`**: The primary stable branch containing the unified benchmarking suite and fully integrated core solvers (`theta`, `cg`, `dg`).
- **`scaling`**: Dedicated branch containing Python scripts and automated shell scripts (e.g., `compare_matfree_vs_sparse.sh`) for comprehensive strong-scaling benchmarks, designed to be run locally or on HPC clusters.
- **`Energy`**: Experimental branch likely exploring specific energy-conserving properties or analyzing energy drift in detail.

## ⚙️ Dependencies

Ensure the following are installed and loaded in your environment:
- **C++17** compatible compiler (e.g., GCC)
- **CMake** (>= 3.10)
- **deal.II** library (>= 9.3) compiled with MPI and p4est support
- **MPI** (OpenMPI, MPICH, etc.)

## 🛠️ Building the Program

Binary files must not be uploaded to the repository. The project uses standard CMake out-of-source builds:

```bash
# Create a build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile using all available CPU cores
make -j$(nproc)
```

## 🏃 Running the Program

The main executable `WaveBenchmark` will be created inside the `build/` directory. 

### Running sequentially
```bash
./build/WaveBenchmark --mode bench --solver all --dim 2 --refine 5
```

### Running in Parallel (MPI)
For high-performance runs, simply invoke the executable with `mpirun` or `mpiexec`:
```bash
mpirun -np 4 ./build/WaveBenchmark --mode bench --solver dg --dim 2 --refine 5
```

### Using the Automated Test Script
The repository includes an automated testing script to sequentially execute standard benchmarks, convergence studies, and generate `.vtu` output for ParaView.
```bash
./test_script.sh --step bench --time 0.05
```

## 🎛️ Command-Line Options

The `WaveBenchmark` executable accepts several arguments to customize the simulation:

| Option | Description | Default |
| :--- | :--- | :--- |
| `--mode` | Execution mode (`bench`, `convergence`, `dispersion`, `both`) | `bench` |
| `--dim` | Spatial dimension (`2` or `3`) | `2` |
| `--refine` | Global mesh refinement level | `6` |
| `--time` | Final simulation time | `45.0` |
| `--solver` | Solvers to run (`all`, `theta`, `cg`, `dg`) | `all` |
| `--gamma` | Damping coefficient $\gamma$ | `0.0` |
| `--bc` | Boundary condition (`dirichlet`, `neumann`) | `dirichlet` |
| `--wave` | Initial wave profile (`default`, `acoustic`, `pulse`) | `default` |
| `--output` | Write VTU visualization files | (Disabled) |
| `--target-dofs` | DOF target for dispersion p-study | `16000` |
| `--non-homogeneous` | Enable non-homogeneous (time-dependent) boundary conditions | (Disabled) |

*Example: Running a 3D dispersion analysis using the CG solver with VTU output enabled:*
```bash
mpirun -np 8 ./build/WaveBenchmark --mode dispersion --dim 3 --solver cg --output
```
