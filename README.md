# Wave Equation Benchmark

Unified benchmarking framework for three wave equation solver strategies
using the [deal.II](https://www.dealii.org/) library (≥ 9.3.1).

## Solver Strategies

| Solver | Class | Method | Parallelism |
|---|---|---|---|
| **Theta-scheme** | `WaveSolverTheta` | θ-scheme (Crank-Nicolson, θ=0.5), assembled sparse + CG | MPI via TrilinosWrappers |
| **Matrix-Free CG** | `WaveSolverMatFree` | Leap-frog, matrix-free FE_Q | MPI + TBB + SIMD |
| **Matrix-Free DG** | `WaveSolverDG` | Leap-frog, matrix-free FE_DGQ + SIPG | MPI + TBB + SIMD |

All three solvers implement the `WaveSolverBase<dim>` abstract interface and
operate on the **same externally provided triangulation**, ensuring a fair
mesh-to-mesh comparison.

## Project Structure

```
Wave-function-nmpde-projects/
 CMakeLists.txt
 common/
   └── cmake-common.cmake     # Shared MPI / Boost / deal.II detection
 src/
    ├── WaveSolverBase.hpp     # Abstract base class
    ├── WaveFunctions.hpp      # Shared initial/boundary/forcing functions
    ├── WaveSolverTheta.hpp/cpp
    ├── WaveSolverMatFree.hpp/cpp
    ├── WaveSolverDG.hpp/cpp
    └── main.cpp               # Benchmark driver
```

## Building

```bash
module load gcc-glibc dealii
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running

### Benchmark mode (default)
Run all three solvers on the same mesh and print timing table:
```bash
./WaveBenchmark --mode bench --refine 6 --time 1.0 --solver all
```

### Single solver
```bash
./WaveBenchmark --solver theta    # theta-scheme only
./WaveBenchmark --solver cg       # matrix-free CG only
./WaveBenchmark --solver dg       # matrix-free DG only
```

### Convergence study (theta-scheme, manufactured exact solution)
```bash
./WaveBenchmark --mode convergence
```

### Both benchmark + convergence
```bash
./WaveBenchmark --mode both --refine 5 --time 1.0
```

### Enable VTU output
```bash
./WaveBenchmark --output
```

### MPI run
```bash
mpirun -n 4 ./WaveBenchmark --refine 7 --time 2.0
```

## Command-Line Options

| Option | Values | Default | Description |
|---|---|---|---|
| `--mode` | `bench`, `convergence`, `both` | `bench` | What to run |
| `--dim` | `2`, `3` | `2` | Spatial dimension |
| `--refine` | integer | `6` | Global mesh refinement level |
| `--time` | float | `1.0` | Final simulation time |
| `--solver` | `all`, `theta`, `cg`, `dg` | `all` | Solver(s) to benchmark |
| `--output` | flag | off | Write VTU files during simulation |

## Physical Problem

Homogeneous wave equation with Gaussian initial data:
```
u_tt - Δu = 0       on  Ω × (0, T)
u = 0               on  ∂Ω × (0, T)
u(x, 0)  = exp(-|x|²/2)
u_t(x,0) = 0
```
Domain: `[-15, 15]^dim` (large enough for the wave packet).

The theta-scheme convergence study uses a manufactured exact solution
`u = t² sin(πx) sin(πy)` on `[0,1]²` to verify second-order convergence.
