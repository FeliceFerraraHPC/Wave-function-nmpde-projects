/**
 * @file main.cpp
 * @brief Unified benchmark driver for three wave equation solvers.
 *
 * Usage:
 *   ./WaveBenchmark [options]
 *
 * Options:
 *   --mode     [bench|convergence|dispersion|both]  Default: bench
 *   --dim      [2|3]                      Default: 2
 *   --refine   N                          Global refinement level. Default: 6
 *   --time     T                          Final simulation time. Default: 45.0
 *   --solver   [all|theta|cg|dg]          Solvers to run. Default: all
 *   --gamma    G                          Damping coeff in u_tt-Delta u+gamma*u_t=0.
 *                                         Default: 0 (undamped, energy-conserving)
 *   --output                              Enable VTU output (disabled by default)
 *   --target-dofs N                       DOF target for dispersion p-study.
 *                                         Default: 16000
 *
 * In --mode bench:
 *   Runs each selected solver on the SAME globally-refined triangulation and
 *   prints a timing + DoF table to stdout.
 *
 * In --mode convergence:
 *   Runs the theta-scheme manufactured-solution convergence study on a
 *   sequence of meshes and prints convergence rates.
 *
 * In --mode dispersion:
 *   Runs the numerical dispersion analysis:
 *   (1) p-degree parametric study (WaveSolverTheta, p=1,2,4,6, matched DOFs).
 *   (2) CG (WaveSolverMatFree) vs DG (WaveSolverDG) comparison at p=4.
 *   Uses a Gaussian-modulated sinusoidal wave packet as initial condition and
 *   measures the phase error via L2 minimization over a time shift.
 *   Writes dispersion_results.csv to the run directory.
 *
 * In --mode both:
 *   Runs the benchmark first, then the convergence study.
 */

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_P4EST
#include <deal.II/distributed/tria.h>
#endif

#include <cctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "WaveFunctions.hpp"
#include "WaveSolverBase.hpp"
#include "WaveSolverDG.hpp"
#include "WaveSolverMatFree.hpp"
#include "WaveSolverTheta.hpp"
#include "dispersion_analysis.hpp"

using namespace dealii;

// ============================================================================
// Parse simple command-line arguments
// ============================================================================

struct ProgramOptions
{
  std::string mode = "bench";       // bench | convergence | dispersion | both
  int dim = 2;                      // spatial dimension (2 or 3)
  unsigned int refine = 6;          // global refinement levels
  double final_time = 45.0;         // final simulation time
  std::string solver = "all";       // all | theta | cg | dg
  double gamma = 0.0;               // damping coeff in u_tt - Delta u + gamma*u_t = 0
  bool write_output = true;         // write VTU files?
  unsigned int target_dofs = 16000; // matched DOF target for dispersion p-study
};

ProgramOptions
parse_args(int argc, char **argv)
{
  ProgramOptions opts;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    if (arg == "--mode" && i + 1 < argc)
      opts.mode = argv[++i];
    else if (arg == "--dim" && i + 1 < argc)
      opts.dim = std::stoi(argv[++i]);
    else if (arg == "--refine" && i + 1 < argc)
      opts.refine = std::stoul(argv[++i]);
    else if (arg == "--time" && i + 1 < argc)
      opts.final_time = std::stod(argv[++i]);
    else if (arg == "--solver" && i + 1 < argc)
      opts.solver = argv[++i];
    else if (arg == "--gamma" && i + 1 < argc)
      opts.gamma = std::stod(argv[++i]);
    else if (arg == "--output")
      opts.write_output = true;
    else if (arg == "--target-dofs" && i + 1 < argc)
      opts.target_dofs = static_cast<unsigned int>(std::stoul(argv[++i]));
  }
  return opts;
}

// ============================================================================
// Benchmark mode: run all selected solvers on the same mesh
// ============================================================================

template <int dim>
void run_benchmark(const ProgramOptions &opts)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << "========================================\n"
        << "  Wave Equation Benchmark  (dim=" << dim << ")\n"
        << "========================================\n"
        << "  Refinement levels : " << opts.refine << "\n"
        << "  Final time        : " << opts.final_time << "\n"
        << "  Damping (gamma)   : " << opts.gamma << "\n"
        << "  MPI ranks         : "
        << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << "\n"
        << "  Threads / rank    : " << MultithreadInfo::n_threads() << "\n"
        << std::endl;

  // --- Build the shared triangulation ---
#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
#else
  Triangulation<dim> tria;
#endif

  // Domain: [-15, 15]^dim — large enough for the Gaussian wave packet.
  GridGenerator::hyper_cube(tria, -15.0, 15.0);
  tria.refine_global(opts.refine);

  pcout << "   Global active cells : " << tria.n_global_active_cells() << "\n\n";

  // --- Initial conditions (shared by all solvers) ---
  const InitialDisplacement<dim> u0; // Gaussian wave packet
  const InitialVelocity<dim> v0;     // zero velocity

  // --- Collect solvers to run ---
  using SolverPtr = std::unique_ptr<WaveSolverBase<dim>>;
  std::vector<SolverPtr> solvers;

  if (opts.solver == "all" || opts.solver == "theta")
    solvers.emplace_back(std::make_unique<WaveSolverTheta<dim>>(
        /*fe_degree=*/4, /*theta=*/0.5, opts.gamma));

  if (opts.solver == "all" || opts.solver == "cg")
    solvers.emplace_back(std::make_unique<WaveSolverMatFree<dim>>(
        0.1 / WaveSolverMatFree<dim>::fe_degree, /*output_skip=*/100, opts.gamma));

  if (opts.solver == "all" || opts.solver == "dg")
    solvers.emplace_back(std::make_unique<WaveSolverDG<dim>>(
        0.05 / (WaveSolverDG<dim>::fe_degree * WaveSolverDG<dim>::fe_degree),
        /*output_skip=*/500, opts.gamma));

  // Header for results table
  pcout << std::left
        << std::setw(36) << "Solver"
        << std::setw(12) << "DoFs"
        << std::setw(14) << "Steps"
        << std::setw(18) << "Compute time (s)"
        << std::setw(18) << "Avg time/step (s)"
        << std::setw(16) << "E drift (rel)"
        << "\n"
        << std::string(114, '-') << "\n";

  // --- Run each solver ---
  for (auto &solver : solvers)
  {
    solver->setup(tria);
    solver->set_initial_conditions(u0, v0);

    const double wtime = solver->run(opts.final_time, opts.write_output);

    const unsigned int n_steps =
        static_cast<unsigned int>(opts.final_time / solver->time_step_size());

    // Relative energy drift: (E_final - E_initial) / E_initial.
    // For this homogeneous problem (no damping, no forcing) a correct,
    // stable scheme should keep this close to 0 for all final_time.
    const auto &history = solver->get_energy_history();
    double e_drift_rel = 0.0;
    if (history.size() >= 2 && history.front().total_energy != 0.0)
      e_drift_rel = (history.back().total_energy -
                     history.front().total_energy) /
                    history.front().total_energy;

    pcout << std::left
          << std::setw(36) << solver->name()
          << std::setw(12) << solver->n_dofs()
          << std::setw(14) << n_steps
          << std::setw(18) << std::fixed << std::setprecision(4) << wtime
          << std::setw(18) << (n_steps > 0 ? wtime / n_steps : 0.0)
          << std::setw(16) << std::scientific << std::setprecision(3)
          << e_drift_rel << std::fixed
          << "\n";

    // Sanitize the solver name into a filename and dump the full history.
    std::string tag = solver->name();
    for (char &c : tag)
      if (!std::isalnum(static_cast<unsigned char>(c)))
        c = '_';
    solver->export_energy_to_csv("energy_" + tag + ".csv");
  }

  pcout << std::string(114, '-') << "\n"
        << std::endl;
}

// ============================================================================
// Convergence mode: theta-scheme manufactured-solution test
// ============================================================================

template <int dim>
void run_convergence(const ProgramOptions &opts)
{
  static_assert(dim == 2, "Convergence study only implemented for dim=2.");

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << "========================================\n"
        << "  Theta-Scheme Convergence Study\n"
        << "========================================\n";

  WaveSolverTheta<dim> solver;
  const std::vector<unsigned int> levels = {5, 6, 7};
  solver.run_convergence_study(levels, opts.final_time, /*fe_degree=*/1);
}

// ============================================================================
// main()
// ============================================================================

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv, numbers::invalid_unsigned_int);

  try
  {
    const ProgramOptions opts = parse_args(argc, argv);

    if (opts.dim == 2)
    {
      if (opts.mode == "bench" || opts.mode == "both")
        run_benchmark<2>(opts);
      if (opts.mode == "convergence" || opts.mode == "both")
        run_convergence<2>(opts);
      if (opts.mode == "dispersion")
      {
        // Use a safe default of T=18 for the dispersion study unless the user
        // explicitly overrode --time (benchmark default of 45 is too long:
        // the wave packet would exit the domain at T≈23).
        const double T_disp = (opts.final_time != 45.0) ? opts.final_time : 18.0;
        run_dispersion<2>(/*k=*/2.0 * M_PI,
                          /*sigma=*/1.0,
                          /*x0=*/-8.0,
                          /*T_final=*/T_disp,
                          /*target_dofs=*/opts.target_dofs,
                          /*refine_cmp=*/opts.refine);
      }
    }
    else if (opts.dim == 3)
    {
      if (opts.mode == "bench" || opts.mode == "both")
        run_benchmark<3>(opts);
      // Convergence study in 3D not implemented (manufactured solution is 2D only)
      // Dispersion study in 3D not implemented (plane-wave setup is 2D only)
    }
    else
    {
      std::cerr << "Error: --dim must be 2 or 3.\n";
      return 1;
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << "\n----------------------------------------------------\n"
              << "Exception on processing:\n"
              << exc.what() << "\nAborting!\n"
              << "----------------------------------------------------\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "\n----------------------------------------------------\n"
              << "Unknown exception! Aborting!\n"
              << "----------------------------------------------------\n";
    return 1;
  }

  return 0;
}