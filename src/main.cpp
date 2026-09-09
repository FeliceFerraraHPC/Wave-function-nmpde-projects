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
  bool user_refine = false;         // was --refine explicitly specified?
  double final_time = 45.0;         // final simulation time
  std::string solver = "all";       // all | theta | cg | dg
  double gamma = 0.0;               // damping coeff in u_tt - Delta u + gamma*u_t = 0
  bool write_output = true;         // write VTU files?
  unsigned int target_dofs = 16000; // matched DOF target for dispersion p-study
  bool use_spatial_scaling = true;  // dt ∝ h^2.5 for p=4 in convergence mode
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
    {
      opts.refine = std::stoul(argv[++i]);
      opts.user_refine = true;
    }
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
    else if (arg == "--cfl-scaling")
      opts.use_spatial_scaling = false;
    else if (arg == "--spatial-scaling")
      opts.use_spatial_scaling = true;
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

  // Domain: [-15, 15]^dim -- large enough for the Gaussian wave packet.
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
// ============================================================================
// Convergence mode: Unified Method of Manufactured Solutions (MMS)
// ============================================================================
struct ConvergenceResult
{
  unsigned int            refinement = 0;
  double                  h = 0.0;
  types::global_dof_index n_dofs = 0;
  double                  l2_error = 0.0;
  double                  l2_eoc = 0.0;
  double                  h1_error = 0.0;
  double                  h1_eoc = 0.0;
};

template <int dim>
void run_convergence_for_solver(
    const std::string &solver_tag,
    const std::vector<unsigned int> &levels,
    double final_time,
    bool use_spatial_scaling,
    double gamma)
{
  const bool is_root = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  ConditionalOStream pcout(std::cout, is_root);

  std::string solver_title;
  if (solver_tag == "cg")
    solver_title = "Matrix-Free Continuous Galerkin (CG, p=4)";
  else if (solver_tag == "dg")
    solver_title = "Matrix-Free Discontinuous Galerkin (SIPG, p=4)";
  else
    solver_title = "Trilinos Theta-Scheme (Crank-Nicolson, p=4)";

  pcout << "\n========================================================================================\n"
        << "  MMS Convergence Study: " << solver_title << "\n"
        << "  Domain: [0, pi]^" << dim << ", Final time T = " << final_time << "\n"
        << "  Exact solution: u(x, t) = cos(sqrt(" << dim << ") * t) * prod sin(x_d)\n"
        << "  Scaling: " << (use_spatial_scaling ? "Strict Spatial (dt ∝ h^2.5 for p=4)" : "Standard CFL (dt = CFL * h)") << "\n"
        << "  Expected: L2 EOC = " << (use_spatial_scaling ? "5.0 (O(h^5))" : "2.0 (temporal O(dt^2))")
        << " | H1 EOC = " << (use_spatial_scaling ? "4.0 (O(h^4))" : "2.0") << "\n"
        << "========================================================================================\n\n";

  std::vector<ConvergenceResult> results;
  results.reserve(levels.size());

  const StandingWaveIC<dim> u0;
  const StandingWaveV0<dim> v0;

  for (size_t i = 0; i < levels.size(); ++i)
  {
    const unsigned int ref = levels[i];
    pcout << ">>> Running Refinement Level " << ref << "..." << std::flush;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
#else
    Triangulation<dim> tria;
#endif
    GridGenerator::hyper_cube(tria, 0.0, numbers::PI);
    tria.refine_global(ref);

    std::unique_ptr<WaveSolverBase<dim>> solver;
    double cfl = 0.025;
    if (solver_tag == "cg")
    {
      cfl = 0.1 / WaveSolverMatFree<dim>::fe_degree;
      solver = std::make_unique<WaveSolverMatFree<dim>>(cfl, /*output_skip=*/10000, gamma);
    }
    else if (solver_tag == "dg")
    {
      cfl = 0.05 / (WaveSolverDG<dim>::fe_degree * WaveSolverDG<dim>::fe_degree);
      solver = std::make_unique<WaveSolverDG<dim>>(cfl, /*output_skip=*/10000, gamma);
    }
    else // theta
    {
      cfl = 0.25;
      solver = std::make_unique<WaveSolverTheta<dim>>(/*fe_degree=*/4, /*theta=*/0.5, gamma);
    }

    solver->setup(tria);

    const double local_min = tria.last()->diameter() / std::sqrt(double(dim));
    const double h_cell = -Utilities::MPI::max(-local_min, MPI_COMM_WORLD);

    double dt = 0.0;
    if (use_spatial_scaling)
    {
      const double h_ref = numbers::PI / 4.0; // level 2 reference: h = pi / 4
      const double dt_ref = cfl * h_ref;
      dt = dt_ref * std::pow(h_cell / h_ref, 2.5); // dt ∝ h^2.5
    }
    else
    {
      dt = cfl * h_cell;
    }

    const unsigned int n_steps =
        std::max(1u, static_cast<unsigned int>(std::ceil(final_time / dt)));
    dt = final_time / n_steps;
    solver->set_time_step(dt);

    StandingWaveExact<dim> u_prev(-dt);
    solver->set_initial_conditions(u0, v0, &u_prev);

    solver->run(final_time, /*write_output=*/false);

    StandingWaveExact<dim> sol_exact(final_time);
    const double e_L2 = solver->compute_error(VectorTools::L2_norm, sol_exact);
    const double e_H1 = solver->compute_error(VectorTools::H1_seminorm, sol_exact);

    ConvergenceResult res;
    res.refinement = ref;
    res.h          = h_cell;
    res.n_dofs     = solver->n_dofs();
    res.l2_error   = e_L2;
    res.l2_eoc     = 0.0;
    res.h1_error   = e_H1;
    res.h1_eoc     = 0.0;

    if (i > 0)
    {
      const double log_h_ratio = std::log(results[i - 1].h / res.h);
      res.l2_eoc = std::log(results[i - 1].l2_error / res.l2_error) / log_h_ratio;
      res.h1_eoc = std::log(results[i - 1].h1_error / res.h1_error) / log_h_ratio;
    }

    results.push_back(res);

    pcout << " DOFs: " << res.n_dofs
          << ", L2: " << std::scientific << std::setprecision(4) << res.l2_error;
    if (i > 0)
      pcout << " (EOC " << std::fixed << std::setprecision(2) << res.l2_eoc << ")";
    pcout << std::endl;
  }

  if (is_root)
  {
    std::cout << "\n========================================================================================\n"
              << "              CONVERGENCE TABLE: " << solver_title << "\n"
              << "========================================================================================\n";
    std::cout << std::left
              << std::setw(12) << "Refinement"
              << std::setw(14) << "h"
              << std::setw(12) << "DOFs"
              << std::setw(16) << "L2 Error"
              << std::setw(12) << "L2 EOC"
              << std::setw(16) << "H1 Error"
              << std::setw(12) << "H1 EOC"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i)
    {
      const auto &r = results[i];
      std::cout << std::left
                << std::setw(12) << r.refinement
                << std::scientific << std::setprecision(4)
                << std::setw(14) << r.h
                << std::defaultfloat
                << std::setw(12) << r.n_dofs
                << std::scientific << std::setprecision(6)
                << std::setw(16) << r.l2_error;
      if (i == 0)
        std::cout << std::setw(12) << "    -     ";
      else
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.l2_eoc;

      std::cout << std::scientific << std::setprecision(6)
                << std::setw(16) << r.h1_error;
      if (i == 0)
        std::cout << std::setw(12) << "    -     ";
      else
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.h1_eoc;

      std::cout << "\n";
    }
    std::cout << std::string(88, '=') << "\n\n";

    const std::string csv_name = "convergence_" + solver_tag + ".csv";
    const std::string csv_path = get_output_path(csv_name);
    std::ofstream csv(csv_path);
    csv << std::setprecision(14);
    csv << "refinement,h,n_dofs,l2_error,l2_eoc,h1_error,h1_eoc\n";
    for (const auto &r : results)
    {
      csv << r.refinement << ','
          << r.h << ','
          << r.n_dofs << ','
          << r.l2_error << ','
          << r.l2_eoc << ','
          << r.h1_error << ','
          << r.h1_eoc << '\n';
    }
    std::cout << "  Convergence results saved to " << csv_path << "\n\n";
  }
}

template <int dim>
void run_convergence(const ProgramOptions &opts)
{
  static_assert(dim == 2, "Convergence study is configured for dim=2.");

  std::vector<unsigned int> levels;
  if (opts.user_refine)
  {
    for (unsigned int l = 2; l <= opts.refine; ++l)
      levels.push_back(l);
  }
  else
  {
    // Canonical MMS levels:
    // Option 2 (strict spatial scaling dt ∝ h^2.5) evaluates {2, 3, 4}
    // where spatial error dominates before floating-point roundoff saturation.
    // Option 1 (CFL scaling dt = CFL * h) safely includes {2, 3, 4, 5}.
    if (opts.use_spatial_scaling)
      levels = {2, 3, 4};
    else
      levels = {2, 3, 4, 5};
  }

  const double final_time = (opts.final_time != 45.0) ? opts.final_time : 1.0;

  std::vector<std::string> solvers_to_run;
  if (opts.solver == "all")
    solvers_to_run = {"cg", "dg", "theta"};
  else
    solvers_to_run = {opts.solver};

  for (const auto &s : solvers_to_run)
  {
    run_convergence_for_solver<dim>(
        s, levels, final_time, opts.use_spatial_scaling, opts.gamma);
  }
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
        // the wave packet would exit the domain at T ~ 23).
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