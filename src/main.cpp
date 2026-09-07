/**
 * @file main.cpp
 * @brief Unified benchmark driver for three wave equation solvers.
 *
 * Usage:
 *   ./WaveBenchmark [options]
 *
 * Options:
 *   --mode     [bench|convergence|both]   Default: bench
 *   --dim      [2|3]                      Default: 2
 *   --refine   N                          Global refinement level. Default: 6
 *   --time     T                          Final simulation time. Default: 1.0
 *   --solver   [all|theta|cg|dg]          Solvers to run. Default: all
 *   --output                              Enable VTU output (disabled by default)
 *
 * In --mode bench:
 *   Runs each selected solver on the SAME globally-refined triangulation and
 *   prints a timing + DoF table to stdout.
 *
 * In --mode convergence:
 *   Runs the theta-scheme manufactured-solution convergence study on a
 *   sequence of meshes and prints convergence rates.
 *
 * In --mode both:
 *   Runs the benchmark first, then the convergence study.
 */

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#ifdef DEAL_II_WITH_P4EST
#  include <deal.II/distributed/tria.h>
#endif

#include <deal.II/base/convergence_table.h>

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

using namespace dealii;

// ============================================================================
// Parse simple command-line arguments
// ============================================================================

struct ProgramOptions
{
  std::string mode             = "bench";   // bench | convergence | both
  int         dim              = 2;         // spatial dimension (2 or 3)
  unsigned int refine          = 6;         // global refinement levels
  double       final_time      = 1.0;       // final simulation time
  std::string  solver          = "all";     // all | theta | cg | dg
  bool         write_output    = false;     // write VTU files?
  std::string  levels_str      = "5-7";     // e.g. 5-7
  unsigned int output_frequency= 100;       // how often to write output
};

void declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("mode", "bench", Patterns::Selection("bench|convergence|both"));
  prm.declare_entry("dim", "2", Patterns::Integer(2, 3));
  prm.declare_entry("refine", "6", Patterns::Integer(1));
  prm.declare_entry("final_time", "1.0", Patterns::Double(0.0));
  prm.declare_entry("solver", "all", Patterns::Selection("all|theta|cg|dg"));
  prm.declare_entry("write_output", "false", Patterns::Bool());
  prm.declare_entry("levels", "5-7", Patterns::Anything());
  prm.declare_entry("output_frequency", "100", Patterns::Integer(1));
}

ProgramOptions
parse_args(int argc, char **argv)
{
  ProgramOptions opts;
  ParameterHandler prm;
  declare_parameters(prm);
  
  if (argc > 1) {
    prm.parse_input(argv[1]);
  }
  
  opts.mode = prm.get("mode");
  opts.dim = prm.get_integer("dim");
  opts.refine = prm.get_integer("refine");
  opts.final_time = prm.get_double("final_time");
  opts.solver = prm.get("solver");
  opts.write_output = prm.get_bool("write_output");
  opts.levels_str = prm.get("levels");
  opts.output_frequency = prm.get_integer("output_frequency");
  
  return opts;
}

// ============================================================================
// Benchmark mode: run all selected solvers on the same mesh
// ============================================================================

template <int dim>
void
run_benchmark(const ProgramOptions &opts)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << "========================================\n"
        << "  Wave Equation Benchmark  (dim=" << dim << ")\n"
        << "========================================\n"
        << "  Refinement levels : " << opts.refine    << "\n"
        << "  Final time        : " << opts.final_time << "\n"
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
  const InitialVelocity<dim>     v0; // zero velocity

  // --- Collect solvers to run ---
  using SolverPtr = std::unique_ptr<WaveSolverBase<dim>>;
  std::vector<SolverPtr> solvers;

  if (opts.solver == "all" || opts.solver == "theta")
    solvers.emplace_back(std::make_unique<WaveSolverTheta<dim>>());

  if (opts.solver == "all" || opts.solver == "cg")
    solvers.emplace_back(std::make_unique<WaveSolverMatFree<dim>>());

  if (opts.solver == "all" || opts.solver == "dg")
    solvers.emplace_back(std::make_unique<WaveSolverDG<dim>>());

  // Header for results table
  pcout << std::left
        << std::setw(36) << "Solver"
        << std::setw(12) << "DoFs"
        << std::setw(14) << "Steps"
        << std::setw(18) << "Compute time (s)"
        << std::setw(18) << "Avg time/step (s)"
        << "\n"
        << std::string(98, '-') << "\n";

  // --- Run each solver ---
  for (auto &solver : solvers)
    {
      solver->setup(tria);
      solver->set_initial_conditions(u0, v0);

      const double wtime = solver->run(opts.final_time, opts.write_output, opts.output_frequency);

      const unsigned int n_steps =
        static_cast<unsigned int>(opts.final_time / solver->time_step_size());

      pcout << std::left
            << std::setw(36) << solver->name()
            << std::setw(12) << solver->n_dofs()
            << std::setw(14) << n_steps
            << std::setw(18) << std::fixed << std::setprecision(4) << wtime
            << std::setw(18) << (n_steps > 0 ? wtime / n_steps : 0.0)
            << "\n";
    }

  pcout << std::string(98, '-') << "\n" << std::endl;
}

// ============================================================================
// Convergence mode: theta-scheme manufactured-solution test
// ============================================================================

template <int dim>
void
run_convergence(const ProgramOptions &opts)
{
  static_assert(dim == 2, "Convergence study only implemented for dim=2.");

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  std::vector<unsigned int> levels;
  {
    std::string s = opts.levels_str;
    size_t dash = s.find('-');
    if (dash != std::string::npos) {
      unsigned int min_l = std::stoul(s.substr(0, dash));
      unsigned int max_l = std::stoul(s.substr(dash+1));
      for (unsigned int l = min_l; l <= max_l; ++l) levels.push_back(l);
    } else {
      levels.push_back(std::stoul(s));
    }
  }

  pcout << "========================================\n"
        << "  Convergence Study (dim=" << dim << ")\n"
        << "========================================\n"
        << "  Levels            : " << opts.levels_str << "\n"
        << "  Final time        : " << opts.final_time << "\n"
        << std::endl;

  std::vector<std::string> solvers_to_run;
  if (opts.solver == "all" || opts.solver == "theta") solvers_to_run.push_back("theta");
  if (opts.solver == "all" || opts.solver == "cg")    solvers_to_run.push_back("cg");
  if (opts.solver == "all" || opts.solver == "dg")    solvers_to_run.push_back("dg");

  for (const auto &solver_name : solvers_to_run)
    {
      pcout << "--- Solver: " << solver_name << " ---\n";
      ConvergenceTable table;
      std::string csv_name = "convergence_" + solver_name + ".csv";
      std::ofstream csv;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          csv.open(csv_name);
          csv << "h,eL2(u),eH1(u)\n";
        }

      for (const unsigned int N_el : levels)
        {
          Triangulation<dim> mesh;
          GridGenerator::hyper_cube(mesh, 0.0, 1.0);
          mesh.refine_global(N_el);

          const double h  = 1.0 / std::pow(2.0, N_el);
          
          std::unique_ptr<WaveSolverBase<dim>> solver;
          if (solver_name == "theta")
            solver = std::make_unique<WaveSolverTheta<dim>>(); // default fe_degree
          else if (solver_name == "cg")
            solver = std::make_unique<WaveSolverMatFree<dim>>();
          else if (solver_name == "dg")
            solver = std::make_unique<WaveSolverDG<dim>>();

          solver->setup(mesh);

          Functions::ZeroFunction<dim> zero;
          solver->set_initial_conditions(zero, zero);

          ManufacturedRHS<dim> rhs_func;
          solver->set_forcing_function(&rhs_func);

          solver->run(opts.final_time, /*write_output=*/false, opts.output_frequency);

          ManufacturedSolutionU<dim> sol_u_T(opts.final_time);

          const double eL2u = solver->compute_error(VectorTools::L2_norm, sol_u_T);
          const double eH1u = solver->compute_error(VectorTools::H1_norm, sol_u_T);

          table.add_value("h",     h);
          table.add_value("L2(u)", eL2u);
          table.add_value("H1(u)", eH1u);

          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            csv << h << "," << eL2u << "," << eH1u << "\n";
        }

      table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
      table.set_scientific("L2(u)", true);
      table.set_scientific("H1(u)", true);
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        table.write_text(std::cout);
      pcout << "\n";
    }
}

// ============================================================================
// main()
// ============================================================================

int
main(int argc, char **argv)
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
        }
      else if (opts.dim == 3)
        {
          if (opts.mode == "bench" || opts.mode == "both")
            run_benchmark<3>(opts);
          // Convergence study in 3D not implemented (manufactured solution is 2D only)
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
