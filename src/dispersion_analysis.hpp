#ifndef DISPERSION_ANALYSIS_HPP
#define DISPERSION_ANALYSIS_HPP

/**
 * @file dispersion_analysis.hpp
 * @brief Numerical Dispersion Analysis for the wave-equation solver suite.
 *
 * Activated via --mode dispersion in the WaveBenchmark driver.
 *
 * Two sub-studies are performed:
 *
 *  1. p-degree parametric study (WaveSolverTheta, CG):
 *     Run the Gaussian sinusoidal wave packet for p = 1, 2, 4, 6 at a
 *     matched total DOF count.  Compares phase lag vs polynomial order.
 *
 *  2. CG vs DG comparison (both p = 4):
 *     WaveSolverMatFree (CG) vs WaveSolverDG (DG/SIPG) on the same mesh.
 *     High-order DG is expected to show significantly lower phase dispersion.
 *
 * Phase shift is found by minimising
 *   ||u_num(T) - u_exact(T - s)||_{L2}   over  s in [-T/3, T/3]
 * via ternary search (40 iterations, ~10 significant digits).
 *
 * Initial condition:
 *   u0(x,y) = exp(-(x-x0)^2/(2sigma^2)) * cos(k*(x-x0))
 *   v0(x,y) = c * exp(-(x-x0)^2/(2sigma^2))
 *              * [(x-x0)/sigma^2 * cos(k*(x-x0)) + k*sin(k*(x-x0))]
 *
 * Defaults: k = 2pi, sigma = 1.0, x0 = -8.0, T = 18.0, domain [-15,15]^2.
 */

#include "WaveFunctions.hpp"
#include "WaveSolverBase.hpp"
#include "WaveSolverDG.hpp"
#include "WaveSolverMatFree.hpp"
#include "WaveSolverTheta.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#ifdef DEAL_II_WITH_P4EST
#include <deal.II/distributed/tria.h>
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

// ============================================================================
// write_dispersion_csv
// ============================================================================
inline void
write_dispersion_csv(const std::vector<DispersionData> &results,
                     const std::string &filename)
{
  std::ofstream out(filename);
  out << std::setprecision(14);
  out << "solver,fe_degree,n_dofs,wavenumber,final_time,"
         "l2_raw,l2_aligned,phase_shift_dt,phase_error_rad,phase_lag_rel,"
         "peak_x_exact,peak_x_num,peak_amp\n";
  for (const auto &d : results)
    out << d.solver_name << ','
        << d.fe_degree << ','
        << d.n_dofs << ','
        << d.wavenumber << ','
        << d.final_time << ','
        << d.l2_error << ','
        << d.l2_aligned << ','
        << d.phase_shift << ','
        << d.phase_error_rad << ','
        << d.phase_lag_rel << ','
        << d.peak_x_exact << ','
        << d.peak_x_num << ','
        << d.peak_amp << '\n';
}

// ============================================================================
// print_dispersion_table
// ============================================================================
inline void
print_dispersion_table(const std::vector<DispersionData> &results,
                       std::ostream &os = std::cout)
{
  const int W1 = 30, W2 = 4, W3 = 8, W4 = 11, W5 = 10, W6 = 11, W7 = 14, W8 = 13, W9 = 11;
  const int total = W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9;

  os << '\n'
     << std::string(total, '=') << '\n'
     << "  Numerical Dispersion Analysis Results\n"
     << std::string(total, '-') << '\n'
     << std::left
     << std::setw(W1) << "Solver"
     << std::setw(W2) << "p"
     << std::setw(W3) << "DoFs"
     << std::setw(W4) << "Peak x"
     << std::setw(W5) << "Peak Amp"
     << std::setw(W6) << "Phase Dt"
     << std::setw(W7) << "Phase err(rad)"
     << std::setw(W8) << "Rel lag"
     << std::setw(W9) << "L2 (align)"
     << '\n'
     << std::string(total, '-') << '\n';

  for (const auto &d : results)
    os << std::left
       << std::setw(W1) << d.solver_name
       << std::setw(W2) << d.fe_degree
       << std::setw(W3) << d.n_dofs
       << std::setw(W4) << std::fixed << std::setprecision(4) << d.peak_x_num
       << std::setw(W5) << std::fixed << std::setprecision(4) << d.peak_amp
       << std::setw(W6) << std::fixed << std::setprecision(5) << d.phase_shift
       << std::setw(W7) << std::scientific << std::setprecision(3) << d.phase_error_rad
       << std::setw(W8) << std::scientific << std::setprecision(3) << d.phase_lag_rel
       << std::setw(W9) << std::scientific << std::setprecision(3) << d.l2_aligned
       << '\n';

  os << std::string(total, '=') << "\n\n";
}

// ============================================================================
// measure_dispersion  — measure peak location, phase error, and L2 norms
// ============================================================================
template <int dim>
DispersionData
measure_dispersion(WaveSolverBase<dim> &solver,
                   GaussianSinusoidExact<dim> &u_exact,
                   const std::string &label,
                   const unsigned int p,
                   const double k,
                   const double c,
                   const double T,
                   const double x0 = -8.0,
                   const double L_y = 30.0)
{
  // 1. Analytical peak position (accounting for 2D waveguide mode phase speed)
  double c_eff = c;
  if constexpr (dim == 2)
    c_eff = std::sqrt(c * c + (M_PI * M_PI) / (k * k * L_y * L_y));
  const double x_exact_peak = x0 + c_eff * T;

  Point<dim> exact_center;
  exact_center[0] = x_exact_peak;
  // In dim==2, y = 0 is the center of the domain [-15, 15]

  // 2. Measure raw point-by-point L2 error at nominal final time T
  u_exact.set_time(T);
  const double l2_err = solver.compute_error(VectorTools::L2_norm, u_exact);

  // 3. Directly locate numerical wave peak (highest crest) along the centerline
  const auto [num_peak_pt, num_peak_val] = solver.find_peak(exact_center, /*x_span=*/2.5, /*n_pts=*/500);
  const double x_num_peak = num_peak_pt[0];

  // 4. Compute physical phase shift from peak displacement:
  //    Delta_x > 0 means x_num < x_exact (numerical wave is lagging behind)
  const double dx_shift = x_exact_peak - x_num_peak;
  const double dt_shift = dx_shift / c_eff;
  const double dphi     = k * dx_shift;   // Delta_phi = k * Delta_x  (radians)
  const double rel_lag  = dt_shift / T;   // Delta_t / T

  // 5. Measure aligned L2 error with peak shift compensated
  u_exact.set_time(T - dt_shift);
  const double l2_aligned = solver.compute_error(VectorTools::L2_norm, u_exact);
  u_exact.set_time(T); // restore exact solution time

  DispersionData d;
  d.solver_name     = label;
  d.fe_degree       = p;
  d.n_dofs          = solver.n_dofs();
  d.wavenumber      = k;
  d.final_time      = T;
  d.l2_error        = l2_err;
  d.l2_aligned      = l2_aligned;
  d.phase_shift     = dt_shift;
  d.phase_error_rad = dphi;
  d.phase_lag_rel   = rel_lag;
  d.peak_x_exact    = x_exact_peak;
  d.peak_x_num      = x_num_peak;
  d.peak_amp        = num_peak_val;

  return d;
}

// ============================================================================
// run_dispersion<dim>
// ============================================================================
template <int dim>
void run_dispersion(const double k = 2.0 * M_PI,
                    const double sigma = 1.0,
                    const double x0 = -8.0,
                    const double T_final = 18.0,
                    const unsigned int target_dofs = 16000,
                    const unsigned int refine_cmp = 5)
{
  static_assert(dim == 2,
                "Dispersion study is implemented for dim=2 only (plane-wave in x).");

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  constexpr double c = 1.0;
  constexpr double domain_min = -15.0;
  constexpr double domain_max = 15.0;

  pcout << "\n========================================\n"
        << "  Numerical Dispersion Analysis (dim=2)\n"
        << "========================================\n"
        << "  Wavenumber k  : " << k << "  (lambda = " << 2.0 * M_PI / k << ")\n"
        << "  Envelope sigma: " << sigma << "\n"
        << "  Initial center: x0 = " << x0 << "\n"
        << "  Final time    : T  = " << T_final << "\n"
        << "  Exact peak at : x  = " << x0 + c * T_final << "\n"
        << "  Target DoFs   : " << target_dofs << "\n"
        << std::endl;

  // Initial conditions and exact solution (shared by all sub-studies)
  const GaussianSinusoidIC<dim> u0_ic(k, x0, sigma);
  const GaussianSinusoidV0<dim> v0_ic(k, x0, sigma, c);
  GaussianSinusoidExact<dim> u_exact(k, x0, sigma, c);

  std::vector<DispersionData> results;

  // =========================================================================
  // PART 1 — p-degree parametric study  (WaveSolverTheta, theta=0.5 i.e. CN)
  //           p = 1, 2, 4, 6  at matched total DOF count.
  //
  // For FE_Q(p) on [a,b]^2 uniform mesh with 2^refine cells per direction:
  //   n_dofs = (p * 2^refine + 1)^2
  //   => 2^refine = (sqrt(target_dofs) - 1) / p
  // =========================================================================
  pcout << "------------------------------------------------------------\n"
        << "  Part 1: polynomial-degree study  (Theta CN, p = 1,2,4,6)\n"
        << "------------------------------------------------------------\n";

  const std::vector<unsigned int> p_values = {1, 2, 4, 6};

  for (const unsigned int p : p_values)
  {
    const double N_float = (std::sqrt(static_cast<double>(target_dofs)) - 1.0) / p;
    const unsigned int refine =
        std::max(2u, static_cast<unsigned int>(std::round(std::log2(std::max(1.0, N_float)))));

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
#else
    Triangulation<dim> tria;
#endif
    GridGenerator::hyper_cube(tria, domain_min, domain_max);
    tria.refine_global(refine);

    pcout << "  p=" << p
          << "  refine=" << refine
          << "  cells=" << tria.n_global_active_cells()
          << std::flush;

    WaveSolverTheta<dim> solver(p, /*theta=*/0.5, /*gamma=*/0.0);
    solver.setup(tria);
    solver.set_initial_conditions(u0_ic, v0_ic);
    // Use a small, matched time step across all p so that Crank-Nicolson
    // temporal dispersion is negligible (~0.009 rad) and does not swamp
    // the spatial dispersion of polynomial degree p.
    solver.set_time_step(0.005);
    solver.run(T_final, /*write_output=*/false);

    const DispersionData d =
        measure_dispersion<dim>(solver, u_exact, "Theta CN (CG)", p, k, c, T_final, x0);
    results.push_back(d);

    pcout << "  DoFs=" << d.n_dofs
          << "  peak_x=" << std::fixed << std::setprecision(4) << d.peak_x_num
          << "  amp=" << std::fixed << std::setprecision(4) << d.peak_amp
          << "  Dt=" << std::fixed << std::setprecision(5) << d.phase_shift
          << "  Dphi=" << std::scientific << std::setprecision(3) << d.phase_error_rad
          << " rad\n";
  }

  // =========================================================================
  // PART 2 — CG (WaveSolverMatFree p=4) vs DG (WaveSolverDG p=4)
  //           Same mesh (refine_cmp refinements).
  // =========================================================================
  pcout << "\n------------------------------------------------------------\n"
        << "  Part 2: CG vs DG comparison  (p=4, refine=" << refine_cmp << ")\n"
        << "------------------------------------------------------------\n";

  {
#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
#else
    Triangulation<dim> tria;
#endif
    GridGenerator::hyper_cube(tria, domain_min, domain_max);
    tria.refine_global(refine_cmp);

    pcout << "  cells=" << tria.n_global_active_cells() << "\n";

    // --- CG (matrix-free) ---
    {
      WaveSolverMatFree<dim> solver(
        0.1 / WaveSolverMatFree<dim>::fe_degree,
        /*output_skip=*/100,
        /*gamma=*/0.0);
      solver.setup(tria);
      solver.set_initial_conditions(u0_ic, v0_ic);

      pcout << "  [CG MatFree p=4] running ..." << std::flush;
      solver.run(T_final, /*write_output=*/false);

      const DispersionData d = measure_dispersion<dim>(
          solver, u_exact,
          solver.name(), WaveSolverMatFree<dim>::fe_degree,
          k, c, T_final, x0);
      results.push_back(d);

      pcout << " DoFs=" << d.n_dofs
            << "  peak_x=" << std::fixed << std::setprecision(4) << d.peak_x_num
            << "  amp=" << std::fixed << std::setprecision(4) << d.peak_amp
            << "  Dt=" << std::fixed << std::setprecision(5) << d.phase_shift
            << "  Dphi=" << std::scientific << std::setprecision(3) << d.phase_error_rad
            << " rad\n";
    }

    // --- DG (matrix-free, SIPG) ---
    {
      WaveSolverDG<dim> solver(
          0.05 / (WaveSolverDG<dim>::fe_degree * WaveSolverDG<dim>::fe_degree),
          /*output_skip=*/500,
          /*gamma=*/0.0);
      solver.setup(tria);
      solver.set_initial_conditions(u0_ic, v0_ic);

      pcout << "  [DG  MatFree p=4] running ..." << std::flush;
      solver.run(T_final, /*write_output=*/false);

      const DispersionData d = measure_dispersion<dim>(
          solver, u_exact,
          solver.name(), WaveSolverDG<dim>::fe_degree,
          k, c, T_final, x0);
      results.push_back(d);

      pcout << " DoFs=" << d.n_dofs
            << "  peak_x=" << std::fixed << std::setprecision(4) << d.peak_x_num
            << "  amp=" << std::fixed << std::setprecision(4) << d.peak_amp
            << "  Dt=" << std::fixed << std::setprecision(5) << d.phase_shift
            << "  Dphi=" << std::scientific << std::setprecision(3) << d.phase_error_rad
            << " rad\n";
    }
  }

  // =========================================================================
  // Output — summary table + CSV
  // =========================================================================
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    print_dispersion_table(results, std::cout);
    write_dispersion_csv(results, "dispersion_results.csv");
    std::cout << "  Results written to  dispersion_results.csv\n\n";
  }
}

#endif // DISPERSION_ANALYSIS_HPP
