#ifndef WAVE_SOLVER_MATFREE_HPP
#define WAVE_SOLVER_MATFREE_HPP

#include "WaveSolverBase.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

// ---------------------------------------------------------------------------
// Matrix-free leap-frog operator (continuous Galerkin)
// Implements: dst = M^{-1} * (2*M - dt^2*A) * src[0] - src[1]
// ---------------------------------------------------------------------------
template <int dim, int fe_degree = 4>
class WaveOperationCG
{
public:
  WaveOperationCG(const MatrixFree<dim, double> &data_in,
                  double time_step,
                  double gamma = 0.0);

  void
  apply(LinearAlgebra::distributed::Vector<double> &dst,
        const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const;

private:
  void
  local_apply(
      const MatrixFree<dim, double> &data,
      LinearAlgebra::distributed::Vector<double> &dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  const MatrixFree<dim, double> &data_;
  const double time_step_;
  const double gamma_;
  const VectorizedArray<double> delta_t_sqr_;

  // Lumped inverse *effective* mass: (1/(1 + 0.5*dt*gamma)) / M_lumped[i].
  // Reduces to the plain lumped inverse mass when gamma_ == 0.
  LinearAlgebra::distributed::Vector<double> inv_effective_mass_matrix_;
};

// ---------------------------------------------------------------------------
// WaveSolverMatFree -- wraps WaveOperationCG in the WaveSolverBase interface
// ---------------------------------------------------------------------------
template <int dim>
class WaveSolverMatFree : public WaveSolverBase<dim>
{
public:
  // Compile-time polynomial degree for the CG solver.
  static constexpr unsigned int fe_degree = 4;

  /**
   * @param cfl_number  CFL-type factor scaling dt relative to h/degree^2.
   * @param output_skip How many steps between VTU snapshots.
   * @param gamma       Damping coefficient (gamma >= 0) in
   *                    u_tt - Delta u + gamma*u_t = 0. Default 0 => undamped.
   */
  explicit WaveSolverMatFree(double cfl_number = 0.1 / fe_degree,
                             unsigned int output_skip = 100,
                             double gamma = 0.0);

  // WaveSolverBase interface -------------------------------------------------
  void
  setup(const Triangulation<dim> &tria) override;

  void
  set_initial_conditions(const Function<dim> &u0,
                         const Function<dim> &v0,
                         const Function<dim> *u_prev = nullptr) override;

  double
  run(double T, bool write_output = false) override;

  double
  compute_error(VectorTools::NormType norm_type,
                const Function<dim> &exact_solution) const override;

  std::pair<Point<dim>, double>
  find_peak(const Point<dim> &center,
            double x_span = 1.5,
            unsigned int n_pts = 300) const override;

  /**
   * Energy at the most recently completed step.
   *
   * Two diagnostics are returned in the same EnergyData snapshot:
   *
   * 1. NATURAL (central-difference) energy -- fields kinetic/potential/total_energy.
   *    Kinetic: v^n ~ (u^{n+1} - u^{n-1}) / (2dt), evaluated via consistent
   *    Gauss quadrature.  Potential: 0.5*||grad u^n||^2.  This is what the
   *    literature calls the "collocated" leapfrog energy; it drifts by O(dt^2)
   *    per step because kinetic and potential live at different half-integer levels.
   *
   * 2. STAGGERED (half-step) energy -- fields stag_kinetic/stag_potential/stag_total.
   *    Kinetic: 0.5 * ||(u^{n+1} - u^n)/dt||^2_M where M is the lumped
   *    (Gauss-Lobatto diagonal) mass -- the numerically exact discrete invariant.
   *    Potential: 0.5 * a_h(u^n, u^{n+1}) -- volume grad-grad cross-term.
   *    This quantity is exactly conserved by the leapfrog integrator (zero drift
   *    up to floating-point rounding for gamma=0, no forcing).
   *
   * The gap |E_nat - E_stag| / E_stag is itself an observable: it measures the
   * O(dt^2) approximation error in the natural diagnostic and should scale as dt^2.
   */
  EnergyData
  compute_energy() const override;

  const std::vector<EnergyData> &
  get_energy_history() const override
  {
    return energy_history_;
  }

  void
  export_energy_to_csv(const std::string &filename) const override
  {
    write_energy_history_csv(energy_history_, filename);
  }

  double
  current_time() const override
  {
    return time_;
  }

  double
  time_step_size() const override
  {
    return time_step_;
  }

  void
  set_time_step(double dt) override
  {
    time_step_ = dt;
    user_time_step_ = true;
  }

  unsigned int
  n_dofs() const override;

  std::string
  name() const override
  {
    return "Matrix-Free CG (MPI+TBB+SIMD)";
  }

private:
  void
  output_results(unsigned int timestep_number);

  ConditionalOStream pcout_;
  const Triangulation<dim> *tria_ptr_ = nullptr;
  Triangulation<dim> dummy_tria_; ///< Placeholder before setup().
  const FE_Q<dim> fe_;
  DoFHandler<dim> dof_handler_;
  const MappingQ1<dim> mapping_;
  AffineConstraints<double> constraints_;
  IndexSet locally_relevant_dofs_;
  MatrixFree<dim, double> matrix_free_data_;
  LinearAlgebra::distributed::Vector<double> solution_;
  LinearAlgebra::distributed::Vector<double> old_solution_;
  LinearAlgebra::distributed::Vector<double> old_old_solution_;

  // Lumped (Gauss-Lobatto diagonal) mass vector.  Built once in setup() and
  // used in compute_energy() to form the exact discrete kinetic inner product
  // for the staggered energy: E_kin_stag = 0.5 * diff^T M_lumped diff.
  LinearAlgebra::distributed::Vector<double> lumped_mass_;

  const double cfl_number_;
  const unsigned int output_timestep_skip_;
  const double gamma_; // damping coefficient in u_tt - Delta u + gamma*u_t = 0
  double time_ = 0.0;
  double time_step_ = 1.0;
  bool user_time_step_ = false;

  std::vector<EnergyData> energy_history_;
  mutable double initial_total_energy_ = -1.0;
  mutable double initial_stag_total_energy_ = -1.0; // lazy-cached stag E_tot(0)
};

#endif // WAVE_SOLVER_MATFREE_HPP