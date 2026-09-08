#ifndef WAVE_SOLVER_THETA_HPP
#define WAVE_SOLVER_THETA_HPP

#include "WaveSolverBase.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

/**
 * @brief Theta-scheme (Crank-Nicolson by default) wave equation solver.
 *
 * Solves:
 *   u_tt - Delta u = f(x,t)
 * using the second-order theta-scheme split into two first-order steps:
 *   (M + theta^2 dt^2 A) u^{n+1} = rhs_u
 *   M v^{n+1} = rhs_v
 *
 * Parallelism: MPI-distributed assembly and solve via TrilinosWrappers.
 * The DoFHandler uses locally owned / locally relevant IndexSets.
 *
 * @tparam dim  Spatial dimension.
 */
template <int dim>
class WaveSolverTheta : public WaveSolverBase<dim>
{
public:
  /**
   * @param fe_degree  Polynomial degree of the FE_Q finite element.
   * @param theta      Time-stepping parameter. 0.5 = Crank-Nicolson (default).
   * @param gamma      Damping coefficient (gamma >= 0) in
   *                   u_tt - Delta u + gamma*u_t = f. Default 0 => undamped.
   */
  explicit WaveSolverTheta(unsigned int fe_degree = 4,
                           double theta = 0.5,
                           double gamma = 0.0);

  // WaveSolverBase interface --------------------------------------------------
  void
  setup(const Triangulation<dim> &tria) override;

  void
  set_initial_conditions(const Function<dim> &u0,
                         const Function<dim> &v0) override;

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
   * Energy of the current state.
   *
   * Two diagnostics are returned:
   *
   * 1. EXACT (natural) energy -- uses the explicitly-tracked velocity v^n:
   *      E_kin = 0.5 * v^T M v,   E_pot = 0.5 * u^T A u
   *    This is O(1) accurate (no dt error) because the theta-scheme maintains
   *    v^n as an explicit variable.
   *
   * 2. STAGGERED energy -- mimics the leapfrog staggered observable so that
   *    all three solvers expose a comparable metric:
   *      v_{n-1/2} = (u^n - u^{n-1}) / dt   (backward difference)
   *      E_kin_stag = 0.5 * v_{n-1/2}^T M v_{n-1/2}
   *      E_pot_stag = 0.5 * (u^{n-1})^T A u^n  =  0.5 * a(u^{n-1}, u^n)
   *
   * NOTE: compute_energy() must be called BEFORE old_solution_u_ is updated
   * (i.e. before `old_solution_u_ = solution_u_`) so that old_solution_u_
   * still holds u^{n-1} when this function runs.  run() guarantees this order.
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
    return "Theta-scheme (Trilinos MPI)";
  }

  // Optional: run the manufactured-solution convergence study (serial helper).
  void
  run_convergence_study(const std::vector<unsigned int> &refinement_levels,
                        double final_time,
                        unsigned int fe_degree,
                        double theta = 0.5);

private:
  void
  setup_system();

  void
  assemble_matrices();

  void
  solve_u();

  void
  solve_v();

  void
  output_results(unsigned int step_number) const;

  // Forcing term and boundary values -- set by the caller via function pointers
  // or use default zero functions.
  const Function<dim> *forcing_function_ptr_ = nullptr;
  const Function<dim> *boundary_u_ptr_ = nullptr;
  const Function<dim> *boundary_v_ptr_ = nullptr;

  ConditionalOStream pcout_;
  const Triangulation<dim> *tria_ptr_ = nullptr; ///< Non-owning pointer to external mesh.
  const unsigned int fe_degree_;
  const double theta_;
  const double gamma_; // damping coefficient in u_tt - Delta u + gamma*u_t = f
  FE_Q<dim> fe_;
  Triangulation<dim> dummy_tria_; ///< Placeholder before setup(); declared after fe_ to match init-list order.
  DoFHandler<dim> dof_handler_;
  IndexSet locally_owned_dofs_;
  IndexSet locally_relevant_dofs_;
  AffineConstraints<double> constraints_;
  TrilinosWrappers::SparseMatrix mass_matrix_;
  TrilinosWrappers::SparseMatrix laplace_matrix_;
  TrilinosWrappers::SparseMatrix matrix_u_;
  TrilinosWrappers::SparseMatrix matrix_v_;
  TrilinosWrappers::MPI::Vector solution_u_;
  TrilinosWrappers::MPI::Vector solution_v_;
  TrilinosWrappers::MPI::Vector old_solution_u_;
  TrilinosWrappers::MPI::Vector old_solution_v_;
  TrilinosWrappers::MPI::Vector system_rhs_;

  double time_step_ = 1.0 / 64.0;
  bool user_time_step_ = false;
  double time_ = 0.0;
  unsigned int timestep_number_ = 0;

  std::vector<EnergyData> energy_history_;
  mutable double initial_total_energy_ = -1.0;      // lazy-cached E_tot(0)
  mutable double initial_stag_total_energy_ = -1.0; // lazy-cached stag E_tot(0)
};

#endif // WAVE_SOLVER_THETA_HPP