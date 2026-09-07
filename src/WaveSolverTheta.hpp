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

/**
 * @brief Theta-scheme (Crank-Nicolson by default) wave equation solver.
 *
 * Solves:
 *   u_tt - Delta u = f(x,t)
 *
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
   */
  explicit WaveSolverTheta(unsigned int fe_degree = 4,
                           double       theta     = 0.5);

  // WaveSolverBase interface --------------------------------------------------

  void
  setup(const Triangulation<dim> &tria) override;

  void
  set_initial_conditions(const Function<dim> &u0,
                         const Function<dim> &v0) override;

  void
  set_forcing_function(const Function<dim> *f) override;

  double run(double T, bool write_output = false, unsigned int output_frequency = 100) override;

  double
  compute_error(VectorTools::NormType norm_type,
                const Function<dim>  &exact_solution) const override;

  double       current_time()   const override { return time_; }
  double       time_step_size() const override { return time_step_; }
  unsigned int n_dofs()         const override;
  std::string  name()           const override { return "Theta-scheme (Trilinos MPI)"; }

private:
  void setup_system();
  void assemble_matrices();
  void solve_u();
  void solve_v();
  void output_results(unsigned int step_number) const;

  // Forcing term and boundary values — set by the caller via function pointers
  // or use default zero functions.
  const Function<dim> *forcing_function_ptr_  = nullptr;
  const Function<dim> *boundary_u_ptr_        = nullptr;
  const Function<dim> *boundary_v_ptr_        = nullptr;

  ConditionalOStream pcout_;

  const Triangulation<dim> *tria_ptr_ = nullptr; ///< Non-owning pointer to external mesh.

  const unsigned int fe_degree_;
  const double       theta_;

  FE_Q<dim>          fe_;
  Triangulation<dim> dummy_tria_; ///< Placeholder before setup(); declared after fe_ to match init-list order.
  DoFHandler<dim>    dof_handler_;

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

  double       time_step_ = 1.0 / 64.0;
  double       time_      = 0.0;
  unsigned int timestep_number_ = 0;
};

#endif // WAVE_SOLVER_THETA_HPP
