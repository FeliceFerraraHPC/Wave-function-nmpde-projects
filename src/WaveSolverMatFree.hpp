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
  WaveOperationCG(const MatrixFree<dim, double> &data_in, double time_step);

  void
  apply(LinearAlgebra::distributed::Vector<double>                      &dst,
        const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const;

private:
  void
  local_apply(
    const MatrixFree<dim, double>                                   &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int>                     &cell_range) const;

  const MatrixFree<dim, double>             &data_;
  const VectorizedArray<double>              delta_t_sqr_;
  LinearAlgebra::distributed::Vector<double> inv_mass_matrix_;
};

// ---------------------------------------------------------------------------
// WaveSolverMatFree — wraps WaveOperationCG in the WaveSolverBase interface
// ---------------------------------------------------------------------------
template <int dim>
class WaveSolverMatFree : public WaveSolverBase<dim>
{
public:
  // Compile-time polynomial degree for the CG solver.
  static constexpr unsigned int fe_degree = 4;

  explicit WaveSolverMatFree(double cfl_number        = 0.1 / fe_degree,
                             unsigned int output_skip = 100);

  // WaveSolverBase interface -------------------------------------------------

  void
  setup(const Triangulation<dim> &tria) override;

  void
  set_initial_conditions(const Function<dim> &u0,
                         const Function<dim> &v0) override;

  double
  run(double T, bool write_output = false) override;

  double
  compute_error(VectorTools::NormType norm_type,
                const Function<dim>  &exact_solution) const override;

  double       current_time()   const override { return time_; }
  double       time_step_size() const override { return time_step_; }
  unsigned int n_dofs()         const override;
  std::string  name()           const override
  { return "Matrix-Free CG (MPI+TBB+SIMD)"; }

private:
  void output_results(unsigned int timestep_number);

  ConditionalOStream pcout_;

  const Triangulation<dim> *tria_ptr_ = nullptr;

  Triangulation<dim>   dummy_tria_; ///< Placeholder before setup().
  const FE_Q<dim>      fe_;
  DoFHandler<dim>      dof_handler_;
  const MappingQ1<dim> mapping_;

  AffineConstraints<double> constraints_;
  IndexSet                  locally_relevant_dofs_;

  MatrixFree<dim, double> matrix_free_data_;

  LinearAlgebra::distributed::Vector<double> solution_;
  LinearAlgebra::distributed::Vector<double> old_solution_;
  LinearAlgebra::distributed::Vector<double> old_old_solution_;

  const double       cfl_number_;
  const unsigned int output_timestep_skip_;
  double             time_      = 0.0;
  double             time_step_ = 1.0;
};

#endif // WAVE_SOLVER_MATFREE_HPP
