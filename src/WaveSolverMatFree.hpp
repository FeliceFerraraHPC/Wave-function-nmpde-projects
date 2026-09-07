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
                 double                         time_step,
                 double                         gamma = 0.0);

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
  const double                               time_step_;
  const double                               gamma_;
  const VectorizedArray<double>              delta_t_sqr_;
  // Lumped inverse *effective* mass: (1/(1 + 0.5*dt*gamma)) / M_lumped[i].
  // Reduces to the plain lumped inverse mass when gamma_ == 0.
  LinearAlgebra::distributed::Vector<double> inv_effective_mass_matrix_;
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

  /**
   * @param cfl_number  CFL-type factor scaling dt relative to h/degree^2.
   * @param output_skip How many steps between VTU snapshots.
   * @param gamma       Damping coefficient (gamma >= 0) in
   *                    u_tt - Delta u + gamma*u_t = 0. Default 0 => undamped.
   */
  explicit WaveSolverMatFree(double cfl_number        = 0.1 / fe_degree,
                             unsigned int output_skip = 100,
                             double gamma             = 0.0);

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

  /**
   * Energy at the most recently completed step. Since leapfrog only stores
   * displacement snapshots, velocity is estimated via central difference:
   *   v^n ~ (u^{n+1} - u^{n-1}) / (2*dt)
   * using solution_ (u^{n+1}), old_old_solution_ (u^{n-1}), and the
   * gradient of old_solution_ (u^n) for the potential energy. This mirrors
   * exactly the convention used in the original WaveOperation::compute_energy,
   * extended with the dissipation rate D = gamma * v^2 = 2*gamma*E_kin.
   */
  EnergyData
  compute_energy() const override;

  const std::vector<EnergyData> &
  get_energy_history() const override
  { return energy_history_; }

  void
  export_energy_to_csv(const std::string &filename) const override
  { write_energy_history_csv(energy_history_, filename); }

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
  const double       gamma_; // damping coefficient in u_tt - Delta u + gamma*u_t = 0
  double             time_      = 0.0;
  double             time_step_ = 1.0;

  std::vector<EnergyData> energy_history_;
  mutable double          initial_total_energy_ = -1.0;
};

#endif // WAVE_SOLVER_MATFREE_HPP
