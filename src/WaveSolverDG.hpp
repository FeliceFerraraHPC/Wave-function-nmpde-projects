#ifndef WAVE_SOLVER_DG_HPP
#define WAVE_SOLVER_DG_HPP

#include "WaveSolverBase.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
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
// Matrix-free leap-frog operator — Discontinuous Galerkin (SIPG)
// ---------------------------------------------------------------------------
template <int dim, int fe_degree = 4>
class WaveOperationDG
{
public:
  WaveOperationDG(const MatrixFree<dim, double> &data_in,
                  double                         time_step,
                  double                         cell_diameter = 1.0,
                  double                         gamma         = 0.0);

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

  void
  local_apply_face(
    const MatrixFree<dim, double>                                   &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int>                     &face_range) const;

  void
  local_apply_boundary_face(
    const MatrixFree<dim, double>                                   &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int>                     &face_range) const;

  const MatrixFree<dim, double>             &data_;
  const double                               time_step_;
  const double                               gamma_;
  const VectorizedArray<double>              delta_t_sqr_;
  const double                               h_inv_;  ///< 1/h for SIPG penalty
  // Lumped inverse *effective* mass, see WaveOperationCG for details.
  LinearAlgebra::distributed::Vector<double> inv_effective_mass_matrix_;
};

// ---------------------------------------------------------------------------
// WaveSolverDG — wraps WaveOperationDG in the WaveSolverBase interface
// ---------------------------------------------------------------------------
template <int dim>
class WaveSolverDG : public WaveSolverBase<dim>
{
public:
  static constexpr unsigned int fe_degree = 4;

  /**
   * @param cfl_number  CFL-type factor scaling dt relative to h/degree^2.
   * @param output_skip How many steps between VTU snapshots.
   * @param gamma       Damping coefficient (gamma >= 0) in
   *                    u_tt - Delta u + gamma*u_t = 0. Default 0 => undamped.
   */
  explicit WaveSolverDG(double       cfl_number  = 0.05 / (fe_degree * fe_degree),
                        unsigned int output_skip = 500,
                        double       gamma       = 0.0);

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
   * Energy at the most recently completed step, computed from the volume
   * terms only (0.5*v^2 kinetic + 0.5*|grad u|^2 potential), the same way
   * as WaveSolverMatFree::compute_energy(), plus the dissipation rate
   * D = gamma * v^2 = 2*gamma*E_kin. NOTE: for the SIPG DG bilinear
   * form used here, the fully consistent discrete "energy" would also
   * include the interior-face jump penalty term 0.5*sigma*integral of
   * [u]^2 ds, which this volume-only version omits. For a solution that is
   * well-resolved relative to the mesh, that term is small and this still
   * gives a useful conservation/dissipation check; if you need an exact
   * SIPG energy balance, the jump term should be added by looping over
   * matrix_free_data_'s interior faces the same way local_apply_face() does.
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
  { return "Matrix-Free DG/SIPG (MPI+TBB+SIMD)"; }

private:
  void output_results(unsigned int timestep_number);

  ConditionalOStream pcout_;

  const Triangulation<dim> *tria_ptr_ = nullptr;

  Triangulation<dim>              dummy_tria_; ///< Placeholder before setup().
  const FE_DGQArbitraryNodes<dim> fe_;
  DoFHandler<dim>                 dof_handler_;
  const MappingQ1<dim>            mapping_;

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

#endif // WAVE_SOLVER_DG_HPP
