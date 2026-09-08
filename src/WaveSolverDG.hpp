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
// Matrix-free leap-frog operator -- Discontinuous Galerkin (SIPG)
// ---------------------------------------------------------------------------
template <int dim, int fe_degree = 4>
class WaveOperationDG
{
public:
  WaveOperationDG(const MatrixFree<dim, double> &data_in,
                  double time_step,
                  double cell_diameter = 1.0,
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

  void
  local_apply_face(
      const MatrixFree<dim, double> &data,
      LinearAlgebra::distributed::Vector<double> &dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &face_range) const;

  void
  local_apply_boundary_face(
      const MatrixFree<dim, double> &data,
      LinearAlgebra::distributed::Vector<double> &dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &face_range) const;

  const MatrixFree<dim, double> &data_;
  const double time_step_;
  const double gamma_;
  const VectorizedArray<double> delta_t_sqr_;
  const double h_inv_; ///< 1/h for SIPG penalty

  // Lumped inverse *effective* mass, see WaveOperationCG for details.
  LinearAlgebra::distributed::Vector<double> inv_effective_mass_matrix_;
};

// ---------------------------------------------------------------------------
// WaveSolverDG -- wraps WaveOperationDG in the WaveSolverBase interface
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
  explicit WaveSolverDG(double cfl_number = 0.05 / (fe_degree * fe_degree),
                        unsigned int output_skip = 500,
                        double gamma = 0.0);

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
   *    Volume-only (grad u^n term); omits SIPG face penalties.  This is the
   *    same observable as the CG solver, enabling direct comparison.
   *
   * 2. STAGGERED (half-step) energy -- fields stag_kinetic/stag_potential/stag_total.
   *    Kinetic: 0.5 * ||(u^{n+1}-u^n)/dt||^2_M   (exact GL lumped-mass inner product)
   *    Potential: 0.5 * a_h(u^n, u^{n+1}) -- the FULL SIPG discrete bilinear form,
   *    including interior-face jump penalty and boundary Dirichlet penalty terms.
   *    This gives the true staggered discrete Hamiltonian for the SIPG leapfrog scheme.
   *
   * The gap |E_nat - E_stag| is informative: it quantifies the O(dt^2) drift
   * in the natural diagnostic AND the effect of omitting face terms in the
   * potential estimate.
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

  unsigned int
  n_dofs() const override;

  std::string
  name() const override
  {
    return "Matrix-Free DG/SIPG (MPI+TBB+SIMD)";
  }

private:
  void
  output_results(unsigned int timestep_number);

  ConditionalOStream pcout_;
  const Triangulation<dim> *tria_ptr_ = nullptr;
  Triangulation<dim> dummy_tria_; ///< Placeholder before setup().
  const FE_DGQArbitraryNodes<dim> fe_;
  DoFHandler<dim> dof_handler_;
  const MappingQ1<dim> mapping_;
  AffineConstraints<double> constraints_;
  IndexSet locally_relevant_dofs_;
  MatrixFree<dim, double> matrix_free_data_;
  LinearAlgebra::distributed::Vector<double> solution_;
  LinearAlgebra::distributed::Vector<double> old_solution_;
  LinearAlgebra::distributed::Vector<double> old_old_solution_;

  // Lumped (GL diagonal) mass: used for the exact discrete kinetic energy.
  LinearAlgebra::distributed::Vector<double> lumped_mass_;

  const double cfl_number_;
  const unsigned int output_timestep_skip_;
  const double gamma_; // damping coefficient in u_tt - Delta u + gamma*u_t = 0
  double time_ = 0.0;
  double time_step_ = 1.0;

  std::vector<EnergyData> energy_history_;
  mutable double initial_total_energy_ = -1.0;
  mutable double initial_stag_total_energy_ = -1.0;
};

#endif // WAVE_SOLVER_DG_HPP