#ifndef WAVE_EQUATION_HPP
#define WAVE_EQUATION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace dealii;

// ============================================================================
// Initial Conditions
// ============================================================================

/**
 * Initial displacement u_0(x): a Gaussian wave packet centred at the origin.
 */
template <int dim>
class InitialDisplacement : public Function<dim>
{
public:
  InitialDisplacement(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int = 0) const override
  {
    const double width = 1.0;
    double       r2    = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      r2 += p[d] * p[d];
    return std::exp(-r2 / (2. * width * width));
  }
};

/**
 * Initial velocity u_1(x) = du/dt(x, 0).
 * Default: zero (wave packet starts from rest).
 */
template <int dim>
class InitialVelocity : public Function<dim>
{
public:
  InitialVelocity(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> & /*p*/, const unsigned int = 0) const override
  {
    return 0.0;
  }
};

/**
 * Class representing the right-hand side forcing term f(x, t).
 */
template <int dim>
class ForcingTerm : public Function<dim>
{
public:
  ForcingTerm(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> & /*p*/, const unsigned int = 0) const override
  {
    return 0.0; // f = 0 for homogeneous wave equation
  }
};

// Type alias for backwards compatibility
template <int dim>
using InitialCondition = InitialDisplacement<dim>;

// ============================================================================
// Energy Data Structure
// ============================================================================

/**
 * Energy components and dissipation metrics at time t for the simplified
 * damped wave equation:
 *
 *   u_tt - c^2 * Laplacian(u) + gamma * u_t = 0
 *
 * Discrete energy:
 *   E_kin = 0.5 * integral of |u_t|^2 dx            (kinetic)
 *   E_pot = 0.5 * c^2 * integral of |grad u|^2 dx   (potential / elastic)
 *   E_tot = E_kin + E_pot
 *   D     = gamma * integral of |u_t|^2 dx           (instantaneous dissipation rate)
 *   dE    = E_tot(t) - E_tot(0)                       (<= 0 when gamma > 0)
 */
struct EnergyData
{
  double time;             // Current simulation time t
  double kinetic_energy;   // E_kin = 0.5 * ||u_t||^2
  double potential_energy; // E_pot = 0.5 * c^2 * ||grad u||^2
  double total_energy;     // E_tot = E_kin + E_pot
  double dissipation_rate; // D     = gamma * ||u_t||^2
  double energy_decay;     // E_tot(t) - E_tot(0): negative => energy dissipated
};

// ============================================================================
// WaveOperation
// ============================================================================

/**
 * Matrix-free operator for the forced, damped wave equation:
 *
 *   u_tt - c^2 * Laplacian(u) + gamma * u_t = f(x, t)
 *
 * Leapfrog (Stormer-Verlet) time discretisation with Crank-Nicolson damping:
 *
 *   (1 + 0.5*dt*gamma) * M * u^{n+1}
 *       = 2 * M * u^n
 *         - (1 - 0.5*dt*gamma) * M * u^{n-1}
 *         - dt^2 * c^2 * K * u^n
 *         + dt^2 * M * f^n
 *
 * where M is the lumped (diagonal) mass matrix, K is the stiffness matrix,
 * and f^n = f(x, t^n) is the forcing term evaluated at time t^n.
 *
 * Parameters
 * ----------
 * c      : uniform wave speed  (c > 0)
 * gamma  : uniform damping coefficient  (gamma >= 0)
 *          gamma = 0  => energy-conserving undamped wave equation (with f=0)
 *          gamma > 0  => energy-dissipating damped wave equation
 * forcing: right-hand side f(x, t).  Use ForcingTerm<dim> for f = 0.
 */
template <int dim, int fe_degree = 4>
class WaveOperation
{
public:
  WaveOperation(const MatrixFree<dim, double> &data_in,
                const double                   time_step_in,
                const double                   c_in       = 1.0,
                const double                   gamma_in   = 0.0,
                const Function<dim>           *forcing_in = nullptr);

  /**
   * Advance one leapfrog step:
   *   dst    = u^{n+1}
   *   src[0] = u^n,  src[1] = u^{n-1}
   *   current_time = t^n  (used to evaluate f(x, t^n))
   */
  void
  apply(LinearAlgebra::distributed::Vector<double>                      &dst,
        const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
        const double                                                      current_time) const;

  /**
   * Compute PDE-consistent initial acceleration:
   *   a_0 = c^2 * Laplacian(u_0) - gamma * u_1 + f(x, 0)
   * used to start the leapfrog accurately via
   *   u^{-1} = u_0 - dt*u_1 + (dt^2/2)*a_0
   */
  void
  compute_initial_acceleration(
    LinearAlgebra::distributed::Vector<double>       &a_0,
    const LinearAlgebra::distributed::Vector<double> &u_0,
    const LinearAlgebra::distributed::Vector<double> &u_1) const;

  /**
   * Compute energy components using central-difference velocity estimate:
   *   v^n ~ (u^{n+1} - u^{n-1}) / (2*dt)
   *
   *   current_u    = u^n
   *   old_u        = u^{n-1}
   *   next_u       = u^{n+1}
   *   current_time = t^n
   *   initial_energy = E_tot(0)  (for energy_decay field)
   */
  EnergyData
  compute_energy(
    const LinearAlgebra::distributed::Vector<double> &current_u,
    const LinearAlgebra::distributed::Vector<double> &old_u,
    const LinearAlgebra::distributed::Vector<double> &next_u,
    const double                                      current_time,
    const double                                      initial_energy = 0.0) const;

private:
  // Cell-loop kernel for the leapfrog step
  void
  local_apply(
    const MatrixFree<dim, double>                                   &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int>                     &cell_range) const;

  // Cell-loop kernel for the initial acceleration
  void
  local_compute_initial_acceleration(
    const MatrixFree<dim, double>                                   &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int>                     &cell_range) const;

  const MatrixFree<dim, double> &data;
  const double                   time_step;   // dt
  const double                   c;           // wave speed
  const double                   c_sqr;       // c^2 (precomputed)
  const double                   gamma;       // damping coefficient
  const VectorizedArray<double>  delta_t_sqr; // dt^2 (SIMD broadcast)

  // Pointer to the forcing function f(x,t).  nullptr => f = 0 everywhere.
  const Function<dim> *forcing;

  // Evaluation time for f(x, t^n); set thread-safely in apply() before the cell loop.
  mutable double eval_time;

  // Lumped inverse effective mass:  1 / (1 + 0.5*dt*gamma) / M_lumped[i]
  LinearAlgebra::distributed::Vector<double> inv_effective_mass_matrix;

  // Lumped inverse plain mass:  1 / M_lumped[i]   (used for a_0 solve)
  LinearAlgebra::distributed::Vector<double> inv_mass_matrix;
};

// ============================================================================
// WaveProblem  (simulation manager)
// ============================================================================

/**
 * Manages the full simulation lifecycle for:
 *
 *   u_tt - c^2 * Laplacian(u) + gamma * u_t = f(x, t)   on Omega x (0, T]
 *   u = 0                                                 on boundary (Dirichlet)
 *   u(x,0) = u_0(x),  u_t(x,0) = u_1(x)
 *
 * Outputs:
 *   - VTU solution snapshots
 *   - energy_dissipation.csv (time, E_kin, E_pot, E_tot, diss_rate, delta_E)
 */
template <int dim>
class WaveProblem
{
public:
  static constexpr unsigned int fe_degree = 4;

  /**
   * @param final_time_in            End time T (default 30)
   * @param c_in                     Wave speed c > 0 (default 1.0)
   * @param gamma_in                 Damping gamma >= 0 (default 0.0 => conservative)
   * @param forcing_in               Right-hand side f(x,t) (default ForcingTerm => f=0)
   * @param initial_displacement_in  u_0(x)
   * @param initial_velocity_in      u_1(x)
   */
  WaveProblem(
    const double                         final_time_in = 30.0,
    const double                         c_in          = 1.0,
    const double                         gamma_in      = 0.0,
    std::shared_ptr<const Function<dim>> forcing_in    =
      std::make_shared<ForcingTerm<dim>>(),
    std::shared_ptr<const Function<dim>> initial_displacement_in =
      std::make_shared<InitialDisplacement<dim>>(),
    std::shared_ptr<const Function<dim>> initial_velocity_in =
      std::make_shared<InitialVelocity<dim>>())
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD)
#endif
    , fe(QGaussLobatto<1>(fe_degree + 1))
    , dof_handler(triangulation)
    , n_global_refinements(10 - 2 * dim)
    , time(0.0)
    , time_step(10.)
    , final_time(final_time_in)
    , cfl_number(.1 / fe_degree)
    , output_timestep_skip(100)
    , c(c_in)
    , gamma(gamma_in)
    , forcing(forcing_in)
    , initial_displacement(initial_displacement_in)
    , initial_velocity(initial_velocity_in)
    , initial_total_energy(0.0)
  {}

  void run();

  void export_energy_to_csv(const std::string &filename = "energy_dissipation.csv") const;

  const std::vector<EnergyData> &
  get_energy_history() const
  {
    return energy_history;
  }

private:
  void make_grid_and_dofs();
  void output_results(const unsigned int timestep_number);
  void log_energy(const EnergyData &energy_data);

  ConditionalOStream pcout;

#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> triangulation;
#else
  Triangulation<dim> triangulation;
#endif

  const FE_Q<dim>      fe;
  DoFHandler<dim>      dof_handler;
  const MappingQ1<dim> mapping;

  AffineConstraints<double> constraints;
  IndexSet                  locally_relevant_dofs;

  MatrixFree<dim, double> matrix_free_data;

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> old_solution;
  LinearAlgebra::distributed::Vector<double> old_old_solution;

  const unsigned int n_global_refinements;
  double             time;
  double             time_step;
  const double       final_time;
  const double       cfl_number;
  const unsigned int output_timestep_skip;

  // Physical parameters (uniform / constant over the domain)
  const double c;     // wave speed
  const double gamma; // damping coefficient

  std::shared_ptr<const Function<dim>> forcing;             // f(x,t)
  std::shared_ptr<const Function<dim>> initial_displacement;
  std::shared_ptr<const Function<dim>> initial_velocity;

  std::vector<EnergyData> energy_history;
  double                  initial_total_energy;
};

// Type alias for consistency with lab naming
template <int dim>
using WaveEquation = WaveProblem<dim>;

#endif // WAVE_EQUATION_HPP
