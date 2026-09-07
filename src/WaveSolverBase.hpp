#ifndef WAVE_SOLVER_BASE_HPP
#define WAVE_SOLVER_BASE_HPP

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#include <string>

using namespace dealii;

/**
 * @brief Abstract base class for all wave equation solvers.
 *
 * All three strategies (Theta-scheme, Matrix-Free CG, Matrix-Free DG)
 * implement this interface, enabling polymorphic use in the benchmark driver.
 *
 * Lifecycle:
 *   1. Construct the solver.
 *   2. Call setup(tria) with an externally owned, already-refined triangulation.
 *   3. Call set_initial_conditions(u0, v0).
 *   4. Call run(T) to advance to final time T — returns compute wall time.
 *   5. Optionally call compute_error() against an exact solution.
 *
 * The triangulation must outlive the solver.
 */
template <int dim>
class WaveSolverBase
{
public:
  virtual ~WaveSolverBase() = default;

  /**
   * Initialize the solver on an externally provided triangulation.
   * The triangulation must already be refined; this method only sets up
   * DOFs, sparsity patterns, matrices, and vectors.
   *
   * @param tria  Const reference to the (parallel) triangulation.
   *              Accepts Triangulation<dim> or any derived type
   *              (e.g., parallel::distributed::Triangulation<dim>).
   */
  virtual void
  setup(const Triangulation<dim> &tria) = 0;

  /**
   * Interpolate or project initial conditions for displacement u and
   * velocity v.
   */
  virtual void
  set_initial_conditions(const Function<dim> &u0,
                         const Function<dim> &v0) = 0;

  /**
   * Attach a forcing function (right-hand side).
   */
  virtual void
  set_forcing_function(const Function<dim> *f) = 0;

  /**
   * Run the full time loop from t = 0 to t = T.
   *
   * @param T             Final simulation time.
   * @param write_output  If true, write VTU output files during the loop.
   * @return              Total compute wall time in seconds (I/O excluded).
   */
  virtual double run(double T, bool write_output = false, unsigned int output_frequency = 100) = 0;

  /**
   * Compute the error of the displacement solution against an exact function.
   * The exact_solution must have its time set before calling this.
   *
   * @param norm_type      VectorTools norm type (L2_norm, H1_norm, ...).
   * @param exact_solution Exact solution function at the current time.
   * @return               The requested error norm.
   */
  virtual double
  compute_error(VectorTools::NormType  norm_type,
                const Function<dim>   &exact_solution) const = 0;

  /// Current simulation time.
  virtual double
  current_time() const = 0;

  /// Size of the time step used by the solver.
  virtual double
  time_step_size() const = 0;

  /// Total number of degrees of freedom.
  virtual unsigned int
  n_dofs() const = 0;

  /// Human-readable name of the solver strategy.
  virtual std::string
  name() const = 0;
};

#endif // WAVE_SOLVER_BASE_HPP
