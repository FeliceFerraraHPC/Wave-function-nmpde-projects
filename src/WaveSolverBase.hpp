#ifndef WAVE_SOLVER_BASE_HPP
#define WAVE_SOLVER_BASE_HPP

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <string>
#include <vector>

using namespace dealii;

// ============================================================================
// EnergyData
// ============================================================================

/**
 * @brief Energy snapshot for the damped wave equation u_tt - Delta u + gamma*u_t = f.
 *
 *   E_kin = 0.5 * integral of |u_t|^2 dx        (kinetic energy)
 *   E_pot = 0.5 * integral of |grad u|^2 dx     (potential / elastic energy)
 *   E_tot = E_kin + E_pot
 *   D     = gamma * integral of |u_t|^2 dx = 2*gamma*E_kin  (dissipation rate)
 *   energy_decay = E_tot(t) - E_tot(0)          (<= 0 for gamma > 0)
 *
 * With gamma = 0 (the previous default), E_tot(t) should stay constant for a
 * correct, stable discretization -- the sanity check discussed before. With
 * gamma > 0, E_tot(t) should instead decay, and the instantaneous rate of
 * that decay should match D. Comparing E_tot(t) against E_tot(0) (or dE_tot/dt
 * against -D) is a cheap, always-available consistency check that
 * complements (but does not replace) a real manufactured-solution
 * error/convergence study, since a scheme can conserve/dissipate this
 * discrete energy correctly while still converging to the wrong solution
 * (e.g. dispersion error) -- see compute_error() for that.
 */
struct EnergyData
{
  double time             = 0.0; // simulation time of this snapshot
  double kinetic_energy   = 0.0; // E_kin  (natural: v^n = (u^{n+1}-u^{n-1})/(2dt))
  double potential_energy = 0.0; // E_pot  (natural: 0.5*||grad u^n||^2)
  double total_energy     = 0.0; // E_tot = E_kin + E_pot  (natural, O(dt^2) drift)
  double dissipation_rate = 0.0; // D = gamma * ||u_t||^2 = 2*gamma*E_kin
  double energy_decay     = 0.0; // E_tot(t) - E_tot(0)

  // -----------------------------------------------------------------------
  // Staggered half-step energy — exactly conserved by the leapfrog integrator.
  //
  //   E^{n+1/2}_stag  =  0.5 * ||(u^{n+1} - u^n)/dt||^2_M           (kinetic)
  //                   +  0.5 *  a_h(u^n, u^{n+1})                    (potential)
  //
  // The kinetic term uses the lumped mass M (Gauss-Lobatto diagonal) so the
  // inner product is the numerically exact discrete conserved quantity.
  // The potential a_h is the full discrete bilinear form: volume grad-grad
  // term for CG, and volume + SIPG face penalty terms for DG.
  //
  // For the theta-scheme (which tracks v^n explicitly) the staggered formula
  // is defined analogously using the backward difference (u^n - u^{n-1})/dt
  // so that all three solvers expose a comparable observable.
  // -----------------------------------------------------------------------
  double stag_kinetic_energy   = 0.0; // 0.5 * ||(u^{n+1}-u^n)/dt||^2_M
  double stag_potential_energy = 0.0; // 0.5 * a_h(u^n, u^{n+1})
  double stag_total_energy     = 0.0; // stag_kin + stag_pot
  double stag_energy_decay     = 0.0; // stag_total(t) - stag_total(0)
};

/**
 * Write an energy history (as accumulated by a solver's compute_energy()
 * calls) to a CSV file, including the drift relative to the first
 * recorded snapshot (normally t = 0).
 */
inline void
write_energy_history_csv(const std::vector<EnergyData> &history,
                         const std::string             &filename)
{
  std::ofstream out(filename);
  out << "time,kinetic_energy,potential_energy,total_energy,"
         "dissipation_rate,energy_decay,"
         "stag_kinetic_energy,stag_potential_energy,stag_total_energy,stag_energy_decay\n";

  for (const auto &e : history)
    out << e.time << ',' << e.kinetic_energy << ',' << e.potential_energy
        << ',' << e.total_energy << ',' << e.dissipation_rate << ','
        << e.energy_decay << ','
        << e.stag_kinetic_energy << ',' << e.stag_potential_energy << ','
        << e.stag_total_energy  << ',' << e.stag_energy_decay << '\n';
}

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
   * Run the full time loop from t = 0 to t = T.
   *
   * @param T             Final simulation time.
   * @param write_output  If true, write VTU output files during the loop.
   * @return              Total compute wall time in seconds (I/O excluded).
   */
  virtual double
  run(double T, bool write_output = false) = 0;

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

  /**
   * Compute the kinetic/potential/total energy of the solver's *current*
   * state (i.e. the most recently completed time step). Each solver
   * computes this the way that is natural/exact for its own internal
   * representation (see the .cpp files for details).
   */
  virtual EnergyData
  compute_energy() const = 0;

  /// Energy snapshots recorded periodically during run().
  virtual const std::vector<EnergyData> &
  get_energy_history() const = 0;

  /// Write get_energy_history() to a CSV file (time, E_kin, E_pot, E_tot, drift).
  virtual void
  export_energy_to_csv(const std::string &filename) const = 0;

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
