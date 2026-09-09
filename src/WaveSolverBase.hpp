#ifndef WAVE_SOLVER_BASE_HPP
#define WAVE_SOLVER_BASE_HPP

#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace dealii;

// ============================================================================
// Output Path Helpers: automatically route generated files to build/ if running
// from repository root, or into current directory if already in build/
// ============================================================================
inline std::string get_output_path(const std::string &filename)
{
  if (std::filesystem::is_directory("build"))
    return "build/" + filename;
  return filename;
}

inline std::string get_output_dir()
{
  if (std::filesystem::is_directory("build"))
    return "build/";
  return "./";
}

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
  double time = 0.0;             // simulation time of this snapshot
  double kinetic_energy = 0.0;   // E_kin  (natural: v^n = (u^{n+1}-u^{n-1})/(2dt))
  double potential_energy = 0.0; // E_pot  (natural: 0.5*||grad u^n||^2)
  double total_energy = 0.0;     // E_tot = E_kin + E_pot  (natural, O(dt^2) drift)
  double dissipation_rate = 0.0; // D = gamma * ||u_t||^2 = 2*gamma*E_kin
  double energy_decay = 0.0;     // E_tot(t) - E_tot(0)

  // -----------------------------------------------------------------------
  // Staggered half-step energy -- exactly conserved by the leapfrog integrator.
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
  double stag_kinetic_energy = 0.0;   // 0.5 * ||(u^{n+1}-u^n)/dt||^2_M
  double stag_potential_energy = 0.0; // 0.5 * a_h(u^n, u^{n+1})
  double stag_total_energy = 0.0;     // stag_kin + stag_pot
  double stag_energy_decay = 0.0;     // stag_total(t) - stag_total(0)
};

/**
 * Write an energy history (as accumulated by a solver's compute_energy()
 * calls) to a CSV file, including the drift relative to the first
 * recorded snapshot (normally t = 0).
 */
inline void
write_energy_history_csv(const std::vector<EnergyData> &history,
                         const std::string &filename)
{
  const std::string out_path = get_output_path(filename);
  std::ofstream out(out_path);
  // Full double precision (17 significant digits round-trips a double
  // exactly). This matters specifically for stag_total_energy/stag_energy_decay:
  // that quantity is conserved down to ~1e-9-1e-10 (floating-point roundoff),
  // and the default 6-digit stream precision was silently rounding every row
  // to the same literal string, hiding the very thing this diagnostic exists
  // to show.
  out << std::setprecision(17);
  out << "time,kinetic_energy,potential_energy,total_energy,"
         "dissipation_rate,energy_decay,"
         "stag_kinetic_energy,stag_potential_energy,stag_total_energy,stag_energy_decay\n";
  for (const auto &e : history)
    out << e.time << ',' << e.kinetic_energy << ',' << e.potential_energy
        << ',' << e.total_energy << ',' << e.dissipation_rate << ','
        << e.energy_decay << ','
        << e.stag_kinetic_energy << ',' << e.stag_potential_energy << ','
        << e.stag_total_energy << ',' << e.stag_energy_decay << '\n';
}

// ============================================================================
// DispersionData
// ============================================================================
/**
 * @brief Result record produced by the numerical dispersion analysis.
 * One DispersionData entry is created per (solver, polynomial-degree) pair.
 * The phase shift is found by minimising
 *   ||u_num(T) - u_exact(T - s)||_L2   over s in R
 * via ternary search.  The optimal s is the time delay of the numerical
 * solution: positive means the numerical wave is slower (lagging) and
 * negative means it is faster (leading).
 */
struct DispersionData
{
  std::string solver_name;      ///< Human-readable solver label
  unsigned int fe_degree = 0;   ///< Polynomial order p
  unsigned int n_dofs = 0;      ///< Total degrees of freedom
  double wavenumber = 0.0;      ///< Carrier wavenumber k  (rad / length)
  double final_time = 0.0;      ///< Simulation end time T
  double l2_error = 0.0;        ///< ||u_num - u_exact||_{L2}  at T (raw point-by-point error)
  double l2_aligned = 0.0;      ///< ||u_num - u_exact(T - dt)||_{L2} (shape/amplitude error with phase lag removed)
  double phase_shift = 0.0;     ///< dt: optimal time shift (time units)
  double phase_error_rad = 0.0; ///< dphi = k * c * dt  (radians)
  double phase_lag_rel = 0.0;   ///< dt / T  (dimensionless relative lag)
  double peak_x_exact = 0.0;    ///< Analytical peak x-coordinate
  double peak_x_num = 0.0;      ///< Numerical peak x-coordinate
  double peak_amp = 0.0;        ///< Numerical peak amplitude
};

/**
 * @brief 1D peak locator along the x-coordinate around an expected center point.
 * Uses a three-stage strategy:
 *   1. Uniform coarse scan across [center[0] - x_span, center[0] + x_span]
 *   2. Fine sub-grid refinement around the detected highest crest
 *   3. 3-point parabolic interpolation for sub-grid precision
 * @tparam EvalFunc  Callable with signature: double(const Point<dim> &p)
 */
template <int dim, typename EvalFunc>
std::pair<Point<dim>, double>
locate_peak_1d(const EvalFunc &eval,
               const Point<dim> &center,
               const double x_span = 1.5,
               const unsigned int n_pts = 300)
{
  const double dx = (2.0 * x_span) / (n_pts - 1);
  double best_x = center[0];
  double max_val = -1e30;

  // Stage 1: Uniform grid scan
  for (unsigned int i = 0; i < n_pts; ++i)
  {
    const double x = center[0] - x_span + i * dx;
    Point<dim> p = center;
    p[0] = x;
    const double val = eval(p);
    if (val > max_val)
    {
      max_val = val;
      best_x = x;
    }
  }

  // Stage 2: Fine sub-grid refinement around the detected crest
  const double fine_dx = dx / 20.0;
  for (int step = -20; step <= 20; ++step)
  {
    const double x = best_x + step * fine_dx;
    Point<dim> p = center;
    p[0] = x;
    const double val = eval(p);
    if (val > max_val)
    {
      max_val = val;
      best_x = x;
    }
  }

  // Stage 3: Parabolic 3-point sub-grid interpolation
  const double h_fit = fine_dx * 0.5;
  Point<dim> p_left = center;
  p_left[0] = best_x - h_fit;
  Point<dim> p_right = center;
  p_right[0] = best_x + h_fit;
  const double u_l = eval(p_left);
  const double u_r = eval(p_right);
  const double u_m = max_val;
  const double denom = (u_l - 2.0 * u_m + u_r);
  if (std::abs(denom) > 1e-12 && denom < 0.0)
  {
    const double delta = -0.5 * h_fit * (u_r - u_l) / denom;
    if (std::abs(delta) < h_fit)
    {
      best_x += delta;
      max_val = u_m - 0.125 * (u_r - u_l) * (u_r - u_l) / denom;
    }
  }

  Point<dim> peak_pt = center;
  peak_pt[0] = best_x;
  return {peak_pt, max_val};
}

// ============================================================================
// Abstract Base Class
// ============================================================================
/**
 * @brief Abstract base class for all wave equation solvers.
 * All three strategies (Theta-scheme, Matrix-Free CG, Matrix-Free DG)
 * implement this interface, enabling polymorphic use in the benchmark driver.
 * Lifecycle:
 *   1. Construct the solver.
 *   2. Call setup(tria) with an externally owned, already-refined triangulation.
 *   3. Call set_initial_conditions(u0, v0).
 *   4. Call run(T) to advance to final time T -> returns compute wall time.
 *   5. Optionally call compute_error() against an exact solution.
 * The triangulation must outlive the solver.
 */
template <int dim>
class WaveSolverBase
{
public:
  virtual ~WaveSolverBase() = default;

  /**
   * Find the peak (maximum crest) of the numerical displacement field u
   * along a 1D line in x around an expected center.
   *
   * @param center  Expected center Point<dim> (e.g. (x_exact_peak, 0.0)).
   * @param x_span  Half-width of search interval in x (default 1.5).
   * @param n_pts   Number of sample points for initial scan (default 300).
   * @return        Pair of {peak_location_Point, peak_value}.
   */
  virtual std::pair<Point<dim>, double>
  find_peak(const Point<dim> &center,
            double x_span = 1.5,
            unsigned int n_pts = 300) const = 0;

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
   * velocity v. For leapfrog solvers, u_prev can optionally provide an exact
   * u(-dt) to eliminate the initial first-step truncation error.
   */
  virtual void
  set_initial_conditions(const Function<dim> &u0,
                         const Function<dim> &v0,
                         const Function<dim> *u_prev = nullptr) = 0;

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
  compute_error(VectorTools::NormType norm_type,
                const Function<dim> &exact_solution) const = 0;

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

  /// Optional: override time step size before run().
  virtual void
  set_time_step(double dt)
  {
    (void)dt;
  }

  /// Total number of degrees of freedom.
  virtual unsigned int
  n_dofs() const = 0;

  /// Human-readable name of the solver strategy.
  virtual std::string
  name() const = 0;

  enum class BoundaryType
  {
    Dirichlet,
    Neumann
  };

  virtual void
  set_boundary_type(BoundaryType bt)
  {
    boundary_type_ = bt;
  }

  virtual void
  set_non_homogeneous(bool nh, const Function<dim> *exact_solution = nullptr)
  {
    non_homogeneous_ = nh;
    exact_solution_ = exact_solution;
  }

  virtual bool
  is_non_homogeneous() const
  {
    return non_homogeneous_;
  }

protected:
  BoundaryType boundary_type_ = BoundaryType::Dirichlet;
  bool non_homogeneous_ = false;
  const Function<dim> *exact_solution_ = nullptr;
};

#endif // WAVE_SOLVER_BASE_HPP