#ifndef WAVE_FUNCTIONS_HPP
#define WAVE_FUNCTIONS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <cmath>

using namespace dealii;

/**
 * @brief Shared problem functions used by all three solvers in benchmark mode.
 *
 * Physical problem: damped wave equation
 *   u_tt - Delta u + gamma*u_t = 0   on  Omega x (0, T)
 *   u = 0                            on  dOmega x (0, T)
 *   u(x, 0)  = u0(x)                 (Gaussian wave packet)
 *   u_t(x,0) = 0                     (starts from rest)
 *
 * gamma >= 0 is the damping coefficient, set per-solver at construction
 * time (default 0 => undamped, energy-conserving case).
 */

// ---------------------------------------------------------------------------
// Gaussian initial displacement  u0(x) = exp(-|x|^2 / (2 * width^2))
// ---------------------------------------------------------------------------
template <int dim>
class InitialDisplacement : public Function<dim>
{
public:
  explicit InitialDisplacement(const double width = 1.0,
                               const double time  = 0.)
    : Function<dim>(1, time)
    , width_(width)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    double r2 = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
      r2 += p[d] * p[d];
    return std::exp(-r2 / (2.0 * width_ * width_));
  }

private:
  const double width_;
};

// ---------------------------------------------------------------------------
// Zero initial velocity  v0(x) = 0  (wave starts from rest)
// ---------------------------------------------------------------------------
template <int dim>
class InitialVelocity : public Function<dim>
{
public:
  explicit InitialVelocity(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
  {
    return 0.0;
  }
};

// ---------------------------------------------------------------------------
// Zero forcing term  f(x,t) = 0  (homogeneous wave equation)
// ---------------------------------------------------------------------------
template <int dim>
class ZeroForcing : public Function<dim>
{
public:
  explicit ZeroForcing(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
  {
    return 0.0;
  }
};

// ---------------------------------------------------------------------------
// Standing wave manufactured exact solution (MMS) for convergence studies:
//   u_exact(x, t) = cos(omega * t) * prod_{d=0}^{dim-1} sin(x_d)
//   with omega = sqrt(dim) and c = 1.
//
// Solves:  u_tt - Delta u = 0   on  (0, pi)^dim x (0, T]
//   with u = 0 on dOmega, u(x, 0) = prod sin(x_d), u_t(x, 0) = 0, and f = 0.
// ---------------------------------------------------------------------------
template <int dim>
class StandingWaveExact : public Function<dim>
{
public:
  explicit StandingWaveExact(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t     = this->get_time();
    const double omega = std::sqrt(static_cast<double>(dim));
    double       val   = std::cos(omega * t);
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(p[d]);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double   t           = this->get_time();
    const double   omega       = std::sqrt(static_cast<double>(dim));
    const double   time_factor = std::cos(omega * t);
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
    {
      double spatial_deriv = std::cos(p[d]);
      for (unsigned int j = 0; j < dim; ++j)
        if (j != d)
          spatial_deriv *= std::sin(p[j]);
      grad[d] = spatial_deriv * time_factor;
    }
    return grad;
  }
};

template <int dim>
class StandingWaveIC : public Function<dim>
{
public:
  explicit StandingWaveIC(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    double val = 1.0;
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(p[d]);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
    {
      double spatial_deriv = std::cos(p[d]);
      for (unsigned int j = 0; j < dim; ++j)
        if (j != d)
          spatial_deriv *= std::sin(p[j]);
      grad[d] = spatial_deriv;
    }
    return grad;
  }
};

template <int dim>
class StandingWaveV0 : public Function<dim>
{
public:
  explicit StandingWaveV0(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
  {
    return 0.0;
  }
};

// ---------------------------------------------------------------------------
// Acoustic sound wave exact solution for Neumann boundary conditions:
//   u_exact(x, t) = cos(omega * t) * prod_{d=0}^{dim-1} cos(x_d)
//   with omega = sqrt(dim) and c = 1 on Omega = (0, pi)^dim x (0, T].
//
// Solves:  u_tt - Delta u = 0
//   with grad(u) . n = 0 on dOmega (sound-hard walls, homogeneous Neumann),
//   u(x, 0) = prod cos(x_d), u_t(x, 0) = 0, and f = 0.
// ---------------------------------------------------------------------------
template <int dim>
class AcousticSoundWaveExact : public Function<dim>
{
public:
  explicit AcousticSoundWaveExact(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t     = this->get_time();
    const double omega = std::sqrt(static_cast<double>(dim));
    double       val   = std::cos(omega * t);
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::cos(p[d]);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double   t           = this->get_time();
    const double   omega       = std::sqrt(static_cast<double>(dim));
    const double   time_factor = std::cos(omega * t);
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
    {
      double spatial_deriv = -std::sin(p[d]);
      for (unsigned int j = 0; j < dim; ++j)
        if (j != d)
          spatial_deriv *= std::cos(p[j]);
      grad[d] = spatial_deriv * time_factor;
    }
    return grad;
  }
};

template <int dim>
class AcousticSoundWaveIC : public Function<dim>
{
public:
  explicit AcousticSoundWaveIC(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    double val = 1.0;
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::cos(p[d]);
    return val;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
    {
      double spatial_deriv = -std::sin(p[d]);
      for (unsigned int j = 0; j < dim; ++j)
        if (j != d)
          spatial_deriv *= std::cos(p[j]);
      grad[d] = spatial_deriv;
    }
    return grad;
  }
};

template <int dim>
class AcousticSoundWaveV0 : public Function<dim>
{
public:
  explicit AcousticSoundWaveV0(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/ = 0) const override
  {
    return 0.0;
  }
};

// ---------------------------------------------------------------------------
// Localized acoustic sound pulse (pressure blast) in a cavity/duct:
//   u0(x) = amplitude * exp(-|x - center|^2 / (2 * width^2))
// Ideal for observing sound wave reflections off rigid (Neumann) walls
// and highlighting DG's sharp gradient resolution without Gibbs oscillations.
// ---------------------------------------------------------------------------
template <int dim>
class AcousticPulseIC : public Function<dim>
{
public:
  explicit AcousticPulseIC(const Point<dim> &center    = Point<dim>(),
                           const double      width     = 1.0,
                           const double      amplitude = 1.0)
    : Function<dim>(1, 0.0)
    , center_(center)
    , width_(width)
    , amplitude_(amplitude)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    double r2 = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
    {
      const double diff = p[d] - center_[d];
      r2 += diff * diff;
    }
    return amplitude_ * std::exp(-r2 / (2.0 * width_ * width_));
  }

private:
  const Point<dim> center_;
  const double      width_;
  const double      amplitude_;
};

// Aliases for convenience
template <int dim>
using ExactSolution = StandingWaveExact<dim>;

// ---------------------------------------------------------------------------
// Legacy manufactured exact solution for the theta-scheme convergence study.
//   u(x,y,t)   = t^2 * sin(pi*x) * sin(pi*y)
//   u_t(x,y,t) = 2t  * sin(pi*x) * sin(pi*y)
//   RHS f       = 2 * sin(pi*x) * sin(pi*y) * (1 + t^2 * pi^2)
// ---------------------------------------------------------------------------
template <int dim>
class ManufacturedSolutionU : public Function<dim>
{

public:
  explicit ManufacturedSolutionU(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    return t * t * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    Tensor<1, dim> g;
    g[0] = t * t * M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]);
    g[1] = t * t * M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]);
    return g;
  }
};

template <int dim>
class ManufacturedSolutionV : public Function<dim>
{

public:
  explicit ManufacturedSolutionV(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    return 2.0 * t * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
  }
};

template <int dim>
class ManufacturedRHS : public Function<dim>
{

public:
  explicit ManufacturedRHS(const double time = 0.)
    : Function<dim>(1, time)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    return 2.0 * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) *
           (1.0 + t * t * M_PI * M_PI);
  }
};

// ---------------------------------------------------------------------------
// Gaussian-modulated sinusoidal wave packet — initial displacement.
//
//   dim=1:  u0(x) = exp( -(x-x0)^2/(2σ^2) ) * cos( k*(x-x0) )
//   dim=2:  u0(x,y) = [above] * sin( π*(y-y_min)/L_y )
//
// The sin factor in y ensures u0 = 0 on the top/bottom walls y = y_min and
// y = y_min + L_y, making the IC compatible with homogeneous Dirichlet BCs
// on all four sides of the square domain.  With L_y = 30, y_min = -15 the
// factor modifies the phase speed by only ≈ 1.4e-4 (negligible).
//
// The exact solution of the 2D wave equation (c=1) with this IC is:
//   u_exact(x,y,t) = exp(-(x-x0-t)^2/(2σ^2)) * cos(k*(x-x0-t))
//                    * sin(π*(y-y_min)/L_y)
// to high accuracy (see GaussianSinusoidExact).
// ---------------------------------------------------------------------------
template <int dim>
class GaussianSinusoidIC : public Function<dim>
{
public:
  explicit GaussianSinusoidIC(const double k     = 2.0 * M_PI,
                               const double x0    = -8.0,
                               const double sigma = 1.0,
                               const double y_min = -15.0,
                               const double L_y   = 30.0)
    : Function<dim>(1, 0.0), k_(k), x0_(x0), sigma_(sigma),
      y_min_(y_min), L_y_(L_y)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double xi  = p[0] - x0_;
    double val = std::exp(-xi * xi / (2.0 * sigma_ * sigma_)) * std::cos(k_ * xi);
    // In dim==2 multiply by a y-mode that is 0 on top/bottom walls and 1 at
    // the domain midpoint, making the IC exactly compatible with Dirichlet BCs.
    if constexpr (dim == 2)
      val *= std::sin(M_PI * (p[1] - y_min_) / L_y_);
    return val;
  }

private:
  const double k_;
  const double x0_;
  const double sigma_;
  const double y_min_;
  const double L_y_;
};

// ---------------------------------------------------------------------------
// Gaussian-modulated sinusoidal wave packet — initial velocity.
//
//   v0 = u_t(x,0) = -c * ∂u0/∂x
//      = c * exp(-(x-x0)^2/(2σ^2))
//        * [ (x-x0)/σ^2 * cos(k*(x-x0)) + k*sin(k*(x-x0)) ]
//        * sin(π*(y-y_min)/L_y)   (dim==2 only)
//
// Setting v0 = -c * ∂u0/∂x drives a purely rightward-traveling wave at t=0.
// ---------------------------------------------------------------------------
template <int dim>
class GaussianSinusoidV0 : public Function<dim>
{
public:
  explicit GaussianSinusoidV0(const double k     = 2.0 * M_PI,
                               const double x0    = -8.0,
                               const double sigma = 1.0,
                               const double c     = 1.0,
                               const double y_min = -15.0,
                               const double L_y   = 30.0)
    : Function<dim>(1, 0.0), k_(k), x0_(x0), sigma_(sigma), c_(c),
      y_min_(y_min), L_y_(L_y)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double xi  = p[0] - x0_;
    const double env = std::exp(-xi * xi / (2.0 * sigma_ * sigma_));
    double val = c_ * env *
                 ((xi / (sigma_ * sigma_)) * std::cos(k_ * xi) +
                  k_ * std::sin(k_ * xi));
    if constexpr (dim == 2)
      val *= std::sin(M_PI * (p[1] - y_min_) / L_y_);
    return val;
  }

private:
  const double k_;
  const double x0_;
  const double sigma_;
  const double c_;
  const double y_min_;
  const double L_y_;
};

// ---------------------------------------------------------------------------
// Gaussian-modulated sinusoidal wave packet — exact solution.
//
//   dim=1:  u_exact(x,t)   = exp(-(x-x0-c*t)^2/(2σ^2)) * cos(k*(x-x0-c*t))
//   dim=2:  u_exact(x,y,t) = [above] * sin(π*(y-y_min)/L_y)
//
// The 2D exact solution corresponds to the modified wave equation
//   u_tt - u_xx + (π/L_y)^2 * u = 0  (Klein-Gordon in x with y-mode)
// whose phase speed is c_ph = sqrt(1 + (π/(k*L_y))^2) ≈ 1 + 1.4e-4 for
// k=2π, L_y=30 — negligible, so c=1 is used throughout.
//
// Call set_time(t) before passing to compute_error().
// ---------------------------------------------------------------------------
template <int dim>
class GaussianSinusoidExact : public Function<dim>
{
public:
  explicit GaussianSinusoidExact(const double k     = 2.0 * M_PI,
                                  const double x0    = -8.0,
                                  const double sigma = 1.0,
                                  const double c     = 1.0,
                                  const double y_min = -15.0,
                                  const double L_y   = 30.0)
    : Function<dim>(1, 0.0), k_(k), x0_(x0), sigma_(sigma), c_(c),
      y_min_(y_min), L_y_(L_y)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    double c_eff = c_;
    if constexpr (dim == 2)
      c_eff = std::sqrt(c_ * c_ + (M_PI * M_PI) / (k_ * k_ * L_y_ * L_y_));
    const double xi = p[0] - x0_ - c_eff * t;
    double val = std::exp(-xi * xi / (2.0 * sigma_ * sigma_)) * std::cos(k_ * xi);
    if constexpr (dim == 2)
      val *= std::sin(M_PI * (p[1] - y_min_) / L_y_);
    return val;
  }

private:
  const double k_;
  const double x0_;
  const double sigma_;
  const double c_;
  const double y_min_;
  const double L_y_;
};

#endif // WAVE_FUNCTIONS_HPP
