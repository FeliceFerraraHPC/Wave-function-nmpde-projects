#ifndef WAVE_FUNCTIONS_HPP
#define WAVE_FUNCTIONS_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <cmath>

using namespace dealii;

/**
 * @brief Shared problem functions used by all three solvers in benchmark mode.
 *
 * Physical problem: homogeneous wave equation
 *   u_tt - Delta u = 0   on  Omega x (0, T)
 *   u = 0                on  dOmega x (0, T)
 *   u(x, 0)  = u0(x)     (Gaussian wave packet)
 *   u_t(x,0) = 0         (starts from rest)
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
// Manufactured exact solution for the theta-scheme convergence study.
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

#endif // WAVE_FUNCTIONS_HPP
