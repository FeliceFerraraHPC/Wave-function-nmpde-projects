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
#include <vector>

using namespace dealii;

/**
 * Class evaluating matrix-free operations for the wave equation.
 */
template <int dim, int fe_degree = 4>
class WaveOperation
{
public:
  // Constructor.
  WaveOperation(const MatrixFree<dim, double> &data_in, const double time_step);

  // Apply operator: dst = WaveOp(src[0], src[1])
  void
  apply(LinearAlgebra::distributed::Vector<double>                      &dst,
        const std::vector<LinearAlgebra::distributed::Vector<double> *> &src)
    const;

private:
  // Local cell loop application.
  void
  local_apply(
    const MatrixFree<dim, double>                                   &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int>                     &cell_range)
    const;

  const MatrixFree<dim, double>             &data;
  const VectorizedArray<double>              delta_t_sqr;
  LinearAlgebra::distributed::Vector<double> inv_mass_matrix;
};

/**
 * Class representing the initial condition (Gaussian wave packet).
 */
template <int dim>
class InitialCondition : public Function<dim>
{
public:
  // Constructor.
  InitialCondition(const unsigned int n_components = 1,
                   const double       time         = 0.)
    : Function<dim>(n_components, time)
  {}

  // Value evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    (void)this->get_time();

    const double width  = 1.0;
    double       result = 1.0;
    for (unsigned int d = 0; d < dim; ++d)
      result *= std::exp(-(p[d] * p[d]) / (2. * width * width));
    return result;
  }
};

/**
 * Class managing the matrix-free wave equation problem.
 */
template <int dim>
class WaveProblem
{
public:
  // Polynomial degree.
  static constexpr unsigned int fe_degree = 4;

  // Constructor.
  WaveProblem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD)
#endif
    , fe(QGaussLobatto<1>(fe_degree + 1))
    , dof_handler(triangulation)
    , n_global_refinements(10 - 2 * dim)
    , time(-10)
    , time_step(10.)
    , final_time(10.)
    , cfl_number(.1 / fe_degree)
    , output_timestep_skip(200)
  {}

  // Run the simulation.
  void
  run();

private:
  // Setup grid, mesh refinement, degrees of freedom, and matrix-free data.
  void
  make_grid_and_dofs();

  // Output solution to file (.vtu/.pvtu) and compute L2 norm.
  void
  output_results(const unsigned int timestep_number);

  // Parallel output stream.
  ConditionalOStream pcout;

#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> triangulation;
#else
  Triangulation<dim> triangulation;
#endif

  const FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

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
};

// Type alias for consistency with lab naming
template <int dim>
using WaveEquation = WaveProblem<dim>;

#endif // WAVE_EQUATION_HPP
