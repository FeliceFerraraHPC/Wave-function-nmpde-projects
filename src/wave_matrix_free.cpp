/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2012 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University, 2011-2012
 */

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <iostream>
#include <iomanip>

namespace Step48
{
  using namespace dealii;

  const unsigned int dimension = 2;
  const unsigned int fe_degree = 4;

  template <int dim, int fe_degree>
  class WaveOperation
  {
  public:
    WaveOperation(const MatrixFree<dim, double> &data_in,
                        const double                   time_step);

    void apply(LinearAlgebra::distributed::Vector<double> &dst,
               const std::vector<LinearAlgebra::distributed::Vector<double> *>
                 &src) const;

  private:
    const MatrixFree<dim, double>             &data;
    const VectorizedArray<double>              delta_t_sqr;
    LinearAlgebra::distributed::Vector<double> inv_mass_matrix;

    void local_apply(
      const MatrixFree<dim, double>                                   &data,
      LinearAlgebra::distributed::Vector<double>                      &dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
  };

  template <int dim, int fe_degree>
  WaveOperation<dim, fe_degree>::WaveOperation(
    const MatrixFree<dim, double> &data_in,
    const double                   time_step)
    : data(data_in)
    , delta_t_sqr(make_vectorized_array(time_step * time_step))
  {
    data.initialize_dof_vector(inv_mass_matrix);

    FEEvaluation<dim, fe_degree> fe_eval(data);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        fe_eval.reinit(cell);
        for (const unsigned int q : fe_eval.quadrature_point_indices())
          fe_eval.submit_value(make_vectorized_array(1.), q);
        fe_eval.integrate(EvaluationFlags::values);
        fe_eval.distribute_local_to_global(inv_mass_matrix);
      }

    inv_mass_matrix.compress(VectorOperation::add);
    for (unsigned int k = 0; k < inv_mass_matrix.locally_owned_size(); ++k)
      if (inv_mass_matrix.local_element(k) > 1e-15)
        inv_mass_matrix.local_element(k) =
          1. / inv_mass_matrix.local_element(k);
      else
        inv_mass_matrix.local_element(k) = 1;
  }

  template <int dim, int fe_degree>
  void WaveOperation<dim, fe_degree>::local_apply(
    const MatrixFree<dim>                                           &data,
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    AssertDimension(src.size(), 2);
    FEEvaluation<dim, fe_degree> current(data), old(data);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        current.reinit(cell);
        old.reinit(cell);

        current.read_dof_values(*src[0]);
        old.read_dof_values(*src[1]);

        current.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        old.evaluate(EvaluationFlags::values);

        for (const unsigned int q : current.quadrature_point_indices())
          {
            const VectorizedArray<double> current_value = current.get_value(q);
            const VectorizedArray<double> old_value     = old.get_value(q);

            // Leap-frog discretization of the linear wave equation
            // u_tt - Δu = 0:
            // u^{n+1} = 2 u^n - u^{n-1} + dt^2 Δu^n.
            current.submit_value(2. * current_value - old_value, q);
            current.submit_gradient(-delta_t_sqr * current.get_gradient(q), q);
          }

        current.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        current.distribute_local_to_global(dst);
      }
  }

  template <int dim, int fe_degree>
  void WaveOperation<dim, fe_degree>::apply(
    LinearAlgebra::distributed::Vector<double>                      &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const
  {
    data.cell_loop(
      &WaveOperation<dim, fe_degree>::local_apply, this, dst, src, true);
    dst.scale(inv_mass_matrix);
  }

  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition(const unsigned int n_components = 1,
                     const double       time         = 0.)
      : Function<dim>(n_components, time)
    {}
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/) const override
    {
      (void)this->get_time();

      const double width = 1.0;
      double       result = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        result *= std::exp(-(p[d] * p[d]) / (2. * width * width));
      return result;
    }
  };

  template <int dim>
  class WaveProblem
  {
  public:
    WaveProblem();
    void run();

  private:
    ConditionalOStream pcout;

    void make_grid_and_dofs();
    void output_results(const unsigned int timestep_number);

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

    LinearAlgebra::distributed::Vector<double> solution, old_solution,
      old_old_solution;

    const unsigned int n_global_refinements;
    double             time, time_step;
    const double       final_time;
    const double       cfl_number;
    const unsigned int output_timestep_skip;
  };

  template <int dim>
  WaveProblem<dim>::WaveProblem()
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

  template <int dim>
  void WaveProblem<dim>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, -15, 15);
    triangulation.refine_global(n_global_refinements);
    {
      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          if (cell->center().norm() < 11)
            cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();

      for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
          if (cell->center().norm() < 6)
            cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }

    pcout << "   Number of global active cells: "
          << triangulation.n_global_active_cells() << std::endl;

    dof_handler.distribute_dofs(fe);

    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    typename MatrixFree<dim>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim>::AdditionalData::TasksParallelScheme::partition_partition;

    matrix_free_data.reinit(mapping,
                            dof_handler,
                            constraints,
                            QGaussLobatto<1>(fe_degree + 1),
                            additional_data);

    matrix_free_data.initialize_dof_vector(solution);
    old_solution.reinit(solution);
    old_old_solution.reinit(solution);
  }

  template <int dim>
  void
  WaveProblem<dim>::output_results(const unsigned int timestep_number)
  {
    constraints.distribute(solution);

    Vector<float> norm_per_cell(triangulation.n_active_cells());
    solution.update_ghost_values();
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Functions::ZeroFunction<dim>(),
                                      norm_per_cell,
                                      QGauss<dim>(fe_degree + 1),
                                      VectorTools::L2_norm);
    const double solution_norm =
      VectorTools::compute_global_error(triangulation,
                                        norm_per_cell,
                                        VectorTools::L2_norm);

    pcout << "   Time:" << std::setw(8) << std::setprecision(3) << time
          << ", solution norm: " << std::setprecision(5) << std::setw(7)
          << solution_norm << std::endl;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", timestep_number, MPI_COMM_WORLD, 3);

    solution.zero_out_ghost_values();
  }

  template <int dim>
  void WaveProblem<dim>::run()
  {
    {
      pcout << "Number of MPI ranks:            "
            << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;
      pcout << "Number of threads on each rank: "
            << MultithreadInfo::n_threads() << std::endl;
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;
      pcout << "Vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ')'
            << std::endl
            << std::endl;
    }
    make_grid_and_dofs();

    const double local_min_cell_diameter =
      triangulation.last()->diameter() / std::sqrt(dim);
    const double global_min_cell_diameter =
      -Utilities::MPI::max(-local_min_cell_diameter, MPI_COMM_WORLD);
    time_step = cfl_number * global_min_cell_diameter;
    time_step = (final_time - time) / (int((final_time - time) / time_step));
    pcout << "   Time step size: " << time_step
          << ", finest cell: " << global_min_cell_diameter << std::endl
          << std::endl;

    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(1, time),
                             solution);
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(1, time - time_step),
                             old_solution);
    output_results(0);

    std::vector<LinearAlgebra::distributed::Vector<double> *>
      previous_solutions({&old_solution, &old_old_solution});

    WaveOperation<dim, fe_degree> wave_op(matrix_free_data,
                                       time_step);

    unsigned int timestep_number = 1;

    Timer  timer;
    double wtime       = 0;
    double output_time = 0;
    for (time += time_step; time <= final_time;
         time += time_step, ++timestep_number)
      {
        timer.restart();
        old_old_solution.swap(old_solution);
        old_solution.swap(solution);
        wave_op.apply(solution, previous_solutions);
        wtime += timer.wall_time();

        timer.restart();
        if (timestep_number % output_timestep_skip == 0)
          output_results(timestep_number / output_timestep_skip);

        output_time += timer.wall_time();
      }
    timer.restart();
    output_results(timestep_number / output_timestep_skip + 1);
    output_time += timer.wall_time();

    pcout << std::endl
          << "   Performed " << timestep_number << " time steps." << std::endl;

    pcout << "   Average wallclock time per time step: "
          << wtime / timestep_number << 's' << std::endl;

    pcout << "   Spent " << output_time << "s on output and " << wtime
          << "s on computations." << std::endl;
  }
} // namespace Step48

int main(int argc, char **argv)
{
  using namespace Step48;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  try
    {
      WaveProblem<dimension> wave_problem;
      wave_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
