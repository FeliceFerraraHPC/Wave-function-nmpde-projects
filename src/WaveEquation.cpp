#include "WaveEquation.hpp"

// ============================================================================
// WaveOperation Implementation
// ============================================================================

template <int dim, int fe_degree>
WaveOperation<dim, fe_degree>::WaveOperation(
  const MatrixFree<dim, double> &data_in,
  const double                   time_step)
  : data(data_in)
  , delta_t_sqr(make_vectorized_array(time_step * time_step))
{
  data.initialize_dof_vector(inv_mass_matrix); // matrix is diagonal, so we can use a vector to store it

  FEEvaluation<dim, fe_degree> fe_eval(data); // provides all the infos for the computation of the mass matrix

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        fe_eval.submit_value(make_vectorized_array(1.), q);
      fe_eval.integrate(EvaluationFlags::values);
      fe_eval.distribute_local_to_global(inv_mass_matrix);
    }

  // we invert the diagonal entries to have the inverse mass matrix, which is needed for the time stepping
  inv_mass_matrix.compress(VectorOperation::add);
  for (unsigned int k = 0; k < inv_mass_matrix.locally_owned_size(); ++k)
    {
      if (inv_mass_matrix.local_element(k) > 1e-15)
        inv_mass_matrix.local_element(k) =
          1. / inv_mass_matrix.local_element(k);
      else
        inv_mass_matrix.local_element(k) = 1;
    }
}

template <int dim, int fe_degree>
void
WaveOperation<dim, fe_degree>::local_apply(
  const MatrixFree<dim, double>                                   &data,
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const std::pair<unsigned int, unsigned int>                     &cell_range)
  const
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
          current.submit_value(2. * current_value - old_value, q); // needed for the time stepping
          current.submit_gradient(-delta_t_sqr * current.get_gradient(q), q); // needed for the Laplacian/spatial term
        }

      current.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      current.distribute_local_to_global(dst);
    }
}

template <int dim, int fe_degree>
void
WaveOperation<dim, fe_degree>::apply(
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const
{
  data.cell_loop( // function defined in FE operator class
    &WaveOperation<dim, fe_degree>::local_apply, this, dst, src, true); // calls the local_apply function for each cell
  dst.scale(inv_mass_matrix);
}

// ============================================================================
// WaveProblem Implementation
// ============================================================================

template <int dim>
void
WaveProblem<dim>::make_grid_and_dofs()
{
  // NOTE: This simple ad hoc refinement could be done better by adapting the mesh to the solution
  // using error estimators during the time stepping as done in other example programs,
  // and using parallel::distributed::SolutionTransfer to transfer the solution to the new mesh.
  
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

  // Enforce u = 0 on all boundary faces (boundary_id = 0 by default for hyper_cube)
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_partition;

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
void
WaveProblem<dim>::run()
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

  // 1. Set time to 0
  time = 0.0;

  // 2. Interpolate u_0 and u_1
  LinearAlgebra::distributed::Vector<double> u_0, u_1;
  u_0.reinit(solution);
  u_1.reinit(solution);

  VectorTools::interpolate(mapping,
                           dof_handler,
                           InitialDisplacement<dim>(),
                           u_0);
  VectorTools::interpolate(mapping,
                           dof_handler,
                           InitialVelocity<dim>(),
                           u_1);

  constraints.distribute(u_0);
  constraints.distribute(u_1);

  // solution corresponds to u^0
  solution = u_0;

  // 3. Compute u^{-1} = u_0 - dt * u_1 + (dt^2 / 2) * Δu_0
  // Note: If u_1 = 0, u^{-1} = u^1, meaning the first leapfrog step reduces to:
  // u^1 = u^0 + (dt^2 / 2) * Δu_0
  // which is equivalent to setting old_solution = u_0 - dt * u_1
  old_solution = u_0;
  old_solution.add(-time_step, u_1); // old_solution = u_0 - dt * u_1

  output_results(0);

  std::vector<LinearAlgebra::distributed::Vector<double> *>
    previous_solutions({&old_solution, &old_old_solution});

  WaveOperation<dim, fe_degree> wave_op(matrix_free_data, time_step);

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
      constraints.distribute(solution); // Restores exact zero values on boundary DOFs
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

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class WaveOperation<2, 4>;
template class WaveProblem<2>;

template class WaveOperation<3, 4>;
template class WaveProblem<3>;
