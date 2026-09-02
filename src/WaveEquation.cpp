#include "WaveEquation.hpp"

// ============================================================================
// WaveOperation — Constructor
// ============================================================================

template <int dim, int fe_degree>
WaveOperation<dim, fe_degree>::WaveOperation(
  const MatrixFree<dim, double> &data_in,
  const double                   time_step_in,
  const double                   c_in,
  const double                   gamma_in,
  const Function<dim>           *forcing_in)
  : data(data_in)
  , time_step(time_step_in)
  , c(c_in)
  , c_sqr(c_in * c_in)
  , gamma(gamma_in)
  , delta_t_sqr(make_vectorized_array(time_step_in * time_step_in))
  , forcing(forcing_in)
  , eval_time(0.0)
{
  // -------------------------------------------------------------------------
  // Assemble both lumped diagonal mass matrices in a single cell loop:
  //
  //   inv_effective_mass_matrix[i]  ~ 1 / [(1 + 0.5*dt*gamma) * M_lumped[i]]
  //   inv_mass_matrix[i]            ~ 1 / M_lumped[i]
  //
  // Since gamma is uniform, the effective mass is just a global rescaling of
  // the plain mass matrix.  We still build them separately so the code remains
  // correct if gamma is later made spatially varying.
  // -------------------------------------------------------------------------
  data.initialize_dof_vector(inv_effective_mass_matrix);
  data.initialize_dof_vector(inv_mass_matrix);

  FEEvaluation<dim, fe_degree> fe_eval(data);

  const double eff_mass_weight = 1.0 + 0.5 * time_step_in * gamma_in; // (1 + 0.5*dt*gamma)

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);

      // Submit unit value at every quadrature point — this builds the lumped
      // mass row sums (diagonal of M when using Gauss–Lobatto + FE_Q).
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        fe_eval.submit_value(make_vectorized_array(1.0), q);

      fe_eval.integrate(EvaluationFlags::values);

      // Accumulate into inv_effective_mass_matrix (will invert below)
      fe_eval.distribute_local_to_global(inv_effective_mass_matrix);

      // Accumulate into inv_mass_matrix (same values, different scaling applied below)
      fe_eval.distribute_local_to_global(inv_mass_matrix);
    }

  // Compress parallel additions and invert the diagonals.
  inv_effective_mass_matrix.compress(VectorOperation::add);
  inv_mass_matrix.compress(VectorOperation::add);

  for (unsigned int k = 0; k < inv_effective_mass_matrix.locally_owned_size(); ++k)
    {
      const double val = inv_effective_mass_matrix.local_element(k);
      if (std::abs(val) > 1e-15)
        {
          inv_effective_mass_matrix.local_element(k) = 1.0 / (eff_mass_weight * val);
          inv_mass_matrix.local_element(k)           = 1.0 / val;
        }
      else
        {
          inv_effective_mass_matrix.local_element(k) = 1.0;
          inv_mass_matrix.local_element(k)           = 1.0;
        }
    }
}

// ============================================================================
// WaveOperation — local_apply  (leapfrog step kernel)
// ============================================================================

template <int dim, int fe_degree>
void
WaveOperation<dim, fe_degree>::local_apply(
  const MatrixFree<dim, double>                                   & /*data*/,
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const std::pair<unsigned int, unsigned int>                     &cell_range) const
{
  AssertDimension(src.size(), 2);

  FEEvaluation<dim, fe_degree> current(data), old(data);

  // Precompute scalar coefficients for the leapfrog RHS:
  //   val_coeff_curr  =  2.0                      (coefficient of u^n)
  //   val_coeff_old   = -(1 - 0.5*dt*gamma)       (coefficient of u^{n-1})
  //   grad_coeff      = -dt^2 * c^2               (stiffness contribution)
  //
  // Full update (before applying inv_effective_mass_matrix):
  //   RHS = 2*M*u^n - (1 - 0.5*dt*gamma)*M*u^{n-1} - dt^2*c^2*K*u^n + dt^2*M*f^n
  const double val_coeff_curr = 2.0;
  const double val_coeff_old  = -(1.0 - 0.5 * time_step * gamma);
  const double dt_sqr         = time_step * time_step;
  const VectorizedArray<double> grad_coeff =
    make_vectorized_array(-c_sqr) * delta_t_sqr;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      current.reinit(cell);
      old.reinit(cell);

      current.read_dof_values(*src[0]); // u^n
      old.read_dof_values(*src[1]);     // u^{n-1}

      current.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      old.evaluate(EvaluationFlags::values);

      const unsigned int n_active_lanes =
        data.n_active_entries_per_cell_batch(cell);

      for (const unsigned int q : current.quadrature_point_indices())
        {
          // Value term: 2*u^n - (1 - 0.5*dt*gamma)*u^{n-1} + dt^2*f(x, t^n)
          VectorizedArray<double> f_val = make_vectorized_array(0.0);
          if (forcing != nullptr)
            {
              const Point<dim, VectorizedArray<double>> q_point =
                current.quadrature_point(q);
              for (unsigned int v = 0; v < n_active_lanes; ++v)
                {
                  Point<dim> p;
                  for (unsigned int d = 0; d < dim; ++d)
                    p[d] = q_point[d][v];
                  f_val[v] = forcing->value(p);
                }
            }

          const VectorizedArray<double> val_term =
            make_vectorized_array(val_coeff_curr) * current.get_value(q) +
            make_vectorized_array(val_coeff_old)  * old.get_value(q) +
            make_vectorized_array(dt_sqr) * f_val;
          current.submit_value(val_term, q);

          // Gradient (stiffness) term: -dt^2 * c^2 * grad(u^n)
          current.submit_gradient(grad_coeff * current.get_gradient(q), q);
        }

      current.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      current.distribute_local_to_global(dst);
    }
}

// ============================================================================
// WaveOperation — apply  (public interface)
// ============================================================================

template <int dim, int fe_degree>
void
WaveOperation<dim, fe_degree>::apply(
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const double                                                      current_time) const
{
  // Set evaluation time for f(x, t^n) before the parallel cell loop.
  eval_time = current_time;
  if (forcing != nullptr)
    const_cast<Function<dim> *>(forcing)->set_time(eval_time);

  data.cell_loop(&WaveOperation<dim, fe_degree>::local_apply, this, dst, src, true);

  // Scale by the inverted effective mass diagonal to obtain u^{n+1}.
  dst.scale(inv_effective_mass_matrix);
}

// ============================================================================
// WaveOperation — local_compute_initial_acceleration  (kernel)
// ============================================================================

template <int dim, int fe_degree>
void
WaveOperation<dim, fe_degree>::local_compute_initial_acceleration(
  const MatrixFree<dim, double>                                   & /*data*/,
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const std::pair<unsigned int, unsigned int>                     &cell_range) const
{
  AssertDimension(src.size(), 2);

  FEEvaluation<dim, fe_degree> u0_eval(data), u1_eval(data);

  // a_0 = c^2 * Laplacian(u_0) - gamma * u_1 + f(x, 0)
  //
  // Weak form (tested against phi_i):
  //   <a_0, phi_i> = -c^2 <grad u_0, grad phi_i>   (integration by parts, u=0 on boundary)
  //                  - gamma <u_1, phi_i>
  //                  + <f(x,0), phi_i>
  const VectorizedArray<double> c_sqr_simd    = make_vectorized_array(c_sqr);
  const VectorizedArray<double> gamma_neg_simd = make_vectorized_array(-gamma);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      u0_eval.reinit(cell);
      u1_eval.reinit(cell);

      u0_eval.read_dof_values(*src[0]); // u_0
      u1_eval.read_dof_values(*src[1]); // u_1

      u0_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      u1_eval.evaluate(EvaluationFlags::values);

      const unsigned int n_active_lanes =
        data.n_active_entries_per_cell_batch(cell);

      for (const unsigned int q : u0_eval.quadrature_point_indices())
        {
          // Value contribution: -gamma*u_1 + f(x, 0)
          VectorizedArray<double> f0_val = make_vectorized_array(0.0);
          if (forcing != nullptr)
            {
              const Point<dim, VectorizedArray<double>> q_point =
                u0_eval.quadrature_point(q);
              for (unsigned int v = 0; v < n_active_lanes; ++v)
                {
                  Point<dim> p;
                  for (unsigned int d = 0; d < dim; ++d)
                    p[d] = q_point[d][v];
                  f0_val[v] = forcing->value(p); // time already set to 0 in public interface
                }
            }

          u0_eval.submit_value(gamma_neg_simd * u1_eval.get_value(q) + f0_val, q);

          // Gradient contribution: -c^2 * grad(u_0)  (weak Laplacian)
          u0_eval.submit_gradient(-c_sqr_simd * u0_eval.get_gradient(q), q);
        }

      u0_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      u0_eval.distribute_local_to_global(dst);
    }
}

// ============================================================================
// WaveOperation — compute_initial_acceleration  (public interface)
// ============================================================================

template <int dim, int fe_degree>
void
WaveOperation<dim, fe_degree>::compute_initial_acceleration(
  LinearAlgebra::distributed::Vector<double>       &a_0,
  const LinearAlgebra::distributed::Vector<double> &u_0,
  const LinearAlgebra::distributed::Vector<double> &u_1) const
{
  // Set forcing time to t=0 before the cell loop.
  if (forcing != nullptr)
    const_cast<Function<dim> *>(forcing)->set_time(0.0);

  std::vector<LinearAlgebra::distributed::Vector<double> *> non_const_src = {
    const_cast<LinearAlgebra::distributed::Vector<double> *>(&u_0),
    const_cast<LinearAlgebra::distributed::Vector<double> *>(&u_1)};

  data.cell_loop(
    &WaveOperation<dim, fe_degree>::local_compute_initial_acceleration,
    this,
    a_0,
    non_const_src,
    true);

  // Invert the plain mass matrix to recover the pointwise acceleration.
  a_0.scale(inv_mass_matrix);
}

// ============================================================================
// WaveOperation — compute_energy
// ============================================================================

template <int dim, int fe_degree>
EnergyData
WaveOperation<dim, fe_degree>::compute_energy(
  const LinearAlgebra::distributed::Vector<double> &current_u,
  const LinearAlgebra::distributed::Vector<double> &old_u,
  const LinearAlgebra::distributed::Vector<double> &next_u,
  const double                                      current_time,
  const double                                      initial_energy) const
{
  current_u.update_ghost_values();
  old_u.update_ghost_values();
  next_u.update_ghost_values();

  FEEvaluation<dim, fe_degree> eval_curr(data), eval_old(data), eval_next(data);

  double local_kin_energy  = 0.0;
  double local_pot_energy  = 0.0;
  double local_dissipation = 0.0;

  const double inv_2dt      = 1.0 / (2.0 * time_step); // for central-difference velocity
  const double half_c_sqr   = 0.5 * c_sqr;

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
    {
      eval_curr.reinit(cell);
      eval_old.reinit(cell);
      eval_next.reinit(cell);

      eval_curr.read_dof_values(current_u); // u^n
      eval_old.read_dof_values(old_u);      // u^{n-1}
      eval_next.read_dof_values(next_u);    // u^{n+1}

      eval_curr.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      eval_old.evaluate(EvaluationFlags::values);
      eval_next.evaluate(EvaluationFlags::values);

      const unsigned int n_active_lanes =
        data.n_active_entries_per_cell_batch(cell);

      for (const unsigned int q : eval_curr.quadrature_point_indices())
        {
          // Central-difference velocity: v^n = (u^{n+1} - u^{n-1}) / (2*dt)
          const VectorizedArray<double> v_val =
            (eval_next.get_value(q) - eval_old.get_value(q)) *
            make_vectorized_array(inv_2dt);

          // |grad u^n|^2
          const Tensor<1, dim, VectorizedArray<double>> u_grad =
            eval_curr.get_gradient(q);
          VectorizedArray<double> grad_u_sqr = make_vectorized_array(0.0);
          for (unsigned int d = 0; d < dim; ++d)
            grad_u_sqr += u_grad[d] * u_grad[d];

          const VectorizedArray<double> JxW = eval_curr.JxW(q);
          const VectorizedArray<double> v_sqr = v_val * v_val;

          // E_kin density = 0.5 * |v|^2
          const VectorizedArray<double> kin_dense =
            make_vectorized_array(0.5) * v_sqr * JxW;

          // E_pot density = 0.5 * c^2 * |grad u|^2
          const VectorizedArray<double> pot_dense =
            make_vectorized_array(half_c_sqr) * grad_u_sqr * JxW;

          // Dissipation rate density = gamma * |v|^2
          const VectorizedArray<double> diss_dense =
            make_vectorized_array(gamma) * v_sqr * JxW;

          for (unsigned int v = 0; v < n_active_lanes; ++v)
            {
              local_kin_energy  += kin_dense[v];
              local_pot_energy  += pot_dense[v];
              local_dissipation += diss_dense[v];
            }
        }
    }

  current_u.zero_out_ghost_values();
  old_u.zero_out_ghost_values();
  next_u.zero_out_ghost_values();

  // Global MPI reductions
  const double global_kin_energy  = Utilities::MPI::sum(local_kin_energy,  MPI_COMM_WORLD);
  const double global_pot_energy  = Utilities::MPI::sum(local_pot_energy,  MPI_COMM_WORLD);
  const double global_dissipation = Utilities::MPI::sum(local_dissipation, MPI_COMM_WORLD);
  const double global_tot_energy  = global_kin_energy + global_pot_energy;

  EnergyData result;
  result.time             = current_time;
  result.kinetic_energy   = global_kin_energy;
  result.potential_energy = global_pot_energy;
  result.total_energy     = global_tot_energy;
  result.dissipation_rate = global_dissipation;
  result.energy_decay     = global_tot_energy - initial_energy;

  return result;
}

// ============================================================================
// WaveProblem — make_grid_and_dofs
// ============================================================================

template <int dim>
void
WaveProblem<dim>::make_grid_and_dofs()
{
  // Homogeneous Dirichlet BCs everywhere (boundary id = 0 by default)
  GridGenerator::hyper_cube(triangulation, -15, 15);
  triangulation.refine_global(n_global_refinements);

  // Two levels of adaptive pre-refinement towards the origin
  // where the initial Gaussian is localised.
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

  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  // Homogeneous Dirichlet on boundary id 0 (all faces of the hypercube)
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_partition;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values | update_quadrature_points;

  matrix_free_data.reinit(mapping,
                           dof_handler,
                           constraints,
                           QGaussLobatto<1>(fe_degree + 1),
                           additional_data);

  matrix_free_data.initialize_dof_vector(solution);
  old_solution.reinit(solution);
  old_old_solution.reinit(solution);
}

// ============================================================================
// WaveProblem — output_results
// ============================================================================

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
  data_out.write_vtu_with_pvtu_record("./", "solution", timestep_number, MPI_COMM_WORLD, 3);

  solution.zero_out_ghost_values();
}

// ============================================================================
// WaveProblem — log_energy
// ============================================================================

template <int dim>
void
WaveProblem<dim>::log_energy(const EnergyData &energy_data)
{
  energy_history.push_back(energy_data);
  pcout << "   [ENERGY] t="     << std::setw(7)  << std::setprecision(3) << energy_data.time
        << " | E_kin="           << std::setprecision(5) << std::setw(9) << energy_data.kinetic_energy
        << " | E_pot="           << std::setprecision(5) << std::setw(9) << energy_data.potential_energy
        << " | E_tot="           << std::setprecision(6) << std::setw(10) << energy_data.total_energy
        << " | DissRate="        << std::setprecision(4) << std::setw(8) << energy_data.dissipation_rate
        << " | dE="              << std::setprecision(4) << std::setw(8) << energy_data.energy_decay
        << std::endl;
}

// ============================================================================
// WaveProblem — export_energy_to_csv
// ============================================================================

template <int dim>
void
WaveProblem<dim>::export_energy_to_csv(const std::string &filename) const
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ofstream out_file(filename);
      out_file << std::setprecision(12);
      out_file << "time,kinetic_energy,potential_energy,total_energy,"
                  "dissipation_rate,energy_decay\n";
      for (const auto &entry : energy_history)
        out_file << entry.time             << ","
                 << entry.kinetic_energy   << ","
                 << entry.potential_energy << ","
                 << entry.total_energy     << ","
                 << entry.dissipation_rate << ","
                 << entry.energy_decay     << "\n";
      out_file.close();
      pcout << "   [ENERGY] History exported to: " << filename << std::endl;
    }
}

// ============================================================================
// WaveProblem — run
// ============================================================================

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

  pcout << "Wave speed  c     = " << c     << std::endl;
  pcout << "Damping     gamma = " << gamma << std::endl;
  pcout << "Forcing     f(x,t)= " << (forcing ? "active (user-defined)" : "none (f = 0)") << std::endl
        << std::endl;

  make_grid_and_dofs();

  // CFL-based time step
  double local_min_cell_diameter = std::numeric_limits<double>::max();
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      local_min_cell_diameter =
        std::min(local_min_cell_diameter, cell->diameter() / std::sqrt(dim));

  const double global_min_cell_diameter =
    Utilities::MPI::min(local_min_cell_diameter, MPI_COMM_WORLD);

  time_step = cfl_number * global_min_cell_diameter;
  time_step = (final_time - time) / (int((final_time - time) / time_step));
  pcout << "   Time step size: " << time_step
        << ", finest cell: " << global_min_cell_diameter << std::endl
        << std::endl;

  time = 0.0;

  // ----- Initial conditions -----
  LinearAlgebra::distributed::Vector<double> u_0, u_1;
  u_0.reinit(solution);
  u_1.reinit(solution);

  VectorTools::interpolate(mapping, dof_handler, *initial_displacement, u_0);
  VectorTools::interpolate(mapping, dof_handler, *initial_velocity,     u_1);

  constraints.distribute(u_0);
  constraints.distribute(u_1);

  solution = u_0;

  // ----- Build the wave operator (pass forcing function pointer) -----
  WaveOperation<dim, fe_degree> wave_op(matrix_free_data, time_step, c, gamma, forcing.get());

  // ----- PDE-accurate initial acceleration: a_0 = c^2*Lap(u_0) - gamma*u_1 + f(x,0) -----
  LinearAlgebra::distributed::Vector<double> a_0;
  matrix_free_data.initialize_dof_vector(a_0);
  wave_op.compute_initial_acceleration(a_0, solution, u_1);
  constraints.distribute(a_0);

  // ----- First step:  u^1 = u_0 + dt*u_1 + (dt^2/2)*a_0 -----
  LinearAlgebra::distributed::Vector<double> u_1st_step;
  matrix_free_data.initialize_dof_vector(u_1st_step);
  u_1st_step = solution;
  u_1st_step.add(time_step,                      u_1);
  u_1st_step.add(0.5 * time_step * time_step,    a_0);
  constraints.distribute(u_1st_step);

  // ----- Fictitious step: u^{-1} = u_0 - dt*u_1 + (dt^2/2)*a_0 -----
  old_solution = solution;
  old_solution.add(-time_step,                     u_1);
  old_solution.add(0.5 * time_step * time_step,    a_0);
  constraints.distribute(old_solution);

  // ----- Initial energy -----
  EnergyData e0 = wave_op.compute_energy(solution, old_solution, u_1st_step, 0.0, 0.0);
  initial_total_energy = e0.total_energy;
  e0.energy_decay      = 0.0;
  log_energy(e0);

  output_results(0);

  std::vector<LinearAlgebra::distributed::Vector<double> *>
    previous_solutions({&old_solution, &old_old_solution});

  unsigned int timestep_number = 1;

  Timer  timer;
  double wtime       = 0;
  double output_time = 0;

  for (time += time_step; time <= final_time;
       time += time_step, ++timestep_number)
    {
      timer.restart();
      old_old_solution.swap(old_solution); // u^{n-1}
      old_solution.swap(solution);         // u^n

      wave_op.apply(solution, previous_solutions, time - time_step); // u^{n+1}, f evaluated at t^n
      constraints.distribute(solution);
      wtime += timer.wall_time();

      // Log energy at the first step and every output_timestep_skip steps
      if (timestep_number % output_timestep_skip == 0 || timestep_number == 1)
        {
          const EnergyData e_step = wave_op.compute_energy(
            old_solution, old_old_solution, solution,
            time - time_step, initial_total_energy);
          log_energy(e_step);
        }

      timer.restart();
      if (timestep_number % output_timestep_skip == 0)
        output_results(timestep_number / output_timestep_skip);
      output_time += timer.wall_time();
    }

  timer.restart();
  output_results(timestep_number / output_timestep_skip + 1);
  output_time += timer.wall_time();

  export_energy_to_csv("energy_dissipation.csv");

  pcout << std::endl
        << "   Performed " << timestep_number << " time steps." << std::endl;
  pcout << "   Average wallclock time per time step: "
        << wtime / timestep_number << 's' << std::endl;
  pcout << "   Spent " << output_time << "s on output and "
        << wtime << "s on computations." << std::endl;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class WaveOperation<2, 4>;
template class WaveProblem<2>;

template class WaveOperation<3, 4>;
template class WaveProblem<3>;
