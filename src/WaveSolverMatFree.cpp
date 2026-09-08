#include "WaveSolverMatFree.hpp"

// ============================================================================
// WaveOperationCG -- matrix-free cell loop for continuous Galerkin leap-frog
// ============================================================================
template <int dim, int fe_degree>
WaveOperationCG<dim, fe_degree>::WaveOperationCG(
    const MatrixFree<dim, double> &data_in,
    const double time_step,
    const double gamma)
    : data_(data_in), time_step_(time_step), gamma_(gamma), delta_t_sqr_(make_vectorized_array(time_step * time_step))
{
  data_.initialize_dof_vector(inv_effective_mass_matrix_);
  FEEvaluation<dim, fe_degree> fe_eval(data_);
  for (unsigned int cell = 0; cell < data_.n_cell_batches(); ++cell)
  {
    fe_eval.reinit(cell);
    for (const unsigned int q : fe_eval.quadrature_point_indices())
      fe_eval.submit_value(make_vectorized_array(1.0), q);
    fe_eval.integrate(EvaluationFlags::values);
    fe_eval.distribute_local_to_global(inv_effective_mass_matrix_);
  }
  inv_effective_mass_matrix_.compress(VectorOperation::add);

  // Fold the damping-induced effective-mass scaling (1/(1+0.5*dt*gamma)) into
  // the lumped inverse mass once, up front. With gamma_ == 0 this is exactly
  // the plain lumped inverse mass, so the undamped case is unaffected.
  const double effective_scale = 1.0 / (1.0 + 0.5 * time_step_ * gamma_);
  for (unsigned int k = 0; k < inv_effective_mass_matrix_.locally_owned_size(); ++k)
  {
    if (inv_effective_mass_matrix_.local_element(k) > 1e-15)
      inv_effective_mass_matrix_.local_element(k) =
          effective_scale / inv_effective_mass_matrix_.local_element(k);
    else
      inv_effective_mass_matrix_.local_element(k) = 1.0;
  }
}

template <int dim, int fe_degree>
void WaveOperationCG<dim, fe_degree>::local_apply(
    const MatrixFree<dim, double> &data,
    LinearAlgebra::distributed::Vector<double> &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
{
  AssertDimension(src.size(), 2);
  FEEvaluation<dim, fe_degree> current(data), old(data);

  // Damped leapfrog:
  //   (1+0.5*dt*gamma)*u^{n+1} = 2*u^n - (1-0.5*dt*gamma)*u^{n-1} + dt^2*Delta u^n
  // The (1+0.5*dt*gamma) factor is applied afterwards via
  // inv_effective_mass_matrix_ in apply(); here we only need the
  // (1-0.5*dt*gamma) factor on the old (u^{n-1}) term.
  const VectorizedArray<double> old_coeff =
      make_vectorized_array(1.0 - 0.5 * time_step_ * gamma_);

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
      const VectorizedArray<double> cur_val = current.get_value(q);
      const VectorizedArray<double> old_val = old.get_value(q);

      // Leap-frog: u^{n+1} = 2*u^n - (1-0.5*dt*gamma)*u^{n-1} + dt^2*Delta u^n
      current.submit_value(2.0 * cur_val - old_coeff * old_val, q);
      current.submit_gradient(-delta_t_sqr_ * current.get_gradient(q), q);
    }
    current.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    current.distribute_local_to_global(dst);
  }
}

template <int dim, int fe_degree>
void WaveOperationCG<dim, fe_degree>::apply(
    LinearAlgebra::distributed::Vector<double> &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const
{
  data_.cell_loop(&WaveOperationCG<dim, fe_degree>::local_apply,
                  this,
                  dst,
                  src,
                  /*zero_dst_vector=*/true);
  dst.scale(inv_effective_mass_matrix_);
}

// ============================================================================
// WaveSolverMatFree -- constructor
// ============================================================================
template <int dim>
WaveSolverMatFree<dim>::WaveSolverMatFree(double cfl_number,
                                          unsigned int output_skip,
                                          double gamma)
    : pcout_(std::cout,
             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      fe_(QGaussLobatto<1>(fe_degree + 1)), dof_handler_(dummy_tria_), cfl_number_(cfl_number), output_timestep_skip_(output_skip), gamma_(gamma)
{
}

// ============================================================================
// setup()
// ============================================================================
template <int dim>
void WaveSolverMatFree<dim>::setup(const Triangulation<dim> &tria)
{
  tria_ptr_ = &tria;
  dof_handler_.reinit(tria);
  dof_handler_.distribute_dofs(fe_);

  locally_relevant_dofs_ = DoFTools::extract_locally_relevant_dofs(dof_handler_);
  constraints_.clear();
  constraints_.reinit(locally_relevant_dofs_);
  DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);
  VectorTools::interpolate_boundary_values(mapping_,
                                           dof_handler_,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints_);
  constraints_.close();

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
      MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::
          partition_partition;

  matrix_free_data_.reinit(mapping_,
                           dof_handler_,
                           constraints_,
                           QGaussLobatto<1>(fe_degree + 1),
                           additional_data);

  matrix_free_data_.initialize_dof_vector(solution_);
  old_solution_.reinit(solution_);
  old_old_solution_.reinit(solution_);

  // Build the lumped (Gauss-Lobatto diagonal) mass vector once.
  // With GL nodes and quadrature the mass matrix is diagonal: M_ii = w_i.
  // We store the raw weights (not inverted) so we can use them as a
  // dot-product weight in the staggered kinetic energy E_kin_stag = 0.5*v^T M v.
  lumped_mass_.reinit(solution_);
  {
    FEEvaluation<dim, fe_degree> fe_eval(matrix_free_data_);
    for (unsigned int cell = 0; cell < matrix_free_data_.n_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        fe_eval.submit_value(make_vectorized_array(1.0), q);
      fe_eval.integrate(EvaluationFlags::values);
      fe_eval.distribute_local_to_global(lumped_mass_);
    }
    lumped_mass_.compress(VectorOperation::add);
  }

  // Initialize CFL-based time_step_ so that set_initial_conditions()
  // uses the proper dt for the leapfrog startup u^{-1} = u0 - dt * v0.
  const double local_min = tria_ptr_->last()->diameter() / std::sqrt(double(dim));
  const double global_min =
      -Utilities::MPI::max(-local_min, MPI_COMM_WORLD);
  time_step_ = cfl_number_ * global_min;

  pcout_ << "   [MatFree-CG] DoFs: " << dof_handler_.n_dofs() << std::endl;
}

// ============================================================================
// set_initial_conditions()
// ============================================================================
template <int dim>
void WaveSolverMatFree<dim>::set_initial_conditions(const Function<dim> &u0,
                                                    const Function<dim> &v0)
{
  LinearAlgebra::distributed::Vector<double> u0_vec, v0_vec;
  u0_vec.reinit(solution_);
  v0_vec.reinit(solution_);

  VectorTools::interpolate(mapping_, dof_handler_, u0, u0_vec);
  VectorTools::interpolate(mapping_, dof_handler_, v0, v0_vec);

  constraints_.distribute(u0_vec);
  constraints_.distribute(v0_vec);

  solution_ = u0_vec;
  // u^{-1} = u0 - dt * v0  (leapfrog startup)
  old_solution_ = u0_vec;
  old_solution_.add(-time_step_, v0_vec);
  time_ = 0.0;
}

// ============================================================================
// output_results()
// ============================================================================
template <int dim>
void WaveSolverMatFree<dim>::output_results(unsigned int timestep_number)
{
  constraints_.distribute(solution_);
  solution_.update_ghost_values();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(solution_, "solution");
  data_out.build_patches(mapping_);
  data_out.write_vtu_with_pvtu_record(
      "./", "cg_solution", timestep_number, MPI_COMM_WORLD, 3);

  solution_.zero_out_ghost_values();
}

// ============================================================================
// run()
// ============================================================================
template <int dim>
double
WaveSolverMatFree<dim>::run(double T, bool write_output)
{
  Assert(tria_ptr_ != nullptr, ExcNotInitialized());

  // Derive time step from CFL condition on the mesh.
  const double local_min = tria_ptr_->last()->diameter() / std::sqrt(double(dim));
  const double global_min =
      -Utilities::MPI::max(-local_min, MPI_COMM_WORLD);
  const double dt_old = time_step_;

  if (time_step_ <= 0.0)
    time_step_ = cfl_number_ * global_min;

  // Round to integer number of steps to land exactly at T.
  const unsigned int n_steps =
      std::max(1u, static_cast<unsigned int>(std::round((T - time_) / time_step_)));
  time_step_ = (T - time_) / n_steps;

  pcout_ << "   [MatFree-CG] dt = " << time_step_
         << ", finest cell = " << global_min << std::endl;

  // Adjust old_solution_ for the adjusted dt (leapfrog startup correction).
  // old_solution_ was set to u0 - dt_old * v0 in set_initial_conditions.
  if (std::abs(time_step_ - dt_old) > 1e-14 && dt_old > 0.0)
  {
    old_solution_.sadd(time_step_ / dt_old, 1.0 - time_step_ / dt_old, solution_);
  }

  std::vector<LinearAlgebra::distributed::Vector<double> *> prev_solutions(
      {&old_solution_, &old_old_solution_});

  WaveOperationCG<dim, fe_degree> wave_op(matrix_free_data_, time_step_, gamma_);

  unsigned int timestep_number = 1;
  Timer timer;
  double wtime = 0.0;

  if (write_output)
    output_results(0);

  for (unsigned int step = 1; step <= n_steps; ++step, ++timestep_number)
  {
    time_ = step * time_step_;
    timer.restart();

    old_old_solution_.swap(old_solution_);
    old_solution_.swap(solution_);
    wave_op.apply(solution_, prev_solutions);
    constraints_.distribute(solution_);

    wtime += timer.wall_time();

    // Record energy at the first step (t=0, since old_solution_ == u^0
    // right after this first apply()) and periodically thereafter.
    if (timestep_number == 1 || timestep_number % output_timestep_skip_ == 0)
      energy_history_.push_back(compute_energy());

    if (write_output && timestep_number % output_timestep_skip_ == 0)
      output_results(timestep_number / output_timestep_skip_);
  }

  time_ = T;
  if (write_output)
    output_results(timestep_number / output_timestep_skip_ + 1);

  return wtime;
}

// ============================================================================
// compute_error()
// ============================================================================
template <int dim>
double
WaveSolverMatFree<dim>::compute_error(VectorTools::NormType norm_type,
                                      const Function<dim> &exact_solution) const
{
  solution_.update_ghost_values();
  Vector<float> error_per_cell(tria_ptr_->n_active_cells());

  VectorTools::integrate_difference(mapping_,
                                    dof_handler_,
                                    solution_,
                                    exact_solution,
                                    error_per_cell,
                                    QGauss<dim>(fe_degree + 1),
                                    norm_type);

  const double error =
      VectorTools::compute_global_error(*tria_ptr_, error_per_cell, norm_type);
  solution_.zero_out_ghost_values();
  return error;
}

// ============================================================================
// find_peak()
// ============================================================================
template <int dim>
std::pair<Point<dim>, double>
WaveSolverMatFree<dim>::find_peak(const Point<dim> &center,
                                  double x_span,
                                  unsigned int n_pts) const
{
  solution_.update_ghost_values();
  auto eval = [&](const Point<dim> &p) -> double
  {
    double val = -1e30;
    try
    {
      val = VectorTools::point_value(mapping_, dof_handler_, solution_, p);
    }
    catch (...)
    {
      val = -1e30;
    }
    return Utilities::MPI::max(val, MPI_COMM_WORLD);
  };

  auto res = locate_peak_1d<dim>(eval, center, x_span, n_pts);
  solution_.zero_out_ghost_values();
  return res;
}

// ============================================================================
// compute_energy()
// ============================================================================
template <int dim>
EnergyData
WaveSolverMatFree<dim>::compute_energy() const
{
  // After the swap sequence in run():
  //   solution_         = u^{n+1}   (just computed)
  //   old_solution_     = u^n
  //   old_old_solution_ = u^{n-1}
  solution_.update_ghost_values();
  old_solution_.update_ghost_values();
  old_old_solution_.update_ghost_values();

  // -----------------------------------------------------------------------
  // Common FEValues setup (consistent Gauss quadrature for both energies).
  // -----------------------------------------------------------------------
  const QGauss<dim> quadrature(fe_degree + 1);
  FEValues<dim> fe_values(mapping_, fe_, quadrature,
                          update_values | update_gradients |
                              update_JxW_values);

  const unsigned int n_q = quadrature.size();
  std::vector<double> u_next(n_q);              // u^{n+1}
  std::vector<double> u_prev(n_q);              // u^{n-1}
  std::vector<Tensor<1, dim>> grad_u_curr(n_q); // grad u^n   (for natural E_pot and stag E_pot)
  std::vector<Tensor<1, dim>> grad_u_next(n_q); // grad u^{n+1} (for stag E_pot cross-term)

  double local_kin = 0.0;      // natural kinetic
  double local_pot = 0.0;      // natural potential
  double local_pot_stag = 0.0; // staggered potential: grad u^n * grad u^{n+1} dx
  const double inv_2dt = 1.0 / (2.0 * time_step_);

  for (const auto &cell : dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(solution_, u_next);             // u^{n+1}
      fe_values.get_function_values(old_old_solution_, u_prev);     // u^{n-1}
      fe_values.get_function_gradients(old_solution_, grad_u_curr); // grad u^n
      fe_values.get_function_gradients(solution_, grad_u_next);     // grad u^{n+1}

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const double v_nat = (u_next[q] - u_prev[q]) * inv_2dt;
        const double JxW = fe_values.JxW(q);

        // --- Natural energy (collocated, O(dt^2) drift) ---
        local_kin += 0.5 * v_nat * v_nat * JxW;
        local_pot += 0.5 * (grad_u_curr[q] * grad_u_curr[q]) * JxW;

        // --- Staggered potential: 0.5 * a_h(u^n, u^{n+1}) -- volume term ---
        // For CG the bilinear form a_h is purely volumetric (no face terms).
        local_pot_stag += 0.5 * (grad_u_curr[q] * grad_u_next[q]) * JxW;
      }
    }

  solution_.zero_out_ghost_values();
  old_solution_.zero_out_ghost_values();
  old_old_solution_.zero_out_ghost_values();

  const double kin = Utilities::MPI::sum(local_kin, MPI_COMM_WORLD);
  const double pot = Utilities::MPI::sum(local_pot, MPI_COMM_WORLD);
  const double pot_stag = Utilities::MPI::sum(local_pot_stag, MPI_COMM_WORLD);

  // -----------------------------------------------------------------------
  // Staggered kinetic: EXACT lumped-mass inner product.
  //
  //   E_kin_stag = 0.5 * sum_i  m_i * ((u^{n+1}_i - u^n_i) / dt)^2
  //
  // where m_i = lumped_mass_[i] are the Gauss-Lobatto quadrature weights
  // (the exact diagonal entries of the GL mass matrix).  This is the
  // numerically exact discrete conserved quantity for the leapfrog scheme.
  // No FEValues loop is needed: it reduces to a single pass over locally
  // owned DOFs -- O(N) and cache-friendly.
  // -----------------------------------------------------------------------
  double local_kin_stag = 0.0;
  const double inv_dt_sq = 1.0 / (time_step_ * time_step_);
  for (unsigned int i = 0; i < solution_.locally_owned_size(); ++i)
  {
    const double diff = solution_.local_element(i) - old_solution_.local_element(i);
    local_kin_stag += lumped_mass_.local_element(i) * diff * diff;
  }
  const double kin_stag =
      0.5 * inv_dt_sq * Utilities::MPI::sum(local_kin_stag, MPI_COMM_WORLD);

  // -----------------------------------------------------------------------
  // Lazy-initialise reference energies (first call = t ~ 0 snapshot).
  // -----------------------------------------------------------------------
  if (initial_total_energy_ < 0.0)
    initial_total_energy_ = kin + pot;
  if (initial_stag_total_energy_ < 0.0)
    initial_stag_total_energy_ = kin_stag + pot_stag;

  // -----------------------------------------------------------------------
  // Assemble and return the EnergyData snapshot.
  // -----------------------------------------------------------------------
  EnergyData e;
  e.time = time_ - time_step_; // time level of old_solution_ (u^n)

  // Natural energy
  e.kinetic_energy = kin;
  e.potential_energy = pot;
  e.total_energy = kin + pot;
  e.dissipation_rate = 2.0 * gamma_ * kin; // D = gamma * ||u_t||^2 = 2*gamma*E_kin
  e.energy_decay = e.total_energy - initial_total_energy_;

  // Staggered energy
  e.stag_kinetic_energy = kin_stag;
  e.stag_potential_energy = pot_stag;
  e.stag_total_energy = kin_stag + pot_stag;
  e.stag_energy_decay = e.stag_total_energy - initial_stag_total_energy_;

  return e;
}

// ============================================================================
// n_dofs()
// ============================================================================
template <int dim>
unsigned int
WaveSolverMatFree<dim>::n_dofs() const
{
  return static_cast<unsigned int>(dof_handler_.n_dofs());
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class WaveOperationCG<2, 4>;
template class WaveSolverMatFree<2>;
template class WaveOperationCG<3, 4>;
template class WaveSolverMatFree<3>;