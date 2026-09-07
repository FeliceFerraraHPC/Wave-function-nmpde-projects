#include "WaveSolverDG.hpp"

// ============================================================================
// WaveOperationDG — SIPG matrix-free operator
// ============================================================================

template <int dim, int fe_degree>
WaveOperationDG<dim, fe_degree>::WaveOperationDG(
  const MatrixFree<dim, double> &data_in,
  const double                   time_step,
  const double                   cell_diameter)
  : data_(data_in)
  , delta_t_sqr_(make_vectorized_array(time_step * time_step))
  , h_inv_(1.0 / cell_diameter)
{
  data_.initialize_dof_vector(inv_mass_matrix_);

  FEEvaluation<dim, fe_degree> fe_eval(data_);
  for (unsigned int cell = 0; cell < data_.n_cell_batches(); ++cell)
    {
      fe_eval.reinit(cell);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        fe_eval.submit_value(make_vectorized_array(1.0), q);
      fe_eval.integrate(EvaluationFlags::values);
      fe_eval.distribute_local_to_global(inv_mass_matrix_);
    }

  inv_mass_matrix_.compress(VectorOperation::add);
  for (unsigned int k = 0; k < inv_mass_matrix_.locally_owned_size(); ++k)
    {
      if (inv_mass_matrix_.local_element(k) > 1e-15)
        inv_mass_matrix_.local_element(k) =
          1.0 / inv_mass_matrix_.local_element(k);
      else
        inv_mass_matrix_.local_element(k) = 1.0;
    }
}

// ---------------------------------------------------------------------------
// Cell integral: volume terms (same as CG)
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void
WaveOperationDG<dim, fe_degree>::local_apply(
  const MatrixFree<dim, double>                                   &data,
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const std::pair<unsigned int, unsigned int>                     &cell_range) const
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
          const VectorizedArray<double> cur_val = current.get_value(q);
          const VectorizedArray<double> old_val = old.get_value(q);

          VectorizedArray<double> f_val = make_vectorized_array(0.0);
          if (current_forcing_function_ != nullptr)
            {
              for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
                {
                  Point<dim> p;
                  for (unsigned int d = 0; d < dim; ++d)
                    p[d] = current.quadrature_point(q)[d][v];
                  f_val[v] = current_forcing_function_->value(p);
                }
            }

          current.submit_value(2.0 * cur_val - old_val + delta_t_sqr_ * f_val, q);
          current.submit_gradient(-delta_t_sqr_ * current.get_gradient(q), q);
        }

      current.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      current.distribute_local_to_global(dst);
    }
}

// ---------------------------------------------------------------------------
// Interior face integral: SIPG numerical fluxes
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void
WaveOperationDG<dim, fe_degree>::local_apply_face(
  const MatrixFree<dim, double>                                   &data,
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const std::pair<unsigned int, unsigned int>                     &face_range) const
{
  FEFaceEvaluation<dim, fe_degree> fe_ext(data, false);
  FEFaceEvaluation<dim, fe_degree> fe_int(data, true);

  const double penalty_factor =
    1.5 * (fe_degree + 1) * (fe_degree + dim) / dim;
  const double sigma = penalty_factor * h_inv_;

  for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      fe_int.reinit(face);
      fe_ext.reinit(face);

      fe_int.read_dof_values(*src[0]);
      fe_ext.read_dof_values(*src[0]);

      fe_int.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      fe_ext.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (const unsigned int q : fe_int.quadrature_point_indices())
        {
          const auto normal    = fe_int.get_normal_vector(q);
          const auto jump_u    = fe_int.get_value(q) - fe_ext.get_value(q);
          const auto avg_grad  =
            0.5 * (fe_int.get_gradient(q) + fe_ext.get_gradient(q));

          // SIPG fluxes
          const auto flux_val  = avg_grad * normal - sigma * jump_u;
          const auto flux_grad = 0.5 * jump_u * normal;

          fe_int.submit_value( delta_t_sqr_ * flux_val, q);
          fe_ext.submit_value(-delta_t_sqr_ * flux_val, q);

          fe_int.submit_gradient(delta_t_sqr_ * flux_grad, q);
          fe_ext.submit_gradient(delta_t_sqr_ * flux_grad, q);
        }

      fe_int.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      fe_ext.integrate(EvaluationFlags::values | EvaluationFlags::gradients);

      fe_int.distribute_local_to_global(dst);
      fe_ext.distribute_local_to_global(dst);
    }
}

// ---------------------------------------------------------------------------
// Boundary face integral: weak Dirichlet u = 0 via SIPG
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void
WaveOperationDG<dim, fe_degree>::local_apply_boundary_face(
  const MatrixFree<dim, double>                                   &data,
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const std::pair<unsigned int, unsigned int>                     &face_range) const
{
  FEFaceEvaluation<dim, fe_degree> fe_eval(data, true);

  const double penalty_factor =
    1.5 * (fe_degree + 1) * (fe_degree + dim) / dim;
  const double sigma = penalty_factor * h_inv_;

  for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(*src[0]);
      fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (const unsigned int q : fe_eval.quadrature_point_indices())
        {
          const auto normal   = fe_eval.get_normal_vector(q);
          const auto u_val    = fe_eval.get_value(q);
          const auto grad_u   = fe_eval.get_gradient(q);

          // Dirichlet u = 0 penalty flux
          const auto flux_val  = grad_u * normal - 2.0 * sigma * u_val;
          const auto flux_grad = u_val * normal;

          fe_eval.submit_value(delta_t_sqr_ * flux_val, q);
          fe_eval.submit_gradient(delta_t_sqr_ * flux_grad, q);
        }

      fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }
}

// ---------------------------------------------------------------------------
// apply() — calls data_.loop() to run all three integrals
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void
WaveOperationDG<dim, fe_degree>::apply(
  LinearAlgebra::distributed::Vector<double>                      &dst,
  const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
  const Function<dim>                                             *forcing_function,
  double                                                           current_time) const
{
  current_forcing_function_ = forcing_function;
  current_time_             = current_time;
  
  if (current_forcing_function_ != nullptr)
    const_cast<Function<dim> *>(current_forcing_function_)->set_time(current_time_);

  data_.loop(&WaveOperationDG<dim, fe_degree>::local_apply,
             &WaveOperationDG<dim, fe_degree>::local_apply_face,
             &WaveOperationDG<dim, fe_degree>::local_apply_boundary_face,
             this, dst, src, /*zero_dst_vector=*/true);
  dst.scale(inv_mass_matrix_);
}

// ============================================================================
// WaveSolverDG — constructor
// ============================================================================

template <int dim>
WaveSolverDG<dim>::WaveSolverDG(double cfl_number, unsigned int output_skip)
  : pcout_(std::cout,
           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , fe_(QGaussLobatto<1>(fe_degree + 1))
  , dof_handler_(dummy_tria_)
  , cfl_number_(cfl_number)
  , output_timestep_skip_(output_skip)
{}

// ============================================================================
// setup()
// ============================================================================

template <int dim>
void
WaveSolverDG<dim>::setup(const Triangulation<dim> &tria)
{
  tria_ptr_ = &tria;
  dof_handler_.reinit(tria);
  dof_handler_.distribute_dofs(fe_);

  // DG: no hanging node constraints; Dirichlet enforced weakly via SIPG.
  locally_relevant_dofs_ = DoFTools::extract_locally_relevant_dofs(dof_handler_);
  constraints_.clear();
  constraints_.reinit(locally_relevant_dofs_);
  constraints_.close();

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::
      partition_partition;

  // Enable face data structures for SIPG.
  additional_data.mapping_update_flags_inner_faces =
    update_values | update_gradients | update_JxW_values |
    update_normal_vectors | update_inverse_jacobians | update_quadrature_points;
  additional_data.mapping_update_flags_boundary_faces =
    update_values | update_gradients | update_JxW_values |
    update_normal_vectors | update_inverse_jacobians | update_quadrature_points;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values | update_quadrature_points;

  matrix_free_data_.reinit(mapping_,
                            dof_handler_,
                            constraints_,
                            QGaussLobatto<1>(fe_degree + 1),
                            additional_data);

  matrix_free_data_.initialize_dof_vector(solution_);
  old_solution_.reinit(solution_);
  old_old_solution_.reinit(solution_);

  pcout_ << "   [MatFree-DG] DoFs: " << dof_handler_.n_dofs() << std::endl;
}

// ============================================================================
// set_forcing_function()
// ============================================================================

template <int dim>
void
WaveSolverDG<dim>::set_forcing_function(const Function<dim> *f)
{
  forcing_function_ptr_ = f;
}

// ============================================================================
// set_initial_conditions()
// ============================================================================

template <int dim>
void
WaveSolverDG<dim>::set_initial_conditions(const Function<dim> &u0,
                                          const Function<dim> &v0)
{
  LinearAlgebra::distributed::Vector<double> u0_vec, v0_vec;
  u0_vec.reinit(solution_);
  v0_vec.reinit(solution_);

  VectorTools::interpolate(mapping_, dof_handler_, u0, u0_vec);
  VectorTools::interpolate(mapping_, dof_handler_, v0, v0_vec);
  constraints_.distribute(u0_vec);
  constraints_.distribute(v0_vec);

  solution_     = u0_vec;
  old_solution_ = u0_vec;
  old_solution_.add(-time_step_, v0_vec);

  time_ = 0.0;
}

// ============================================================================
// output_results()
// ============================================================================

template <int dim>
void
WaveSolverDG<dim>::output_results(unsigned int timestep_number)
{
  constraints_.distribute(solution_);
  solution_.update_ghost_values();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(solution_, "solution");
  data_out.build_patches(mapping_);
  data_out.write_vtu_with_pvtu_record(
    "./", "dg_solution", timestep_number, MPI_COMM_WORLD, 3);

  solution_.zero_out_ghost_values();
}

// ============================================================================
// run()
// ============================================================================
template <int dim>
double
WaveSolverDG<dim>::run(double T, bool write_output, unsigned int output_frequency)
{
  Assert(tria_ptr_ != nullptr, ExcNotInitialized());

  // Derive time step from empirical CFL condition.
  const double local_min = tria_ptr_->last()->diameter() / std::sqrt(double(dim));
  const double global_min = -Utilities::MPI::max(-local_min, MPI_COMM_WORLD);
  time_step_ = cfl_number_ * global_min;

  // Round to integer number of steps.
  time_step_ = (T - time_) / static_cast<int>((T - time_) / time_step_);

  pcout_ << "   [MatFree-DG] dt = " << time_step_
         << ", finest cell = " << global_min << std::endl;

  std::vector<LinearAlgebra::distributed::Vector<double> *> prev_solutions(
    {&old_solution_, &old_old_solution_});

  // h_inv for SIPG penalty = 1 / global_min_cell_diameter
  WaveOperationDG<dim, fe_degree> wave_op(
    matrix_free_data_, time_step_, global_min);

  unsigned int timestep_number = 1;
  Timer        timer;
  double       wtime = 0.0;

  if (write_output)
    output_results(0);

  for (time_ += time_step_; time_ <= T + 1e-12; time_ += time_step_, ++timestep_number)
    {
      timer.restart();
      old_old_solution_.swap(old_solution_);
      old_solution_.swap(solution_);
      wave_op.apply(solution_, prev_solutions, forcing_function_ptr_, time_ - time_step_);
      constraints_.distribute(solution_);
      wtime += timer.wall_time();

      if (write_output && (timestep_number % output_frequency == 0 || time_ >= T - 1e-12))
        output_results(timestep_number / output_frequency);
    }

  if (write_output)
    output_results(timestep_number / output_timestep_skip_ + 1);

  return wtime;
}

// ============================================================================
// compute_error()
// ============================================================================

template <int dim>
double
WaveSolverDG<dim>::compute_error(VectorTools::NormType norm_type,
                                 const Function<dim>  &exact_solution) const
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
// n_dofs()
// ============================================================================

template <int dim>
unsigned int
WaveSolverDG<dim>::n_dofs() const
{
  return static_cast<unsigned int>(dof_handler_.n_dofs());
}

// ============================================================================
// Explicit instantiations
// ============================================================================

template class WaveOperationDG<2, 4>;
template class WaveSolverDG<2>;

template class WaveOperationDG<3, 4>;
template class WaveSolverDG<3>;
