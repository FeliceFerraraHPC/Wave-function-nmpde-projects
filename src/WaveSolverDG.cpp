#include "WaveSolverDG.hpp"
#include <limits>

// ============================================================================
// WaveOperationDG -- SIPG matrix-free operator
// ============================================================================
template <int dim, int fe_degree>
WaveOperationDG<dim, fe_degree>::WaveOperationDG(
    const MatrixFree<dim, double> &data_in,
    const double time_step,
    const double cell_diameter,
    const double gamma,
    typename WaveSolverBase<dim>::BoundaryType boundary_type,
    bool non_homogeneous,
    const Function<dim> *exact_solution)
    : data_(data_in),
      time_step_(time_step),
      gamma_(gamma),
      delta_t_sqr_(make_vectorized_array(time_step * time_step)),
      h_inv_(1.0 / cell_diameter),
      boundary_type_(boundary_type),
      non_homogeneous_(non_homogeneous),
      exact_solution_(exact_solution)
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

  // See WaveOperationCG for the derivation; reduces to the plain lumped
  // inverse mass when gamma_ == 0.
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

// ---------------------------------------------------------------------------
// Cell integral: volume terms
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void WaveOperationDG<dim, fe_degree>::local_apply(
    const MatrixFree<dim, double> &data,
    LinearAlgebra::distributed::Vector<double> &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
{
  AssertDimension(src.size(), 2);
  FEEvaluation<dim, fe_degree> current(data), old(data);

  // Same damped leapfrog volume term as WaveOperationCG. Face terms
  // (SIPG fluxes) are unaffected by damping -- see local_apply_face() and
  // local_apply_boundary_face() below, which are unchanged.
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
      current.submit_value(2.0 * cur_val - old_coeff * old_val, q);
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
void WaveOperationDG<dim, fe_degree>::local_apply_face(
    const MatrixFree<dim, double> &data,
    LinearAlgebra::distributed::Vector<double> &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
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
      const auto normal = fe_int.get_normal_vector(q);
      const auto jump_u = fe_int.get_value(q) - fe_ext.get_value(q);
      const auto avg_grad =
          0.5 * (fe_int.get_gradient(q) + fe_ext.get_gradient(q));

      // SIPG fluxes
      const auto flux_val = avg_grad * normal - sigma * jump_u;
      const auto flux_grad = 0.5 * jump_u * normal;

      fe_int.submit_value(delta_t_sqr_ * flux_val, q);
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
// Boundary face integral: weak Dirichlet / Neumann via SIPG
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void WaveOperationDG<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, double> &data,
    LinearAlgebra::distributed::Vector<double> &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
{
  if (boundary_type_ == WaveSolverBase<dim>::BoundaryType::Neumann)
  {
    if (!non_homogeneous_ || exact_solution_ == nullptr)
      return; // Homogeneous Neumann: boundary flux is identically zero!

    FEFaceEvaluation<dim, fe_degree> fe_eval(data, true);
    for (unsigned int face = face_range.first; face < face_range.second; ++face)
    {
      fe_eval.reinit(face);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
      {
        const auto normal = fe_eval.get_normal_vector(q);
        const auto p_vec = fe_eval.quadrature_point(q);

        VectorizedArray<double> g_N;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
        {
          Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vec[d][v];
          Tensor<1, dim> grad_val = exact_solution_->gradient(p);
          Tensor<1, dim> n_val;
          for (unsigned int d = 0; d < dim; ++d)
            n_val[d] = normal[d][v];
          g_N[v] = grad_val * n_val;
        }

        fe_eval.submit_value(delta_t_sqr_ * g_N, q);
      }
      fe_eval.integrate(EvaluationFlags::values);
      fe_eval.distribute_local_to_global(dst);
    }
    return;
  }

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
      const auto normal = fe_eval.get_normal_vector(q);
      const auto u_val = fe_eval.get_value(q);
      const auto grad_u = fe_eval.get_gradient(q);

      VectorizedArray<double> u_diff = u_val;
      if (non_homogeneous_ && exact_solution_ != nullptr)
      {
        const auto p_vec = fe_eval.quadrature_point(q);
        VectorizedArray<double> g_D;
        for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
        {
          Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = p_vec[d][v];
          g_D[v] = exact_solution_->value(p);
        }
        u_diff -= g_D;
      }

      // Dirichlet penalty flux
      const auto flux_val = grad_u * normal - 2.0 * sigma * u_diff;
      const auto flux_grad = u_diff * normal;

      fe_eval.submit_value(delta_t_sqr_ * flux_val, q);
      fe_eval.submit_gradient(delta_t_sqr_ * flux_grad, q);
    }
    fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }
}

// ---------------------------------------------------------------------------
// apply() -- calls data_.loop() to run all three integrals
// ---------------------------------------------------------------------------
template <int dim, int fe_degree>
void WaveOperationDG<dim, fe_degree>::apply(
    LinearAlgebra::distributed::Vector<double> &dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src) const
{
  data_.loop(&WaveOperationDG<dim, fe_degree>::local_apply,
             &WaveOperationDG<dim, fe_degree>::local_apply_face,
             &WaveOperationDG<dim, fe_degree>::local_apply_boundary_face,
             this, dst, src, /*zero_dst_vector=*/true);
  dst.scale(inv_effective_mass_matrix_);
}

// ============================================================================
// WaveSolverDG -- constructor
// ============================================================================
template <int dim>
WaveSolverDG<dim>::WaveSolverDG(double cfl_number,
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
void WaveSolverDG<dim>::setup(const Triangulation<dim> &tria)
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
      update_normal_vectors | update_inverse_jacobians;
  additional_data.mapping_update_flags_boundary_faces =
      update_values | update_gradients | update_JxW_values |
      update_normal_vectors | update_inverse_jacobians |
      update_quadrature_points;

  matrix_free_data_.reinit(mapping_,
                           dof_handler_,
                           constraints_,
                           QGaussLobatto<1>(fe_degree + 1),
                           additional_data);

  matrix_free_data_.initialize_dof_vector(solution_);
  old_solution_.reinit(solution_);
  old_old_solution_.reinit(solution_);

  // Build lumped (GL diagonal) mass vector -- same procedure as CG.
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
  if (!user_time_step_)
    time_step_ = cfl_number_ * global_min;

  pcout_ << "   [MatFree-DG] DoFs: " << dof_handler_.n_dofs() << std::endl;
}

// ============================================================================
// set_initial_conditions()
// ============================================================================
template <int dim>
void WaveSolverDG<dim>::set_initial_conditions(const Function<dim> &u0,
                                               const Function<dim> &v0,
                                               const Function<dim> *u_prev)
{
  LinearAlgebra::distributed::Vector<double> u0_vec, v0_vec;
  u0_vec.reinit(solution_);
  v0_vec.reinit(solution_);

  VectorTools::interpolate(mapping_, dof_handler_, u0, u0_vec);
  VectorTools::interpolate(mapping_, dof_handler_, v0, v0_vec);

  constraints_.distribute(u0_vec);
  constraints_.distribute(v0_vec);

  solution_ = u0_vec;

  if (u_prev != nullptr)
  {
    VectorTools::interpolate(mapping_, dof_handler_, *u_prev, old_solution_);
    constraints_.distribute(old_solution_);
  }
  else
  {
    // u^{-1} = u0 - dt * v0  (leapfrog startup)
    old_solution_ = u0_vec;
    old_solution_.add(-time_step_, v0_vec);
  }
  time_ = 0.0;
}

// ============================================================================
// output_results()
// ============================================================================
template <int dim>
void WaveSolverDG<dim>::output_results(unsigned int timestep_number)
{
  constraints_.distribute(solution_);
  solution_.update_ghost_values();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(solution_, "solution");
  data_out.build_patches(mapping_);

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);
  data_out.write_vtu_with_pvtu_record(
      get_output_dir(), "solution", timestep_number, MPI_COMM_WORLD, 3);
}

// ============================================================================
// run() -- main time loop
// ============================================================================
template <int dim>
double
WaveSolverDG<dim>::run(double T, bool write_output)
{
  const double local_min =
      tria_ptr_->last()->diameter() / std::sqrt(double(dim));
  const double global_min =
      -Utilities::MPI::max(-local_min, MPI_COMM_WORLD);

  const double dt_old = time_step_;

  // If time_step_ has not been set, use CFL condition.
  if (time_step_ <= 0.0)
    time_step_ = cfl_number_ * global_min;

  // Round to integer number of steps to land exactly at T.
  const unsigned int n_steps =
      std::max(1u, static_cast<unsigned int>(std::round((T - time_) / time_step_)));
  time_step_ = (T - time_) / n_steps;

  pcout_ << "   [MatFree-DG] dt = " << time_step_
         << ", finest cell = " << global_min << std::endl;

  // Adjust old_solution_ for the adjusted dt (leapfrog startup correction)
  // only if dt changed and user didn't explicitly specify custom time step.
  if (std::abs(time_step_ - dt_old) > 1e-14 && dt_old > 0.0 && !user_time_step_)
    old_solution_.sadd(time_step_ / dt_old, 1.0 - time_step_ / dt_old, solution_);

  std::vector<LinearAlgebra::distributed::Vector<double> *> prev_solutions(
      {&old_solution_, &old_old_solution_});

  // h_inv for SIPG penalty = 1 / global_min_cell_diameter
  WaveOperationDG<dim, fe_degree> wave_op(
      matrix_free_data_, time_step_, global_min, gamma_, this->boundary_type_,
      this->non_homogeneous_, this->exact_solution_);

  unsigned int timestep_number = 1;
  Timer timer;
  double wtime = 0.0;

  if (write_output)
    output_results(0);

  for (unsigned int step = 1; step <= n_steps; ++step, ++timestep_number)
  {
    time_ = step * time_step_;
    timer.restart();

    wave_op.set_current_time((step - 1) * time_step_);
    if (this->exact_solution_ != nullptr)
      const_cast<Function<dim> *>(this->exact_solution_)->set_time((step - 1) * time_step_);

    old_old_solution_.swap(old_solution_);
    old_solution_.swap(solution_);
    wave_op.apply(solution_, prev_solutions);
    constraints_.distribute(solution_);

    wtime += timer.wall_time();

    // Record energy at the first step (t=0) and periodically thereafter.
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
WaveSolverDG<dim>::compute_error(VectorTools::NormType norm_type,
                                 const Function<dim> &exact_solution) const
{
  solution_.update_ghost_values();
  Vector<double> error_per_cell(tria_ptr_->n_active_cells());

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
WaveSolverDG<dim>::find_peak(const Point<dim> &center,
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
WaveSolverDG<dim>::compute_energy() const
{
  // After the swap sequence in run():
  //   solution_         = u^{n+1}   (just computed)
  //   old_solution_     = u^n
  //   old_old_solution_ = u^{n-1}
  LinearAlgebra::distributed::Vector<double> loc_sol(dof_handler_.locally_owned_dofs(), locally_relevant_dofs_, MPI_COMM_WORLD);
  LinearAlgebra::distributed::Vector<double> loc_old_sol(dof_handler_.locally_owned_dofs(), locally_relevant_dofs_, MPI_COMM_WORLD);
  LinearAlgebra::distributed::Vector<double> loc_old_old_sol(dof_handler_.locally_owned_dofs(), locally_relevant_dofs_, MPI_COMM_WORLD);

  loc_sol = solution_;
  loc_old_sol = old_solution_;
  loc_old_old_sol = old_old_solution_;

  loc_sol.update_ghost_values();
  loc_old_sol.update_ghost_values();
  loc_old_old_sol.update_ghost_values();

  // -----------------------------------------------------------------------
  // SIPG penalty parameter (same formula as WaveOperationDG).
  // Compute h = min cell diameter once for the whole energy call.
  // -----------------------------------------------------------------------
  double local_min_h = std::numeric_limits<double>::max();
  for (const auto &cell : dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
      local_min_h = std::min(local_min_h, cell->diameter());

  const double global_min_h =
      -Utilities::MPI::max(-local_min_h, MPI_COMM_WORLD);
  const double penalty_factor =
      1.5 * (fe_degree + 1) * (fe_degree + dim) / double(dim);
  const double sigma = penalty_factor / global_min_h;

  // -----------------------------------------------------------------------
  // Volume quadrature -- natural (collocated) and staggered potential.
  // -----------------------------------------------------------------------
  const QGauss<dim> quadrature(fe_degree + 1);
  FEValues<dim> fe_values(mapping_, fe_, quadrature,
                          update_values | update_gradients |
                              update_JxW_values);

  const unsigned int n_q = quadrature.size();
  std::vector<double> u_next(n_q), u_prev(n_q);
  std::vector<Tensor<1, dim>> grad_u_curr(n_q);
  std::vector<Tensor<1, dim>> grad_u_next(n_q); // u^{n+1}

  double local_kin = 0.0;
  double local_pot = 0.0;
  double local_pot_stag = 0.0; // staggered volume potential (partial a_h)
  const double inv_2dt = 1.0 / (2.0 * time_step_);

  for (const auto &cell : dof_handler_.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values.get_function_values(loc_sol, u_next);
      fe_values.get_function_values(loc_old_old_sol, u_prev);
      fe_values.get_function_gradients(loc_old_sol, grad_u_curr);
      fe_values.get_function_gradients(loc_sol, grad_u_next);

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const double v_nat = (u_next[q] - u_prev[q]) * inv_2dt;
        const double JxW = fe_values.JxW(q);

        // Natural (volume-only, unchanged from before)
        local_kin += 0.5 * v_nat * v_nat * JxW;
        local_pot += 0.5 * (grad_u_curr[q] * grad_u_curr[q]) * JxW;

        // Staggered potential volume part: grad u^n * grad u^{n+1} dx
        local_pot_stag += 0.5 * (grad_u_curr[q] * grad_u_next[q]) * JxW;
      }
    }

  // -----------------------------------------------------------------------
  // Face quadrature -- staggered potential SIPG face terms.
  //
  // The full SIPG bilinear form is:
  //   a_h(u, w) = sum_K grad u * grad w dx
  //             - sum_{f in F_int} int_f ({{grad u * n}}[w] + {{grad w * n}}[u]) ds
  //             + sum_{f in F_int} sigma int_f [u][w] ds
  //             - sum_{f in F_bnd} int_f (grad u * n*w + grad w * n*u) ds
  //             + sum_{f in F_bnd} 2*sigma int_f u*w ds          (Dirichlet u=0)
  //
  // Here we compute the face contribution to 0.5 * a_h(u^n, u^{n+1}).
  //
  // Parallel note: each interior face f is shared by cells K+ and K-.
  // We process f exactly once by choosing the cell with smaller
  // subdomain_id (or, within the same rank, smaller CellId).
  // -----------------------------------------------------------------------
  const QGauss<dim - 1> face_quad(fe_degree + 1);
  const unsigned int n_fq = face_quad.size();

  FEFaceValues<dim> fv_curr(mapping_, fe_, face_quad,
                            update_values | update_gradients |
                                update_JxW_values | update_normal_vectors);
  FEFaceValues<dim> fv_next(mapping_, fe_, face_quad,
                            update_values | update_gradients);
  FEFaceValues<dim> fv_nbr_curr(mapping_, fe_, face_quad,
                                update_values | update_gradients);
  FEFaceValues<dim> fv_nbr_next(mapping_, fe_, face_quad,
                                update_values | update_gradients);

  std::vector<double> uc(n_fq), un(n_fq), uc_nbr(n_fq), un_nbr(n_fq);
  std::vector<Tensor<1, dim>> gc(n_fq), gn(n_fq), gc_nbr(n_fq), gn_nbr(n_fq);

  double local_face_stag = 0.0;

  for (const auto &cell : dof_handler_.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (unsigned int f = 0; f < cell->n_faces(); ++f)
    {
      if (cell->at_boundary(f))
      {
        if (this->boundary_type_ == WaveSolverBase<dim>::BoundaryType::Neumann)
          continue;

        fv_curr.reinit(cell, f);
        fv_next.reinit(cell, f);

        fv_curr.get_function_values(loc_old_sol, uc);
        fv_next.get_function_values(loc_sol, un);
        fv_curr.get_function_gradients(loc_old_sol, gc);
        fv_next.get_function_gradients(loc_sol, gn);

        for (unsigned int q = 0; q < n_fq; ++q)
        {
          const auto &nrm = fv_curr.normal_vector(q);
          const double JxW = fv_curr.JxW(q);
          const double contrib =
              -(gc[q] * nrm) * un[q] - (gn[q] * nrm) * uc[q] + 2.0 * sigma * uc[q] * un[q];
          local_face_stag += 0.5 * contrib * JxW;
        }
      }
      else
      {
        // ----------------------------------------------------------
        // Interior face: process once per face using subdomain_id.
        // ----------------------------------------------------------
        const auto neighbor = cell->neighbor(f);
        const bool skip =
            (cell->subdomain_id() > neighbor->subdomain_id()) ||
            (cell->subdomain_id() == neighbor->subdomain_id() &&
             neighbor->id() < cell->id());
        if (skip)
          continue;

        const unsigned int nbr_f = cell->neighbor_of_neighbor(f);
        fv_curr.reinit(cell, f);
        fv_next.reinit(cell, f);
        fv_nbr_curr.reinit(neighbor, nbr_f);
        fv_nbr_next.reinit(neighbor, nbr_f);

        fv_curr.get_function_values(loc_old_sol, uc);
        fv_next.get_function_values(loc_sol, un);
        fv_curr.get_function_gradients(loc_old_sol, gc);
        fv_next.get_function_gradients(loc_sol, gn);

        fv_nbr_curr.get_function_values(loc_old_sol, uc_nbr);
        fv_nbr_next.get_function_values(loc_sol, un_nbr);
        fv_nbr_curr.get_function_gradients(loc_old_sol, gc_nbr);
        fv_nbr_next.get_function_gradients(loc_sol, gn_nbr);

        for (unsigned int q = 0; q < n_fq; ++q)
        {
          const auto &nrm = fv_curr.normal_vector(q);
          const double JxW = fv_curr.JxW(q);

          const double jmp_c = uc[q] - uc_nbr[q]; // [u^n]
          const double jmp_n = un[q] - un_nbr[q]; // [u^{n+1}]

          const Tensor<1, dim> avg_gc = 0.5 * (gc[q] + gc_nbr[q]);
          const Tensor<1, dim> avg_gn = 0.5 * (gn[q] + gn_nbr[q]);

          const double contrib =
              -(avg_gc * nrm) * jmp_n - (avg_gn * nrm) * jmp_c + sigma * jmp_c * jmp_n;
          local_face_stag += 0.5 * contrib * JxW;
        }
      }
    }
  }

  solution_.zero_out_ghost_values();
  old_solution_.zero_out_ghost_values();
  old_old_solution_.zero_out_ghost_values();

  const double kin = Utilities::MPI::sum(local_kin, MPI_COMM_WORLD);
  const double pot = Utilities::MPI::sum(local_pot, MPI_COMM_WORLD);
  const double pot_stag_vol = Utilities::MPI::sum(local_pot_stag, MPI_COMM_WORLD);
  const double pot_stag_face = Utilities::MPI::sum(local_face_stag, MPI_COMM_WORLD);
  const double pot_stag = pot_stag_vol + pot_stag_face;

  // -----------------------------------------------------------------------
  // Staggered kinetic: exact lumped-mass inner product (same as CG).
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

  if (initial_total_energy_ < 0.0)
    initial_total_energy_ = kin + pot;
  if (initial_stag_total_energy_ < 0.0)
    initial_stag_total_energy_ = kin_stag + pot_stag;

  EnergyData e;
  e.time = time_ - time_step_;

  // Natural energy (volume-only, unchanged from previous)
  e.kinetic_energy = kin;
  e.potential_energy = pot;
  e.total_energy = kin + pot;
  e.dissipation_rate = 2.0 * gamma_ * kin;
  e.energy_decay = e.total_energy - initial_total_energy_;

  // Staggered energy (full SIPG a_h for potential)
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