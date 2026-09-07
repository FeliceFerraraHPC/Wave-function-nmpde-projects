#include "WaveSolverTheta.hpp"
#include "WaveFunctions.hpp"

#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

// ============================================================================
// Constructor
// ============================================================================

template <int dim>
WaveSolverTheta<dim>::WaveSolverTheta(unsigned int fe_degree, double theta)
  : pcout_(std::cout,
           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , fe_degree_(fe_degree)
  , theta_(theta)
  , fe_(fe_degree)
  , dummy_tria_()
  , dof_handler_(dummy_tria_)
{
  // dof_handler_ is re-attached to the external tria in setup() via reinit().
}

// ============================================================================
// setup()  — called by the benchmark driver with the shared triangulation
// ============================================================================

template <int dim>
void
WaveSolverTheta<dim>::setup(const Triangulation<dim> &tria)
{
  tria_ptr_ = &tria;
  // Re-create the DoFHandler attached to the given triangulation.
  dof_handler_.reinit(tria);
  setup_system();
}

template <int dim>
void
WaveSolverTheta<dim>::setup_system()
{
  Assert(tria_ptr_ != nullptr, ExcNotInitialized());

  dof_handler_.distribute_dofs(fe_);

  locally_owned_dofs_    = dof_handler_.locally_owned_dofs();
  locally_relevant_dofs_ = DoFTools::extract_locally_relevant_dofs(dof_handler_);

  // Build constraints (hanging nodes + Dirichlet BCs).
  constraints_.clear();
  constraints_.reinit(locally_relevant_dofs_);
  DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);
  VectorTools::interpolate_boundary_values(
    dof_handler_, 0, Functions::ZeroFunction<dim>(), constraints_);
  constraints_.close();

  // Distributed sparsity pattern.
  DynamicSparsityPattern dsp(locally_relevant_dofs_);
  DoFTools::make_sparsity_pattern(dof_handler_, dsp, constraints_,
                                  /*keep_constrained_dofs=*/false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs_,
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs_);

  // Initialize Trilinos matrices.
  mass_matrix_.reinit(locally_owned_dofs_, locally_owned_dofs_, dsp,
                      MPI_COMM_WORLD);
  laplace_matrix_.reinit(locally_owned_dofs_, locally_owned_dofs_, dsp,
                         MPI_COMM_WORLD);
  matrix_u_.reinit(locally_owned_dofs_, locally_owned_dofs_, dsp,
                   MPI_COMM_WORLD);
  matrix_v_.reinit(locally_owned_dofs_, locally_owned_dofs_, dsp,
                   MPI_COMM_WORLD);

  // Assemble constant matrices.
  assemble_matrices();

  // Initialize solution vectors.
  solution_u_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
  solution_v_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
  old_solution_u_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
  old_solution_v_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);
  system_rhs_.reinit(locally_owned_dofs_, MPI_COMM_WORLD);

  pcout_ << "   [Theta] DoFs: " << dof_handler_.n_dofs() << std::endl;
}

// ============================================================================
// assemble_matrices()  — build M and A (constant across all time steps)
// ============================================================================

template <int dim>
void
WaveSolverTheta<dim>::assemble_matrices()
{
  // MatrixCreator uses WorkStream internally (TBB parallel when available).
  // Use the 3-argument overload: TrilinosWrappers::SparseMatrix doesn't support
  // the nullptr-coefficient overload due to template deduction constraints.
  // Boundary values are enforced per time step via MatrixTools::apply_boundary_values.
  MatrixCreator::create_mass_matrix(
    dof_handler_, QGauss<dim>(fe_.degree + 1), mass_matrix_,
    (const Function<dim> *)nullptr, constraints_);
  MatrixCreator::create_laplace_matrix(
    dof_handler_, QGauss<dim>(fe_.degree + 1), laplace_matrix_,
    (const Function<dim> *)nullptr, constraints_);
}

// ============================================================================
// set_initial_conditions()
// ============================================================================

template <int dim>
void
WaveSolverTheta<dim>::set_initial_conditions(const Function<dim> &u0,
                                             const Function<dim> &v0)
{
  // Use a ghosted vector for projection, then copy to owned.
  TrilinosWrappers::MPI::Vector ghosted(locally_owned_dofs_,
                                        locally_relevant_dofs_,
                                        MPI_COMM_WORLD);

  VectorTools::project(dof_handler_, constraints_,
                       QGauss<dim>(fe_.degree + 1), u0, ghosted);
  old_solution_u_ = ghosted;

  VectorTools::project(dof_handler_, constraints_,
                       QGauss<dim>(fe_.degree + 1), v0, ghosted);
  old_solution_v_ = ghosted;

  time_           = 0.0;
  timestep_number_ = 0;
}

// ============================================================================
// set_forcing_function()
// ============================================================================

template <int dim>
void
WaveSolverTheta<dim>::set_forcing_function(const Function<dim> *f)
{
  forcing_function_ptr_ = f;
}

// ============================================================================
// solve_u() / solve_v()
// ============================================================================

template <int dim>
void
WaveSolverTheta<dim>::solve_u()
{
  SolverControl solver_control(2000, 1e-8 * system_rhs_.l2_norm());
  TrilinosWrappers::SolverCG cg(solver_control);
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(matrix_u_);
  cg.solve(matrix_u_, solution_u_, system_rhs_, preconditioner);
  constraints_.distribute(solution_u_);
}

template <int dim>
void
WaveSolverTheta<dim>::solve_v()
{
  SolverControl solver_control(2000, 1e-8 * system_rhs_.l2_norm());
  TrilinosWrappers::SolverCG cg(solver_control);
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(matrix_v_);
  cg.solve(matrix_v_, solution_v_, system_rhs_, preconditioner);
  constraints_.distribute(solution_v_);
}

// ============================================================================
// output_results()
// ============================================================================

template <int dim>
void
WaveSolverTheta<dim>::output_results(unsigned int step_number) const
{
  // Build ghosted copies for output.
  TrilinosWrappers::MPI::Vector ghosted_u(locally_owned_dofs_,
                                          locally_relevant_dofs_,
                                          MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector ghosted_v(locally_owned_dofs_,
                                          locally_relevant_dofs_,
                                          MPI_COMM_WORLD);
  ghosted_u = solution_u_;
  ghosted_v = solution_v_;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_);
  data_out.add_data_vector(ghosted_u, "U");
  data_out.add_data_vector(ghosted_v, "V");
  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(flags);

  data_out.write_vtu_with_pvtu_record(
    "./", "theta_solution", step_number, MPI_COMM_WORLD, 3);
}

// ============================================================================
// run()  — main time loop
// ============================================================================

template <int dim>
double
WaveSolverTheta<dim>::run(double T, bool write_output, unsigned int output_frequency)
{
  // Choose time step proportional to mesh size (CFL-like) and snap it to T.
  const double h = GridTools::minimal_cell_diameter(*tria_ptr_);
  time_step_      = h / 2.0;
  
  unsigned int num_steps = static_cast<unsigned int>(std::ceil(T / time_step_));
  time_step_ = T / num_steps;
  
  time_           = time_step_;
  timestep_number_ = 1;

  // Work vectors (owned only — no ghosts needed for solve).
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs_, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector forcing_terms(locally_owned_dofs_, MPI_COMM_WORLD);

  ZeroForcing<dim> zero_f;
  const Function<dim> &f =
    (forcing_function_ptr_ != nullptr) ? *forcing_function_ptr_ : zero_f;

  Timer timer;
  double wtime = 0.0;

  for (; time_ <= T + 1e-12; time_ += time_step_, ++timestep_number_)
    {
      timer.restart();

      // --- Build RHS for u ---
      mass_matrix_.vmult(system_rhs_, old_solution_u_);

      mass_matrix_.vmult(tmp, old_solution_v_);
      system_rhs_.add(time_step_, tmp);

      laplace_matrix_.vmult(tmp, old_solution_u_);
      system_rhs_.add(-theta_ * (1.0 - theta_) * time_step_ * time_step_, tmp);

      // Forcing terms at current and previous time.
      const_cast<Function<dim>&>(f).set_time(time_);
      VectorTools::create_right_hand_side(dof_handler_,
                                          QGauss<dim>(fe_.degree + 1),
                                          f, forcing_terms);
      forcing_terms *= theta_ * time_step_;

      const_cast<Function<dim>&>(f).set_time(time_ - time_step_);
      VectorTools::create_right_hand_side(dof_handler_,
                                          QGauss<dim>(fe_.degree + 1),
                                          f, tmp);
      forcing_terms.add((1.0 - theta_) * time_step_, tmp);
      system_rhs_.add(theta_ * time_step_, forcing_terms);

      // Apply Dirichlet BC for u.
      {
        Functions::ZeroFunction<dim> zero_bc;
        std::map<types::global_dof_index, double> bv;
        VectorTools::interpolate_boundary_values(dof_handler_, 0, zero_bc, bv);
        matrix_u_.copy_from(mass_matrix_);
        matrix_u_.add(theta_ * theta_ * time_step_ * time_step_, laplace_matrix_);
        MatrixTools::apply_boundary_values(bv, matrix_u_, solution_u_, system_rhs_);
      }
      solve_u();

      // --- Build RHS for v ---
      laplace_matrix_.vmult(system_rhs_, solution_u_);
      system_rhs_ *= -theta_ * time_step_;

      mass_matrix_.vmult(tmp, old_solution_v_);
      system_rhs_ += tmp;

      laplace_matrix_.vmult(tmp, old_solution_u_);
      system_rhs_.add(-(1.0 - theta_) * time_step_, tmp);

      system_rhs_ += forcing_terms;

      // Apply Dirichlet BC for v.
      {
        Functions::ZeroFunction<dim> zero_bc;
        std::map<types::global_dof_index, double> bv;
        VectorTools::interpolate_boundary_values(dof_handler_, 0, zero_bc, bv);
        matrix_v_.copy_from(mass_matrix_);
        MatrixTools::apply_boundary_values(bv, matrix_v_, solution_v_, system_rhs_);
      }
      solve_v();

      wtime += timer.wall_time();

      old_solution_u_ = solution_u_;
      old_solution_v_ = solution_v_;

      if (write_output && (timestep_number_ % output_frequency == 0 || time_ >= T))
        output_results(timestep_number_);
    }

  return wtime;
}

// ============================================================================
// compute_error()
// ============================================================================

template <int dim>
double
WaveSolverTheta<dim>::compute_error(VectorTools::NormType norm_type,
                                    const Function<dim>  &exact_solution) const
{
  // Build a ghosted copy of solution_u_ for integrate_difference.
  TrilinosWrappers::MPI::Vector ghosted(locally_owned_dofs_,
                                        locally_relevant_dofs_,
                                        MPI_COMM_WORLD);
  ghosted = solution_u_;

  Vector<float> error_per_cell(tria_ptr_->n_active_cells());
  VectorTools::integrate_difference(dof_handler_, ghosted, exact_solution,
                                    error_per_cell,
                                    QGauss<dim>(fe_.degree + 2),
                                    norm_type);
  return VectorTools::compute_global_error(*tria_ptr_, error_per_cell, norm_type);
}

// ============================================================================
// n_dofs()
// ============================================================================

template <int dim>
unsigned int
WaveSolverTheta<dim>::n_dofs() const
{
  return static_cast<unsigned int>(dof_handler_.n_dofs());
}



// ============================================================================
// Explicit instantiations
// ============================================================================

template class WaveSolverTheta<2>;
template class WaveSolverTheta<3>;
