#include "WaveSolverTheta.hpp"
#include "WaveFunctions.hpp"

#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
WaveSolverTheta<dim>::WaveSolverTheta(unsigned int fe_degree,
                                      double theta,
                                      double gamma)
    : pcout_(std::cout,
             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      fe_degree_(fe_degree), theta_(theta), gamma_(gamma), fe_(fe_degree), dummy_tria_(), dof_handler_(dummy_tria_) // dof_handler_ is re-attached to the external tria in setup() via reinit().
{
}

// ============================================================================
// setup() -- called by the benchmark driver with the shared triangulation
// ============================================================================
template <int dim>
void WaveSolverTheta<dim>::setup(const Triangulation<dim> &tria)
{
  tria_ptr_ = &tria;
  // Re-create the DoFHandler attached to the given triangulation.
  dof_handler_.reinit(tria);
  setup_system();
}

template <int dim>
void WaveSolverTheta<dim>::setup_system()
{
  Assert(tria_ptr_ != nullptr, ExcNotInitialized());

  dof_handler_.distribute_dofs(fe_);
  locally_owned_dofs_ = dof_handler_.locally_owned_dofs();
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
  DoFTools::make_sparsity_pattern(dof_handler_,
                                  dsp,
                                  constraints_,
                                  /*keep_constrained_dofs=*/false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs_,
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs_);

  // Initialize Trilinos matrices.
  mass_matrix_.reinit(locally_owned_dofs_,
                      locally_owned_dofs_,
                      dsp,
                      MPI_COMM_WORLD);
  laplace_matrix_.reinit(locally_owned_dofs_,
                         locally_owned_dofs_,
                         dsp,
                         MPI_COMM_WORLD);
  matrix_u_.reinit(locally_owned_dofs_,
                   locally_owned_dofs_,
                   dsp,
                   MPI_COMM_WORLD);
  matrix_v_.reinit(locally_owned_dofs_,
                   locally_owned_dofs_,
                   dsp,
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
// assemble_matrices() -- build M and A (constant across all time steps)
// ============================================================================
template <int dim>
void WaveSolverTheta<dim>::assemble_matrices()
{
  // MatrixCreator uses WorkStream internally (TBB parallel when available).
  // Use the 3-argument overload: TrilinosWrappers::SparseMatrix doesn't support
  // the nullptr-coefficient overload due to template deduction constraints.
  // Boundary values are enforced per time step via MatrixTools::apply_boundary_values.
  MatrixCreator::create_mass_matrix(
      dof_handler_,
      QGauss<dim>(fe_.degree + 1),
      mass_matrix_,
      (const Function<dim> *)nullptr,
      constraints_);

  MatrixCreator::create_laplace_matrix(
      dof_handler_,
      QGauss<dim>(fe_.degree + 1),
      laplace_matrix_,
      (const Function<dim> *)nullptr,
      constraints_);
}

// ============================================================================
// set_initial_conditions()
// ============================================================================
template <int dim>
void WaveSolverTheta<dim>::set_initial_conditions(const Function<dim> &u0,
                                                  const Function<dim> &v0)
{
  // Use a ghosted vector for projection, then copy to owned.
  TrilinosWrappers::MPI::Vector ghosted(locally_owned_dofs_,
                                        locally_relevant_dofs_,
                                        MPI_COMM_WORLD);

  VectorTools::project(dof_handler_,
                       constraints_,
                       QGauss<dim>(fe_.degree + 1),
                       u0,
                       ghosted);
  old_solution_u_ = ghosted;

  VectorTools::project(dof_handler_,
                       constraints_,
                       QGauss<dim>(fe_.degree + 1),
                       v0,
                       ghosted);
  old_solution_v_ = ghosted;

  time_ = 0.0;
  timestep_number_ = 0;
}

// ============================================================================
// solve_u() / solve_v()
// ============================================================================
template <int dim>
void WaveSolverTheta<dim>::solve_u()
{
  SolverControl solver_control(2000, 1e-8 * system_rhs_.l2_norm());
  TrilinosWrappers::SolverCG cg(solver_control);
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(matrix_u_);
  cg.solve(matrix_u_, solution_u_, system_rhs_, preconditioner);
  constraints_.distribute(solution_u_);
}

template <int dim>
void WaveSolverTheta<dim>::solve_v()
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
void WaveSolverTheta<dim>::output_results(unsigned int step_number) const
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
// run() -- main time loop
// ============================================================================
template <int dim>
double
WaveSolverTheta<dim>::run(double T, bool write_output)
{
  // Choose time step: use user-specified time_step_ if set, else CFL-like h/2.
  if (!user_time_step_ || time_step_ <= 0.0)
  {
    const double h = GridTools::minimal_cell_diameter(*tria_ptr_);
    time_step_ = h / 2.0;
  }

  const unsigned int n_steps =
      std::max(1u, static_cast<unsigned int>(std::round((T - time_) / time_step_)));
  time_step_ = (T - time_) / n_steps;
  time_ = time_step_;
  timestep_number_ = 1;

  // Work vectors (owned only -- no ghosts needed for solve).
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs_, MPI_COMM_WORLD);
  TrilinosWrappers::MPI::Vector forcing_terms(locally_owned_dofs_, MPI_COMM_WORLD);

  ZeroForcing<dim> zero_f;
  const Function<dim> &f =
      (forcing_function_ptr_ != nullptr) ? *forcing_function_ptr_ : zero_f;

  Timer timer;
  double wtime = 0.0;

  // Damped theta-scheme coefficients (constant for the whole run, since
  // theta_, time_step_, gamma_ don't change between steps). With gamma_ = 0
  // these reduce to a = b = 1, recovering the original undamped formulas
  // exactly.
  //   a = 1 + theta*dt*gamma
  //   b = 1 - (1-theta)*dt*gamma
  const double a = 1.0 + theta_ * time_step_ * gamma_;
  const double b = 1.0 - (1.0 - theta_) * time_step_ * gamma_;

  // Record the t=0 energy before the loop overwrites solution_u_/solution_v_.
  // (time_ has already been advanced to time_step_ above, so stamp this
  // snapshot with 0.0 explicitly rather than trusting time_.)
  solution_u_ = old_solution_u_;
  solution_v_ = old_solution_v_;
  EnergyData e0 = compute_energy();
  e0.time = 0.0;
  energy_history_.push_back(e0);

  // Assemble constant system matrices for u and v once before the time loop.
  // Both are constant across all time steps (time_step_, theta_, gamma_ are constant).
  // Homogeneous Dirichlet boundary conditions are already symmetrically condensed
  // via constraints_ in mass_matrix_ and laplace_matrix_, preserving SPD symmetry.
  matrix_u_.copy_from(mass_matrix_);
  matrix_u_.add(theta_ * theta_ * time_step_ * time_step_ / a, laplace_matrix_);
  matrix_v_.copy_from(mass_matrix_);
  matrix_v_ *= a; // (1 + theta*dt*gamma) * M

  for (unsigned int step = 1; step <= n_steps;
       ++step, ++timestep_number_, time_ += time_step_)
  {
    timer.restart();

    // --- Build RHS for u ---
    mass_matrix_.vmult(system_rhs_, old_solution_u_);
    mass_matrix_.vmult(tmp, old_solution_v_);
    system_rhs_.add(time_step_ * (1.0 - theta_) + time_step_ * theta_ * b / a,
                    tmp);
    laplace_matrix_.vmult(tmp, old_solution_u_);
    system_rhs_.add(-theta_ * (1.0 - theta_) * time_step_ * time_step_ / a,
                    tmp);

    // Forcing terms at current and previous time.
    const_cast<Function<dim> &>(f).set_time(time_);
    VectorTools::create_right_hand_side(dof_handler_,
                                        QGauss<dim>(fe_.degree + 1),
                                        f,
                                        forcing_terms);
    forcing_terms *= theta_ * time_step_;

    const_cast<Function<dim> &>(f).set_time(time_ - time_step_);
    VectorTools::create_right_hand_side(dof_handler_,
                                        QGauss<dim>(fe_.degree + 1),
                                        f,
                                        tmp);
    forcing_terms.add((1.0 - theta_) * time_step_, tmp);
    system_rhs_.add(theta_ * time_step_ / a, forcing_terms);

    constraints_.set_zero(system_rhs_);
    solve_u();

    // --- Build RHS for v ---
    laplace_matrix_.vmult(system_rhs_, solution_u_);
    system_rhs_ *= -theta_ * time_step_;
    mass_matrix_.vmult(tmp, old_solution_v_);
    system_rhs_.add(b, tmp);
    laplace_matrix_.vmult(tmp, old_solution_u_);
    system_rhs_.add(-(1.0 - theta_) * time_step_, tmp);
    system_rhs_ += forcing_terms;

    constraints_.set_zero(system_rhs_);
    solve_v();

    wtime += timer.wall_time();

    // Record energy BEFORE updating old_solution_u_/v_ so that
    // compute_energy() sees:
    //   solution_u_     = u^n  (just solved)
    //   old_solution_u_ = u^{n-1}  (needed for staggered backward-diff kinetic)
    energy_history_.push_back(compute_energy());

    old_solution_u_ = solution_u_;
    old_solution_v_ = solution_v_;

    if (write_output && timestep_number_ % 10 == 0)
      output_results(timestep_number_);
  }

  time_ = T;
  return wtime;
}

// ============================================================================
// compute_error()
// ============================================================================
template <int dim>
double
WaveSolverTheta<dim>::compute_error(VectorTools::NormType norm_type,
                                    const Function<dim> &exact_solution) const
{
  // Build a ghosted copy of solution_u_ for integrate_difference.
  TrilinosWrappers::MPI::Vector ghosted(locally_owned_dofs_,
                                        locally_relevant_dofs_,
                                        MPI_COMM_WORLD);
  ghosted = solution_u_;

  Vector<float> error_per_cell(tria_ptr_->n_active_cells());
  VectorTools::integrate_difference(dof_handler_,
                                    ghosted,
                                    exact_solution,
                                    error_per_cell,
                                    QGauss<dim>(fe_.degree + 2),
                                    norm_type);

  return VectorTools::compute_global_error(*tria_ptr_, error_per_cell, norm_type);
}

// ============================================================================
// find_peak()
// ============================================================================
template <int dim>
std::pair<Point<dim>, double>
WaveSolverTheta<dim>::find_peak(const Point<dim> &center,
                                double x_span,
                                unsigned int n_pts) const
{
  TrilinosWrappers::MPI::Vector ghosted(locally_owned_dofs_,
                                        locally_relevant_dofs_,
                                        MPI_COMM_WORLD);
  ghosted = solution_u_;

  MappingQ1<dim> mapping;
  auto eval = [&](const Point<dim> &p) -> double
  {
    double val = -1e30;
    try
    {
      val = VectorTools::point_value(mapping, dof_handler_, ghosted, p);
    }
    catch (...)
    {
      val = -1e30;
    }
    return Utilities::MPI::max(val, MPI_COMM_WORLD);
  };

  return locate_peak_1d<dim>(eval, center, x_span, n_pts);
}

// ============================================================================
// compute_energy()
// ============================================================================
template <int dim>
EnergyData
WaveSolverTheta<dim>::compute_energy() const
{
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs_, MPI_COMM_WORLD);

  // -----------------------------------------------------------------------
  // 1. EXACT (natural) energy -- uses explicitly-tracked velocity v^n.
  //    E_kin = 0.5 * v^T M v
  //    E_pot = 0.5 * u^T A u
  // -----------------------------------------------------------------------
  mass_matrix_.vmult(tmp, solution_v_);
  const double kin = 0.5 * (solution_v_ * tmp);

  laplace_matrix_.vmult(tmp, solution_u_);
  const double pot = 0.5 * (solution_u_ * tmp);

  // -----------------------------------------------------------------------
  // 2. STAGGERED energy -- backward-difference approximation to the half-step
  //    leapfrog observable, so all three solvers expose a comparable metric.
  //
  //    v_{n-1/2}  = (u^n - u^{n-1}) / dt
  //    E_kin_stag = 0.5 * v_{n-1/2}^T M v_{n-1/2}
  //    E_pot_stag = 0.5 * (u^{n-1})^T A u^n  =  0.5 * a(u^{n-1}, u^n)
  //
  //    Called BEFORE `old_solution_u_ = solution_u_` in run(), so:
  //      solution_u_     = u^n   (just computed this step)
  //      old_solution_u_ = u^{n-1}
  // -----------------------------------------------------------------------
  TrilinosWrappers::MPI::Vector diff(locally_owned_dofs_, MPI_COMM_WORLD);
  diff = solution_u_;
  diff -= old_solution_u_; // diff = u^n - u^{n-1}
  diff /= time_step_;      // diff = v_{n-1/2}

  mass_matrix_.vmult(tmp, diff);
  const double kin_stag = 0.5 * (diff * tmp);

  // a(u^{n-1}, u^n) = old_u^T A u^n
  laplace_matrix_.vmult(tmp, solution_u_);
  const double pot_stag = 0.5 * (old_solution_u_ * tmp);

  // -----------------------------------------------------------------------
  // Lazy-initialise reference energies.
  // -----------------------------------------------------------------------
  if (initial_total_energy_ < 0.0)
    initial_total_energy_ = kin + pot;
  if (initial_stag_total_energy_ < 0.0)
    initial_stag_total_energy_ = kin_stag + pot_stag;

  EnergyData e;
  e.time = time_;

  // Natural (exact) energy
  e.kinetic_energy = kin;
  e.potential_energy = pot;
  e.total_energy = kin + pot;
  e.dissipation_rate = 2.0 * gamma_ * kin; // D = gamma * v^T M v = 2*gamma*E_kin
  e.energy_decay = e.total_energy - initial_total_energy_;

  // Staggered (backward-diff) energy
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
WaveSolverTheta<dim>::n_dofs() const
{
  return static_cast<unsigned int>(dof_handler_.n_dofs());
}

// ============================================================================
// run_convergence_study() -- optional manufactured-solution test
// ============================================================================
template <int dim>
void WaveSolverTheta<dim>::run_convergence_study(
    const std::vector<unsigned int> &refinement_levels,
    double final_time,
    unsigned int fe_degree,
    double theta)
{
  // The manufactured solution is only defined for dim==2.
  // Use if constexpr so the dim==3 instantiation compiles without error.
  if constexpr (dim == 2)
  {
    ConvergenceTable table;
    std::ofstream csv("convergence_theta.csv");
    csv << "h,eL2(u),eH1(u),eL2(v),eH1(v)\n";

    for (const unsigned int N_el : refinement_levels)
    {
      Triangulation<dim> mesh;
      GridGenerator::hyper_cube(mesh, 0.0, 1.0);
      mesh.refine_global(N_el);

      const double h = 1.0 / std::pow(2.0, N_el);
      const double dt = h / 2.0;

      WaveSolverTheta<dim> solver(fe_degree, theta);
      solver.setup(mesh);

      // Exact solution at t=0 is zero for both u and v.
      Functions::ZeroFunction<dim> zero;
      solver.set_initial_conditions(zero, zero);
      solver.time_step_ = dt;
      solver.time_ = dt;

      // Attach manufactured forcing function.
      ManufacturedRHS<dim> rhs_func;
      solver.forcing_function_ptr_ = &rhs_func;

      solver.run(final_time, /*write_output=*/false);

      ManufacturedSolutionU<dim> sol_u_T(final_time);
      const double eL2u = solver.compute_error(VectorTools::L2_norm, sol_u_T);
      const double eH1u = solver.compute_error(VectorTools::H1_norm, sol_u_T);

      // v-error reporting requires access to solution_v_ -- placeholder for now.
      const double eL2v = 0.0;
      const double eH1v = 0.0;

      table.add_value("h", h);
      table.add_value("L2(u)", eL2u);
      table.add_value("H1(u)", eH1u);
      table.add_value("L2(v)", eL2v);
      table.add_value("H1(v)", eH1v);

      csv << h << "," << eL2u << "," << eH1u << ","
          << eL2v << "," << eH1v << "\n";
    }

    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2(u)", true);
    table.set_scientific("H1(u)", true);
    table.set_scientific("L2(v)", true);
    table.set_scientific("H1(v)", true);
    table.write_text(std::cout);
  }
  else
  {
    std::cerr << "run_convergence_study: not implemented for dim="
              << dim << std::endl;
  }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class WaveSolverTheta<2>;
template class WaveSolverTheta<3>;