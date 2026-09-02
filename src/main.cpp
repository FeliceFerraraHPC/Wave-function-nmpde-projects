#include "WaveEquation.hpp"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  try
    {
      // Spatial dimension: change to 2 for a 2-D run.
      const unsigned int dim = 3;

      // ------------------------------------------------------------------
      // Physical parameters for:
      //
      //   u_tt - c^2 * Lap(u) + gamma * u_t = 0
      //
      // on [-15, 15]^dim with homogeneous Dirichlet BCs and the Gaussian
      // initial displacement u_0(x) = exp(-|x|^2 / 2).
      //
      // Set gamma = 0.0  => energy-conserving (E_tot should stay constant)
      // Set gamma > 0.0  => damped wave, E_tot decays exponentially
      // ------------------------------------------------------------------

      const double final_time = 30.0; // end time T
      const double c          = 1.0;  // wave speed
      const double gamma      = 0.0;  // damping coefficient (0 => conservative)

      // Forcing term f(x, t).
      // ForcingTerm<dim> defaults to f = 0 (homogeneous, energy-conserving).
      // Replace with a custom Function<dim> subclass to add external forcing.
      auto forcing = std::make_shared<ForcingTerm<dim>>();

      // Initial conditions (can be replaced with any Function<dim>)
      auto u_0 = std::make_shared<InitialDisplacement<dim>>();
      auto u_1 = std::make_shared<InitialVelocity<dim>>();

      WaveProblem<dim> wave_problem(final_time, c, gamma, forcing, u_0, u_1);
      wave_problem.run();

      // Energy history is also accessible programmatically:
      //   const auto &hist = wave_problem.get_energy_history();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
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
