#include <iostream>
#include <deal.II/base/convergence_table.h>
#include "WaveEquation.hpp"

static constexpr unsigned int dim = 2;

// Exact solution for position u.
// u(x,y,t) = sin(2π*x) * sin(4π*y) * cos(ω*t), where ω = 2π*√5
class ExactSolutionU : public Function<dim>
{
public:
  // Constructor.
  ExactSolutionU()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    double t = this->get_time();

    return t * t * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    Tensor<1, dim> result;

    result[0] = t * t * M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]);
    result[1] = t * t * M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]);

    return result;
  }
};


// Exact solution for velocity v.
// v(x,y,t) = ∂u/∂t = -ω * sin(2π*x) * sin(4π*y) * sin(ω*t), where ω = 2π*√5
class ExactSolutionV : public Function<dim>
{
public:
  // Constructor.
  ExactSolutionV()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    double t = this->get_time();

    return 2 * t * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]);
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    const double t = this->get_time();
    Tensor<1, dim> result;

    result[0] = 2.0 * t * M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]);
    result[1] = 2.0 * t * M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]);

    return result;
  }
};

void test_convergence();

int main()
{
    try
    {
        // WaveEquation<2> wave_equation_solver;
        // wave_equation_solver.run();

        // Test convergence
        test_convergence();
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

void test_convergence() {
ConvergenceTable table;

  const std::vector<unsigned int> N_el_values = {5, 6, 7};
  const unsigned int              r           = 1;
  double time_step;

  ExactSolutionU exact_solution_u;
  ExactSolutionV exact_solution_v;

  exact_solution_u.set_time(5.0);
  exact_solution_v.set_time(5.0);

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2(u),eH1(u),eL2(v),eH1(v)" << std::endl;

  for (const auto &N_el : N_el_values)
    {
      Triangulation<dim> mesh; 
      GridGenerator::hyper_cube(mesh, 0, 1);
      mesh.refine_global(N_el);
        
      const double h = 1.0 / std::pow(2, N_el);
      time_step = h / 2.0;
    
      WaveEquation<2> problem(r, mesh, time_step, /*t*/0, /*time_step_number*/1, /*theta*/ 0.5);

      problem.run();

      

      const double error_L2_u =
        problem.compute_error(VectorTools::L2_norm, Target::Position, exact_solution_u);
      const double error_H1_u =
        problem.compute_error(VectorTools::H1_norm, Target::Position, exact_solution_u);


      const double error_L2_v =
        problem.compute_error(VectorTools::L2_norm, Target::Velocity, exact_solution_v);
      const double error_H1_v =
        problem.compute_error(VectorTools::H1_norm, Target::Velocity, exact_solution_v);
      table.add_value("h", h);
      table.add_value("L2(u)", error_L2_u);
      table.add_value("H1(u)", error_H1_u);
      table.add_value("L2(v)", error_L2_v);
      table.add_value("H1(v)", error_H1_v);

      convergence_file << h << "," << error_L2_u << "," << error_H1_u << "," << error_L2_v << "," << error_H1_v <<std::endl;
    }

  table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);

  table.set_scientific("L2(u)", true);
  table.set_scientific("H1(u)", true);
  table.set_scientific("L2(v)", true);
  table.set_scientific("H1(v)", true);

  table.write_text(std::cout);

}
