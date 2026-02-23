#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/utilities.h>

using namespace dealii;

enum class Target {
    Position,
    Velocity
};

template <int dim>
class WaveEquation
{
public:
    WaveEquation();
    WaveEquation(const int fe_degree_, const Triangulation<dim> &triangulation_, double time_step_, double time_, unsigned int timestep_number_, const double theta_);
    void run();
    double compute_error(const VectorTools::NormType &norm_type,
                         const Target target,
                         const Function<dim>  &exact_solution) const;


private:
    void setup_system();
    void solve_u();
    void solve_v();
    void output_results() const;
    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    Triangulation<dim> triangulation;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    Vector<double> solution_u, solution_v;
    Vector<double> old_solution_u, old_solution_v;
    Vector<double> system_rhs;

    double time_step;
    double time;
    unsigned int timestep_number;
    const double theta;
};

template <int dim>
class InitialValuesU : public Function<dim>
{
public:
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }
};

template <int dim>
class InitialValuesV : public Function<dim>
{
public:
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
    }
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        double t = this->get_time();
        return 2 * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * (1 + t * t * M_PI * M_PI);
    }
};

template <int dim>
class BoundaryValuesU : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        return 0;
    }
};

template <int dim>
class BoundaryValuesV : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));

        return 0;
    }
};
