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

template <int dim>
class WaveEquation
{
public:
    WaveEquation();
    void run();

private:
    void setup_system();
    void solve_u();
    void solve_v();
    void output_results() const;

    Triangulation<dim> triangulation;
    const FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

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
    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int component = 0) const override
    {
        (void)component;
        Assert(component == 0, ExcIndexRange(component, 0, 1));
        return 0;
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

        if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
            (p[1] > -1. / 3))
            return std::sin(this->get_time() * 4 * numbers::PI);
        else
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

        if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
            (p[1] > -1. / 3))
            return (std::cos(this->get_time() * 4 * numbers::PI) * 4 * numbers::PI);
        else
            return 0;
    }
};