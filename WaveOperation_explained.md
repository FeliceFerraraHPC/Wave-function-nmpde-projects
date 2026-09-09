# WaveOperation — Theory and Implementation Guide

## Table of Contents
1. [Purpose and Role](#1-purpose-and-role)
2. [Input Data — All Parameters Explained](#2-input-data--all-parameters-explained)
3. [The PDE and Its Weak Formulation](#3-the-pde-and-its-weak-formulation)
4. [Time Discretisation: the Leapfrog Scheme](#4-time-discretisation-the-leapfrog-scheme)
5. [The Constructor — Building the Mass Matrix](#5-the-constructor--building-the-mass-matrix)
   - [5.1 Why the Mass Matrix Is Diagonal](#51-why-the-mass-matrix-is-diagonal)
6. [The `local_apply` Kernel — Step by Step](#6-the-local_apply-kernel--step-by-step)
7. [Why Each Coefficient Is What It Is](#7-why-each-coefficient-is-what-it-is)
8. [Initial Acceleration and How the First Step Is Started](#8-initial-acceleration-and-how-the-first-step-is-started)
9. [Numerical Dissipation: Energy Conservation Analysis](#9-numerical-dissipation-energy-conservation-analysis)
   - [9.1 Continuous Hamiltonian Energy and Damping](#91-continuous-hamiltonian-energy-and-damping)
   - [9.2 Natural (Collocated) vs. Staggered Discrete Energy Formulations](#92-natural-collocated-vs-staggered-discrete-energy-formulations)
   - [9.3 Mathematical Proof of Exact Energy Conservation for the Leapfrog Scheme](#93-mathematical-proof-of-exact-energy-conservation-for-the-leapfrog-scheme)
   - [9.4 Spatial Bilinear Forms: CG Volume Term vs. DG/SIPG Face Penalties](#94-spatial-bilinear-forms-cg-volume-term-vs-dgsipg-face-penalties)
   - [9.5 Symplectic Structure & Zero Numerical Dissipation (CG vs. DG Comparison)](#95-symplectic-structure--zero-numerical-dissipation-cg-vs-dg-comparison)
   - [9.6 Code References: Implementation Across the Solver Suite](#96-code-references-implementation-across-the-solver-suite)
10. [Numerical Dispersion Analysis](#10-numerical-dispersion-analysis)
    - [10.1 Physics of Numerical Dispersion in High-Frequency Waves](#101-physics-of-numerical-dispersion-in-high-frequency-waves)
    - [10.2 Consistent Mass vs. Lumped Mass: Superluminal vs. Subluminal Dispersion](#102-consistent-mass-vs-lumped-mass-superluminal-vs-subluminal-dispersion)
    - [10.3 Direct Peak Tracking vs. Carrier Aliasing in Global L2 Minimization](#103-direct-peak-tracking-vs-carrier-aliasing-in-global-l2-minimization)
    - [10.4 Parametric Study: Polynomial Degree p = 1, 2, 4, 6 at Matched DOFs](#104-parametric-study-polynomial-degree-p--1-2-4-6-at-matched-dofs)
    - [10.5 Continuous Galerkin (CG) vs. Discontinuous Galerkin (DG) Dispersion](#105-continuous-galerkin-cg-vs-discontinuous-galerkin-dg-dispersion)
    - [10.6 Code References & Benchmark Execution](#106-code-references--benchmark-execution)
11. [Summary & Architectural Comparison Table](#11-summary--architectural-comparison-table)
12. [Convergence Analysis: Method of Manufactured Solutions (MMS) & Error Scaling](#12-convergence-analysis-method-of-manufactured-solutions-mms--error-scaling)
    - [12.1 Analytical Standing Wave Solution & Dirichlet Boundary Alignment](#121-analytical-standing-wave-solution--dirichlet-boundary-alignment)
    - [12.2 Discretization Error Decomposition: Spatial vs. Temporal Balance](#122-discretization-error-decomposition-spatial-vs-temporal-balance)
    - [12.3 Scaling Laws: Strict Spatial Scaling (dt ∝ h^2.5) vs. CFL Scaling (dt ∝ h)](#123-scaling-laws-strict-spatial-scaling-dt--h25-vs-cfl-scaling-dt--h)
    - [12.4 Startup Formulation: Fictitious Step u^{-1} = u_{exact}(-dt)](#124-startup-formulation-fictitious-step-u-1--u_exact-dt)
    - [12.5 Fine-Grid Roundoff Saturation Barrier](#125-fine-grid-roundoff-saturation-barrier)
    - [12.6 Verification Results Across the Suite (CG, DG, Theta)](#126-verification-results-across-the-suite-cg-dg-theta)
13. [Neumann Boundary Conditions & Acoustic Sound Waves (DG Alignment)](#13-neumann-boundary-conditions--acoustic-sound-waves-dg-alignment)
    - [13.1 Physics of Neumann Boundary Conditions: Sound-Hard Acoustic Walls](#131-physics-of-neumann-boundary-conditions-sound-hard-acoustic-walls)
    - [13.2 Mathematical Formulation Across Solvers (CG, DG, Theta)](#132-mathematical-formulation-across-solvers-cg-dg-theta)
    - [13.3 Exact Acoustic Sound Wave Mode (MMS for Neumann BCs)](#133-exact-acoustic-sound-wave-mode-mms-for-neumann-bcs)
    - [13.4 Acoustic Sound Pulse & Discontinuous Galerkin Advantages](#134-acoustic-sound-pulse--discontinuous-galerkin-advantages)
    - [13.5 Command-Line Interface & Usage Examples](#135-command-line-interface--usage-examples)

---

## 1. Purpose and Role

`WaveOperation<dim, fe_degree>` is a **matrix-free operator** that encodes a single time step of the **damped wave equation** solver. Its main job is to take the solution at the two previous time levels, $u^n$ and $u^{n-1}$, and produce the solution at the next level, $u^{n+1}$.

It never assembles any global sparse matrix. Instead it evaluates the action of the operator **cell by cell**, exploiting the tensor-product structure of the finite element basis and SIMD vectorisation. This is the `MatrixFree` approach of deal.II.

It is used exclusively by `WaveProblem`, which calls:

```cpp
wave_op.apply(solution, {&old_solution, &old_old_solution}, current_time);
```

to advance the simulation by one step.

---

## 2. Input Data — All Parameters Explained

```cpp
WaveOperation(
    const MatrixFree<dim, double> &data_in,      // (A)
    const double                   time_step_in, // (B)
    const double                   c_in = 1.0,   // (C)
    const double                   gamma_in = 0.0, // (D)
    const Function<dim>           *forcing_in = nullptr // (E)
);
```

| Symbol | Parameter | Physical meaning |
|--------|-----------|-----------------|
| — | `data_in` | **(A)** The pre-built `MatrixFree` object. It stores the mesh geometry, quadrature data, and DoF numbering in a format optimised for cell-loop evaluation. Passed by reference — **not** owned by `WaveOperation`. |
| $\Delta t$ | `time_step_in` | **(B)** The time step size $\Delta t$, computed externally via the CFL condition: $\Delta t = \nu \cdot h_{\min}$, where $\nu$ is the CFL number and $h_{\min}$ is the smallest mesh cell size. |
| $c$ | `c_in` | **(C)** The **wave speed** $c > 0$ (m/s). Appears as $c^2$ in the Laplacian term. Controls how fast disturbances propagate. Default: $c = 1$. |
| $\gamma$ | `gamma_in` | **(D)** The **damping coefficient** $\gamma \geq 0$. When $\gamma = 0$ the equation is energy-conserving; when $\gamma > 0$ energy is dissipated at rate $\gamma \int |\partial_t u|^2\,dx$. Default: $\gamma = 0$. |
| $f(x,t)$ | `forcing_in` | **(E)** Pointer to the **right-hand side** forcing function $f: \Omega \times [0,T] \to \mathbb{R}$. If `nullptr`, $f \equiv 0$ everywhere (homogeneous wave equation). |

### Internal derived quantities

Once constructed, `WaveOperation` precomputes and stores:

| Member | Value | Why stored |
|--------|-------|-----------|
| `c_sqr` | $c^2$ | Used in every cell kernel; avoids repeated multiplication |
| `delta_t_sqr` | $\Delta t^2$ as `VectorizedArray` | SIMD-broadcast constant for the cell kernel |
| `inv_effective_mass_matrix` | $\frac{1}{(1 + \frac{\Delta t}{2}\gamma)\,M_{\ell,i}}$ | Applied after each leapfrog step to obtain $u^{n+1}$ |
| `inv_mass_matrix` | $\frac{1}{M_{\ell,i}}$ | Used to solve for initial acceleration $a_0$ |

---

## 3. The PDE and Its Weak Formulation

### 3.1 Strong form

We solve the **damped wave equation** on a bounded domain $\Omega \subset \mathbb{R}^d$:

$$\boxed{ \partial_{tt} u - c^2 \Delta u + \gamma\, \partial_t u = f(x,t) \quad \text{in } \Omega \times (0,T] }$$

with homogeneous **Dirichlet** boundary conditions and initial data:

$$u = 0 \text{ on } \partial\Omega, \qquad u(x,0) = u_0(x), \quad \partial_t u(x,0) = u_1(x).$$

- The term $-c^2 \Delta u$ is the **elastic restoring force** (wave propagation).
- The term $\gamma\, \partial_t u$ is the **damping** (energy dissipation, like friction).
- $f(x,t)$ is an **external forcing**.

### 3.2 Weak form

Multiply by a test function $\phi \in H^1_0(\Omega)$ and integrate over $\Omega$. Using integration by parts on the Laplacian:

$$\int_\Omega \partial_{tt} u\, \phi\, dx
  + c^2 \int_\Omega \nabla u \cdot \nabla \phi\, dx
  + \gamma \int_\Omega \partial_t u\, \phi\, dx
  = \int_\Omega f\, \phi\, dx$$

> The boundary term $-c^2 \int_{\partial\Omega} (\nabla u \cdot \hat{n})\,\phi\, ds$ vanishes because $\phi = 0$ on $\partial\Omega$.

### 3.3 Finite element discretisation

Let $\{N_i\}$ be the FE basis (continuous, piecewise polynomial, degree 4, Gauss–Lobatto nodes). Expand:

$$u_h(x,t) = \sum_j u_j(t)\, N_j(x)$$

Substituting and choosing $\phi = N_i$:

$$\underbrace{\int_\Omega N_i N_j\, dx}_{M_{ij}}\, \ddot{u}_j
+ c^2 \underbrace{\int_\Omega \nabla N_i \cdot \nabla N_j\, dx}_{K_{ij}}\, u_j
+ \gamma M_{ij}\, \dot{u}_j = \int_\Omega f\, N_i\, dx$$

In matrix form:

$$\mathbf{M}\,\ddot{\mathbf{u}} + c^2\mathbf{K}\,\mathbf{u} + \gamma\mathbf{M}\,\dot{\mathbf{u}} = \mathbf{f}(t) \tag{ODE system}$$

where:
- $\mathbf{M}$ is the **mass matrix**
- $\mathbf{K}$ is the **stiffness matrix**
- $\mathbf{f}(t)$ is the load vector from $f(x,t)$

---

## 4. Time Discretisation: the Leapfrog Scheme

### 4.1 Central differences

We discretise $\ddot{u}$ and $\dot{u}$ using **central finite differences** at time $t^n = n\Delta t$:

$$\ddot{u}^n \approx \frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2}$$

$$\dot{u}^n \approx \frac{u^{n+1} - u^{n-1}}{2\Delta t} \quad \text{(Crank–Nicolson for damping)}$$

### 4.2 Substitution into the ODE system

Substituting into $\mathbf{M}\,\ddot{\mathbf{u}} + c^2\mathbf{K}\,\mathbf{u} + \gamma\mathbf{M}\,\dot{\mathbf{u}} = \mathbf{f}^n$:

$$\mathbf{M}\,\frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2}
+ c^2\mathbf{K}\,u^n
+ \gamma\mathbf{M}\,\frac{u^{n+1} - u^{n-1}}{2\Delta t}
= \mathbf{f}^n$$

Multiply through by $\Delta t^2$:

$$\mathbf{M}(u^{n+1} - 2u^n + u^{n-1})
+ \Delta t^2 c^2 \mathbf{K}\, u^n
+ \frac{\Delta t\, \gamma}{2}\mathbf{M}(u^{n+1} - u^{n-1})
= \Delta t^2 \mathbf{f}^n$$

Group terms in $u^{n+1}$ on the left and everything else on the right:

$$\left(1 + \tfrac{\Delta t\,\gamma}{2}\right)\mathbf{M}\, u^{n+1}
= 2\mathbf{M}\, u^n
- \left(1 - \tfrac{\Delta t\,\gamma}{2}\right)\mathbf{M}\, u^{n-1}
- \Delta t^2 c^2 \mathbf{K}\, u^n
+ \Delta t^2 \mathbf{f}^n$$

This is the **leapfrog update** implemented in the code.

### 4.3 Solving for $u^{n+1}$ — why we need $\mathbf{M}^{-1}$

The equation above has $\mathbf{M}$ on the left. To recover $u^{n+1}$ we need to invert it.

With **Gauss–Lobatto quadrature** and the `FE_Q` element evaluated at Gauss–Lobatto nodes, the mass matrix $\mathbf{M}$ becomes **diagonal** (this is the so-called *mass lumping* property — see [Section 5.1](#51-why-the-mass-matrix-is-diagonal) for the full proof). This means inversion is trivial: we just divide each entry by the diagonal element $M_{\ell,i}$.

Furthermore, the factor $\left(1 + \frac{\Delta t\,\gamma}{2}\right)$ is a **scalar** (uniform $\gamma$), so the *effective mass matrix* $\tilde{M}_i = \left(1 + \frac{\Delta t\,\gamma}{2}\right) M_{\ell,i}$ is also diagonal. The code stores:

$$\text{inv\_effective\_mass\_matrix}[i] = \frac{1}{\left(1 + \frac{\Delta t\,\gamma}{2}\right) M_{\ell,i}}$$

so the solve is just a pointwise multiplication (`dst.scale(...)`).

---

## 5. The Constructor — Building the Mass Matrix

```cpp
const double eff_mass_weight = 1.0 + 0.5 * time_step_in * gamma_in; // (1 + ½ dt γ)

for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
{
    fe_eval.reinit(cell);
    for (const unsigned int q : fe_eval.quadrature_point_indices())
        fe_eval.submit_value(make_vectorized_array(1.0), q);  // submit ϕ=1
    fe_eval.integrate(EvaluationFlags::values);
    fe_eval.distribute_local_to_global(inv_effective_mass_matrix);
    fe_eval.distribute_local_to_global(inv_mass_matrix);
}
```

**What this computes:** By submitting the constant value `1` at every quadrature point and integrating with `EvaluationFlags::values`, the cell loop computes:

$$M_{\ell,i} = \int_\Omega N_i\, dx \approx \sum_q w_q\, N_i(x_q) \cdot 1$$

Because Gauss–Lobatto nodes coincide with the FE node positions, $N_i(x_q) = \delta_{iq}$, so each diagonal entry gets the quadrature weight of its own node — **this is exactly the lumped mass** (see Section 5.1 for why).

After the loop, the diagonal entries are inverted:

```cpp
inv_effective_mass_matrix.local_element(k) = 1.0 / (eff_mass_weight * val);
inv_mass_matrix.local_element(k)           = 1.0 / val;
```

The guard `if (std::abs(val) > 1e-15)` protects against degenerate (zero-measure) nodes on hanging node constraints.

---

### 5.1 Why the Mass Matrix Is Diagonal

This is the most important structural fact in the whole solver. Let us prove it from first principles.

#### The mass matrix entry

The $(i,j)$ entry of the global mass matrix is:

$$M_{ij} = \int_\Omega N_i(x)\, N_j(x)\, dx$$

In general (e.g. with Gauss–Legendre quadrature) this integral is **non-zero for $i \neq j$**, producing a full banded matrix that is expensive to invert.

#### Step 1 — The Lagrange basis and its cardinal property

`FE_Q` with `QGaussLobatto<1>(fe_degree + 1)` constructs a **Lagrange interpolation basis** whose nodes $\{x_i\}$ are exactly the **Gauss–Lobatto points** on each reference element.

The defining property of a Lagrange basis is the **cardinal condition**:

$$\boxed{N_i(x_j) = \delta_{ij}}$$

That is, basis function $N_i$ equals **1 at its own node** $x_i$ and **0 at every other node** $x_j$ ($j \neq i$).

#### Step 2 — Numerical quadrature with Gauss–Lobatto points

To approximate $M_{ij} = \int_\Omega N_i\, N_j\, dx$ we use the same Gauss–Lobatto points as **quadrature nodes** $\{x_q\}$ with weights $\{w_q\}$:

$$M_{ij} \approx \sum_q w_q\, N_i(x_q)\, N_j(x_q)$$

#### Step 3 — Apply the cardinal property

Because the quadrature nodes $\{x_q\}$ **are the same set** as the FE nodes $\{x_i\}$, we can substitute the cardinal condition $N_i(x_q) = \delta_{iq}$:

$$M_{ij} \approx \sum_q w_q\, \underbrace{N_i(x_q)}_{= \delta_{iq}}\, \underbrace{N_j(x_q)}_{= \delta_{jq}}
= \sum_q w_q\, \delta_{iq}\, \delta_{jq}$$

The double Kronecker delta is non-zero only when $q = i$ **and** $q = j$ simultaneously, which requires $i = j$:

$$M_{ij} \approx w_i\, \delta_{ij}$$

The mass matrix is **diagonal**, with diagonal entries equal to the Gauss–Lobatto quadrature weights:

$$\mathbf{M} = \operatorname{diag}(w_1, w_2, \ldots, w_n)$$

#### Step 4 — Why this is an approximation (and when it's exact)

The quadrature rule $\sum_q w_q g(x_q)$ is **exact** for polynomials up to degree $2p - 1$ on the reference interval, where $p$ is the number of Gauss–Lobatto points ($p = \text{fe\_degree} + 1$). For `fe_degree = 4` we have $p = 5$ and exactness up to degree $9$.

The integrand $N_i(x)\, N_j(x)$ is a polynomial of degree $2 \times \text{fe\_degree} = 8$.

Since $8 \leq 9$, **the quadrature is exact** and the diagonal mass matrix is not an approximation — it is the **exact integral**.

> **Why not Gauss–Legendre?**
> With standard Gauss–Legendre quadrature the quadrature nodes $\{x_q\}$ are **different** from the FE nodes $\{x_i\}$. The cardinal condition $N_i(x_q) = \delta_{iq}$ no longer holds, so the cross-terms $N_i(x_q)\, N_j(x_q)$ with $i \neq j$ are generally non-zero, and $\mathbf{M}$ remains full.
>
> The coincidence of FE nodes and quadrature nodes — which is what makes $\mathbf{M}$ diagonal — is a special feature of the **Gauss–Lobatto** choice.

#### Summary of the argument

```
FE_Q with Gauss–Lobatto nodes
    → Lagrange basis: N_i(x_q) = δ_iq   (cardinal property)
        ↓
Gauss–Lobatto quadrature (same points as FE nodes)
    → M_ij ≈ Σ_q  w_q · δ_iq · δ_jq  =  w_i · δ_ij
        ↓
M is diagonal!  →  trivial inversion:  M⁻¹ = diag(1/w_i)
        ↓
Leapfrog solve is a single pointwise vector multiplication — O(N)
```

This is the key reason the matrix-free leapfrog scheme is so computationally efficient.

---

## 6. The `local_apply` Kernel — Step by Step

The kernel evaluates the **right-hand side** of the leapfrog update for a range of cell batches (a *batch* = a group of cells processed simultaneously via SIMD):

```
RHS = 2·M·uⁿ  -  (1 - ½dt·γ)·M·uⁿ⁻¹  -  dt²·c²·K·uⁿ  +  dt²·M·fⁿ
```

### 6.1 Coefficients

```cpp
const double val_coeff_curr = 2.0;                           //  coefficient of u^n  (value part)
const double val_coeff_old  = -(1.0 - 0.5*time_step*gamma); //  coefficient of u^{n-1}
const double dt_sqr         = time_step * time_step;         //  Δt²
const VectorizedArray<double> grad_coeff =
    make_vectorized_array(-c_sqr) * delta_t_sqr;             //  -c² Δt²  (stiffness part)
```

| Coefficient | Value | Origin |
|-------------|-------|--------|
| `val_coeff_curr` | $+2$ | from $+2\mathbf{M}u^n$ in the RHS |
| `val_coeff_old` | $-(1 - \frac{\Delta t}{2}\gamma)$ | from $-(1-\frac{\Delta t\gamma}{2})\mathbf{M}u^{n-1}$ |
| `grad_coeff` | $-c^2 \Delta t^2$ | from $-\Delta t^2 c^2 \mathbf{K}u^n$; the gradient term implements the stiffness matrix weakly |

### 6.2 Inside the quadrature loop

For each quadrature point $q$:

```cpp
// 1. Evaluate f(x_q, t^n)
f_val[v] = forcing->value(p);   // per SIMD lane

// 2. Value contribution: 2u^n - (1 - ½dt·γ)u^{n-1} + dt²·f
val_term = val_coeff_curr * u^n(q)
         + val_coeff_old  * u^{n-1}(q)
         + dt_sqr         * f(q);
submit_value(val_term, q);      // → accumulates ∫ val_term · N_i dx  (mass-like)

// 3. Gradient contribution: -c²Δt² · ∇u^n
submit_gradient(grad_coeff * ∇u^n(q), q); // → accumulates -c²Δt² ∫ ∇u^n · ∇N_i dx  (stiffness)
```

The `integrate(values | gradients)` call converts submitted values/gradients to nodal contributions via:

$$\text{local}[i] \mathrel{+}= \sum_q \left[ \text{val\_term}(q)\, N_i(x_q)\, w_q|J_q| \;+\; \text{grad\_coeff}(q) \cdot \nabla N_i(x_q)\, w_q|J_q| \right]$$

After `distribute_local_to_global`, these are added into the global `dst` vector.

### 6.3 Final scaling

Back in `apply()`:

```cpp
data.cell_loop(&WaveOperation::local_apply, this, dst, src, /*zero_dst=*/true);
dst.scale(inv_effective_mass_matrix);   //  u^{n+1} = M̃⁻¹ · RHS
```

This applies $\tilde{M}^{-1}$ pointwise, completing the solve for $u^{n+1}$.

---

## 7. Why Each Coefficient Is What It Is

Here is a side-by-side view connecting the algebraic derivation to the code:

$$\underbrace{\left(1 + \tfrac{\Delta t \gamma}{2}\right)}_{\text{\texttt{eff\_mass\_weight}}} \mathbf{M}\, u^{n+1}
= \underbrace{2}_{\texttt{val\_coeff\_curr}} \mathbf{M}\, u^n
\underbrace{- \left(1 - \tfrac{\Delta t \gamma}{2}\right)}_{\texttt{val\_coeff\_old}} \mathbf{M}\, u^{n-1}
\underbrace{- \Delta t^2 c^2}_{\texttt{grad\_coeff}} \mathbf{K}\, u^n
+ \underbrace{\Delta t^2}_{\texttt{dt\_sqr}} \mathbf{f}^n$$

| Term | Code variable | Role |
|------|--------------|------|
| $\left(1 + \frac{\Delta t\gamma}{2}\right)$ | `eff_mass_weight` (in constructor) | Left-hand side weight on $u^{n+1}$; goes into `inv_effective_mass_matrix` |
| $+2$ | `val_coeff_curr = 2.0` | Inertia from current step: central-difference $\ddot{u}$ numerator contributes $-2u^n$ on the LHS, or $+2u^n$ on the RHS |
| $-\left(1 - \frac{\Delta t\gamma}{2}\right)$ | `val_coeff_old` | Inertia from old step plus Crank–Nicolson damping at $t^{n-1}$; negative because $u^{n-1}$ goes to the RHS with a minus sign from $\ddot{u}$, partially offset by the $+\frac{\Delta t\gamma}{2}$ from the damping |
| $-c^2 \Delta t^2$ | `grad_coeff` | Stiffness contribution moved to RHS; negative because $c^2 \mathbf{K} u^n$ was on the LHS |
| $\Delta t^2$ | `dt_sqr` | Scaling of the load vector $\mathbf{f}^n$ |

> **Key insight:** The stiffness term $-c^2 \Delta t^2 \mathbf{K} u^n$ is never assembled as a matrix. Instead, it is evaluated weakly by submitting $\text{grad\_coeff} \cdot \nabla u^n$ at quadrature points (`submit_gradient`). When integrated, this produces exactly $\int (-c^2 \Delta t^2) \nabla u^n \cdot \nabla N_i\, dx = -c^2 \Delta t^2 K_{ij} u^n_j$.

---

## 8. Initial Acceleration and How the First Step Is Started

The leapfrog scheme needs **two** starting levels $u^0$ and $u^{-1}$ (a fictitious level). We cannot simply set $u^{-1} = u^0 - \Delta t\, u_1$ (forward Euler), because that is only first-order accurate and would pollute the second-order scheme.

The correct second-order start uses the **initial acceleration** $a_0$, obtained from the PDE at $t=0$:

$$a_0 = \partial_{tt} u(x,0) = c^2 \Delta u_0 - \gamma u_1 + f(x,0)$$

The weak form is:
$$\langle a_0, N_i \rangle = -c^2 \langle \nabla u_0, \nabla N_i \rangle - \gamma \langle u_1, N_i \rangle + \langle f(\cdot,0), N_i \rangle$$

(integration by parts on the Laplacian, boundary term vanishes).

This is what `local_compute_initial_acceleration` computes. Then:

$$u^{-1} = u_0 - \Delta t\, u_1 + \frac{\Delta t^2}{2}\, a_0 \qquad \text{(Taylor expansion, 2nd order)}$$
$$u^{+1} = u_0 + \Delta t\, u_1 + \frac{\Delta t^2}{2}\, a_0 \qquad \text{(first leapfrog step)}$$

In the code:
```cpp
wave_op.compute_initial_acceleration(a_0, solution, u_1);

// u^{+1}  (stored in u_1st_step, used as first "next" solution)
u_1st_step = solution;
u_1st_step.add( time_step,                   u_1);
u_1st_step.add( 0.5 * time_step * time_step, a_0);

// u^{-1}  (stored in old_solution — fictitious past level)
old_solution = solution;
old_solution.add(-time_step,                  u_1);
old_solution.add( 0.5 * time_step * time_step, a_0);
```

This guarantees that the very first regular leapfrog call `wave_op.apply(...)` is consistent with the same $O(\Delta t^2)$ accuracy as all subsequent steps.

---

## 9. Numerical Dissipation: Energy Conservation Analysis

A central quality benchmark for any transient wave solver is its ability to conserve or physically dissipate the Hamiltonian energy of the continuous system. In this section, we analyze the continuous energy balance, the distinction between natural and staggered discrete energies, provide a complete mathematical proof of exact energy conservation for the leapfrog scheme, and examine the zero-numerical-dissipation property across the CG and DG solvers.

### 9.1 Continuous Hamiltonian Energy and Damping

Consider the initial-boundary value problem for the damped wave equation on a bounded Lipschitz domain $\Omega \subset \mathbb{R}^d$ ($d \in \{1, 2, 3\}$):

$$\frac{\partial^2 u}{\partial t^2} + \gamma \frac{\partial u}{\partial t} - c^2 \Delta u = f(\mathbf{x}, t) \quad \text{in } \Omega \times (0, T],$$

subject to homogeneous Dirichlet boundary conditions $u = 0$ on $\Gamma_D = \partial\Omega$ (or homogeneous Neumann conditions $\nabla u \cdot \mathbf{n} = 0$ on $\Gamma_N$) and initial conditions $u(\mathbf{x}, 0) = u_0(\mathbf{x})$, $\partial_t u(\mathbf{x}, 0) = v_0(\mathbf{x})$.

The total Hamiltonian energy $E(t)$ of the system is the sum of the kinetic energy $E_{\text{kin}}(t)$ and the potential (strain) energy $E_{\text{pot}}(t)$:

$$E(t) \equiv E_{\text{kin}}(t) + E_{\text{pot}}(t) = \frac{1}{2} \int_\Omega \left( \frac{\partial u}{\partial t} \right)^2 d\Omega + \frac{c^2}{2} \int_\Omega |\nabla u|^2 d\Omega = \frac{1}{2} \|\partial_t u\|_{L^2(\Omega)}^2 + \frac{c^2}{2} \|\nabla u\|_{L^2(\Omega)}^2.$$

To derive the continuous energy balance, we take the inner product of the PDE with the velocity field $v = \partial_t u$:

$$\int_\Omega \frac{\partial u}{\partial t} \frac{\partial^2 u}{\partial t^2} d\Omega + \gamma \int_\Omega \left(\frac{\partial u}{\partial t}\right)^2 d\Omega - c^2 \int_\Omega \frac{\partial u}{\partial t} \Delta u \, d\Omega = \int_\Omega f \frac{\partial u}{\partial t} d\Omega.$$

Recognizing that $\partial_t u \, \partial_{tt} u = \frac{1}{2} \frac{d}{dt} (\partial_t u)^2$, and integrating the Laplacian term by parts:

$$- c^2 \int_\Omega \frac{\partial u}{\partial t} \Delta u \, d\Omega = c^2 \int_\Omega \nabla\left(\frac{\partial u}{\partial t}\right) \cdot \nabla u \, d\Omega - c^2 \int_{\partial\Omega} \frac{\partial u}{\partial t} (\nabla u \cdot \mathbf{n}) \, d\sigma.$$

Since $u = 0$ on $\partial\Omega \implies \partial_t u = 0$ on $\partial\Omega$ (for Dirichlet boundaries), the boundary integral vanishes identically. By the chain rule, $\nabla(\partial_t u) \cdot \nabla u = \frac{1}{2} \frac{d}{dt} |\nabla u|^2$. Consequently:

$$\frac{d}{dt} \left[ \frac{1}{2} \int_\Omega \left(\frac{\partial u}{\partial t}\right)^2 d\Omega + \frac{c^2}{2} \int_\Omega |\nabla u|^2 d\Omega \right] + \gamma \int_\Omega \left(\frac{\partial u}{\partial t}\right)^2 d\Omega = \int_\Omega f \frac{\partial u}{\partial t} d\Omega.$$

Defining the **instantaneous dissipation rate** $D(t) \equiv \gamma \|\partial_t u\|_{L^2(\Omega)}^2 = 2\gamma E_{\text{kin}}(t)$, we obtain the continuous energy identity:

$$\frac{dE(t)}{dt} = - D(t) + \int_\Omega f(\mathbf{x}, t) \frac{\partial u}{\partial t} d\Omega = - 2\gamma E_{\text{kin}}(t) + \langle f, \partial_t u \rangle_{L^2(\Omega)}.$$

This relation yields two fundamental physical conclusions:
1. **Conservative System ($f = 0, \gamma = 0$):**
   $$\frac{dE(t)}{dt} = 0 \iff E(t) = E(0) \quad \forall t \ge 0.$$
   The continuous wave equation is a non-dissipative, conservative Hamiltonian dynamical system.
2. **Dissipative System ($f = 0, \gamma > 0$):**
   $$\frac{dE(t)}{dt} = - 2\gamma E_{\text{kin}}(t) \le 0 \implies E(t) \le E(0) \quad \forall t \ge 0.$$
   Energy decays monotonically at a rate governed strictly by the damping coefficient $\gamma$.

---

### 9.2 Natural (Collocated) vs. Staggered Discrete Energy Formulations

When measuring energy in a discrete finite element simulation, one must distinguish between the **natural (collocated)** energy and the **staggered (symplectic)** energy. Both observables are computed by [`EnergyData`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverBase.hpp#L38-L66) across our solver suite:

#### 1. Natural (Collocated) Discrete Energy $E_{\text{nat}}^n$
In standard post-processing, one evaluates the solution and its velocity at the integer time level $t^n = n\Delta t$. Because the leapfrog scheme computes displacement at integer levels $\{u^{n-1}, u^n, u^{n+1}\}$, the collocated velocity $v^n \approx \partial_t u(t^n)$ is reconstructed via the centered second-order difference:

$$v^n \equiv \frac{u^{n+1} - u^{n-1}}{2\Delta t}.$$

The natural discrete energy is defined as:

$$E_{\text{nat}}^n \equiv \frac{1}{2} (v^n)^T \mathbf{M} v^n + \frac{c^2}{2} (u^n)^T \mathbf{K} u^n = E_{\text{kin, nat}}^n + E_{\text{pot, nat}}^n.$$

**Behavior of $E_{\text{nat}}^n$:**
Even for an undamped, unforced wave, $E_{\text{nat}}^n$ is **not strictly constant**; it exhibits a bounded $\mathcal{O}(\Delta t^2)$ oscillation. This oscillation is not a numerical dissipation or secular drift; rather, it is a phase artifact of evaluating velocity across an interval of width $2\Delta t$ while the potential energy is evaluated at $t^n$. The kinetic and potential energies exchange energy with an $\mathcal{O}(\Delta t^2 \cos(2\omega t^n))$ phase error.

#### 2. Staggered Discrete Energy $E_{\text{stag}}^{n+1/2}$
The leapfrog integrator is fundamentally a staggered time-stepping algorithm: the displacement is naturally located at integer time levels $t^n$, while the discrete velocity canonically lives at half-integer time levels $t^{n+1/2}$:

$$v^{n+1/2} \equiv \frac{u^{n+1} - u^n}{\Delta t}.$$

The staggered discrete energy at $t^{n+1/2}$ is defined as:

$$E_{\text{stag}}^{n+1/2} \equiv \frac{1}{2} \left( \frac{u^{n+1} - u^n}{\Delta t} \right)^T \mathbf{M} \left( \frac{u^{n+1} - u^n}{\Delta t} \right) + \frac{1}{2} a_h(u^n, u^{n+1}),$$

where $\mathbf{M}$ is the lumped mass matrix (Gauss–Lobatto diagonal) and $a_h(u^n, u^{n+1})$ is the symmetric discrete spatial bilinear form evaluated between consecutive time levels $u^n$ and $u^{n+1}$.

As proven below, $E_{\text{stag}}^{n+1/2}$ is an **exact algebraic invariant** of the discrete leapfrog equations of motion.

---

### 9.3 Mathematical Proof of Exact Energy Conservation for the Leapfrog Scheme

We now provide the rigorous algebraic proof that the discrete leapfrog scheme conserves $E_{\text{stag}}^{n+1/2}$ down to floating-point machine precision.

**Theorem (Exact Staggered Energy Conservation):**
Let $\mathbf{M}$ be a symmetric positive-definite mass matrix, and let $\mathbf{K}$ be a symmetric spatial stiffness matrix representing the bilinear form $a_h(\cdot, \cdot)$. In the absence of forcing ($f = 0$) and damping ($\gamma = 0$), the discrete leapfrog time-stepping scheme:

$$\mathbf{M} \left( \frac{u^{n+1} - 2 u^n + u^{n-1}}{\Delta t^2} \right) + \mathbf{K} u^n = 0$$

satisfies:

$$E_{\text{stag}}^{n+1/2} - E_{\text{stag}}^{n-1/2} \equiv 0 \quad \forall n \ge 1.$$

**Proof:**
Take the algebraic inner product of the leapfrog equation with the centered displacement variation $\frac{u^{n+1} - u^{n-1}}{2\Delta t}$:

$$\frac{1}{2\Delta t^3} (u^{n+1} - u^{n-1})^T \mathbf{M} (u^{n+1} - 2u^n + u^{n-1}) + \frac{1}{2\Delta t} (u^{n+1} - u^{n-1})^T \mathbf{K} u^n = 0.$$

We decompose each term individually.

**1. Kinetic Term Decompostion:**
Decompose the vectors into forward and backward differences:
$$u^{n+1} - u^{n-1} = (u^{n+1} - u^n) + (u^n - u^{n-1}),$$
$$u^{n+1} - 2u^n + u^{n-1} = (u^{n+1} - u^n) - (u^n - u^{n-1}).$$
Let $\mathbf{a} \equiv u^{n+1} - u^n$ and $\mathbf{b} \equiv u^n - u^{n-1}$. Using the bilinearity and symmetry of $\mathbf{M}$ ($\mathbf{M}^T = \mathbf{M}$):
$$(\mathbf{a} + \mathbf{b})^T \mathbf{M} (\mathbf{a} - \mathbf{b}) = \mathbf{a}^T \mathbf{M} \mathbf{a} - \mathbf{a}^T \mathbf{M} \mathbf{b} + \mathbf{b}^T \mathbf{M} \mathbf{a} - \mathbf{b}^T \mathbf{M} \mathbf{b} = \mathbf{a}^T \mathbf{M} \mathbf{a} - \mathbf{b}^T \mathbf{M} \mathbf{b}.$$
Dividing by $2\Delta t^3$:
$$\frac{1}{2\Delta t^3} (u^{n+1} - u^{n-1})^T \mathbf{M} (u^{n+1} - 2u^n + u^{n-1}) = \frac{1}{\Delta t} \left[ \frac{1}{2} \left(\frac{u^{n+1}-u^n}{\Delta t}\right)^T \mathbf{M} \left(\frac{u^{n+1}-u^n}{\Delta t}\right) - \frac{1}{2} \left(\frac{u^n-u^{n-1}}{\Delta t}\right)^T \mathbf{M} \left(\frac{u^n-u^{n-1}}{\Delta t}\right) \right]$$
$$= \frac{1}{\Delta t} \left( E_{\text{kin, stag}}^{n+1/2} - E_{\text{kin, stag}}^{n-1/2} \right).$$

**2. Potential Term Decomposition:**
Similarly, expand the stiffness term:
$$(u^{n+1} - u^{n-1})^T \mathbf{K} u^n = (u^{n+1})^T \mathbf{K} u^n - (u^{n-1})^T \mathbf{K} u^n.$$
By symmetry of $\mathbf{K}$ ($\mathbf{K}^T = \mathbf{K}$, which holds because $a_h(u, w) = a_h(w, u)$):
$$(u^{n+1})^T \mathbf{K} u^n = (u^n)^T \mathbf{K} u^{n+1} = a_h(u^n, u^{n+1}),$$
$$(u^{n-1})^T \mathbf{K} u^n = a_h(u^{n-1}, u^n).$$
Dividing by $2\Delta t$:
$$\frac{1}{2\Delta t} (u^{n+1} - u^{n-1})^T \mathbf{K} u^n = \frac{1}{\Delta t} \left[ \frac{1}{2} a_h(u^n, u^{n+1}) - \frac{1}{2} a_h(u^{n-1}, u^n) \right] = \frac{1}{\Delta t} \left( E_{\text{pot, stag}}^{n+1/2} - E_{\text{pot, stag}}^{n-1/2} \right).$$

**3. Total Balance:**
Summing the kinetic and potential increments:
$$\frac{E_{\text{stag}}^{n+1/2} - E_{\text{stag}}^{n-1/2}}{\Delta t} = 0 \implies E_{\text{stag}}^{n+1/2} \equiv E_{\text{stag}}^{n-1/2} = \text{constant} \quad \forall n.$$
$\blacksquare$

This exact conservation holds for any time step $\Delta t$ within the CFL stability limit, independent of spatial mesh resolution $h$. In numerical simulations, the computed `stag_energy_decay` stays strictly at machine precision ($\approx 10^{-14}$ in double precision arithmetic).

---

### 9.4 Spatial Bilinear Forms: CG Volume Term vs. DG/SIPG Face Penalties

The definition of the spatial bilinear form $a_h(\cdot, \cdot)$ depends directly on the spatial finite element formulation:

#### 1. Continuous Galerkin (CG / MatrixFree)
For conforming $H^1$-conforming finite elements ($u \in V_h \subset H_0^1(\Omega)$), the functions are continuous across cell faces ($[u] = 0$). The bilinear form consists solely of the volume grad-grad integral:

$$a_h(u, w) = \int_\Omega c^2 \nabla u \cdot \nabla w \, d\Omega = \sum_{K \in \mathcal{T}_h} \int_K c^2 \nabla u \cdot \nabla w \, d\Omega.$$

In [`WaveSolverMatFree::compute_energy`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverMatFree.cpp#L414-L417), this is evaluated quadrature point by quadrature point:
```cpp
local_pot_stag += 0.5 * (grad_u_curr[q] * grad_u_next[q]) * JxW;
```
The staggered kinetic energy uses the diagonal lumped mass vector `lumped_mass_` directly without needing any quadrature:
```cpp
local_kin_stag += lumped_mass_.local_element(i) * diff * diff;
```

#### 2. Discontinuous Galerkin (DG / SIPG)
For discontinuous spaces ($u \in V_h^{\text{DG}} \not\subset H^1(\Omega)$), functions are discontinuous across inter-element faces $F \in \mathcal{F}_h^{\text{int}}$ and Dirichlet boundary faces $F \in \mathcal{F}_h^{\text{bnd}}$. To ensure consistency, symmetry, and coercivity, the Symmetric Interior Penalty Galerkin (SIPG) bilinear form is employed:

$$a_h(u, w) = \sum_{K \in \mathcal{T}_h} \int_K c^2 \nabla u \cdot \nabla w \, d\Omega - \sum_{F \in \mathcal{F}_h^{\text{int}}} \int_F c^2 \left( \{\nabla u \cdot \mathbf{n}_F\} [w] + \{\nabla w \cdot \mathbf{n}_F\} [u] - \frac{\sigma_F}{h_F} [u] [w] \right) d\sigma - \sum_{F \in \mathcal{F}_h^{\text{bnd}}} \int_F c^2 \left( (\nabla u \cdot \mathbf{n}) w + (\nabla w \cdot \mathbf{n}) u - \frac{2\sigma_F}{h_F} u w \right) d\sigma,$$

where:
- Average: $\{\nabla u\} = \frac{1}{2}(\nabla u^+ + \nabla u^-)$,
- Jump: $[u] = u^+ - u^-$ along normal $\mathbf{n}_F = \mathbf{n}^+$,
- Penalty parameter: $\sigma_F = \gamma_0 \frac{p(p+1)}{h_F}$ with $\gamma_0$ chosen sufficiently large to guarantee coercivity ($a_h(u, u) \ge \alpha \|u\|_{\text{DG}}^2 > 0$).

Because SIPG is symmetric by design ($a_h(u, w) = a_h(w, u)$), the theorem in Section 9.3 applies verbatim. In [`WaveSolverDG::compute_energy`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverDG.cpp#L510-L642), the face contributions are computed by iterating over all interior faces (ensuring each face is visited exactly once across MPI ranks using cell subdomain IDs) and boundary faces.

---

### 9.5 Symplectic Structure & Zero Numerical Dissipation (CG vs. DG Comparison)

The leapfrog time integrator belongs to the class of **symplectic Störmer–Verlet integrators**. In Hamiltonian dynamics, a symplectic numerical integrator preserves the differential 2-form $d\mathbf{p} \wedge d\mathbf{q}$ in phase space:

$$\det\left( \frac{\partial (u^{n+1}, v^{n+1/2})}{\partial (u^n, v^{n-1/2})} \right) \equiv 1.$$

By backward error analysis (Hairer, Lubich, and Wanner), a symplectic integrator does not solve the exact Hamiltonian $H(u, v) = \frac{1}{2} v^T \mathbf{M} v + \frac{1}{2} u^T \mathbf{K} u$, but rather tracks the **exact trajectory of a modified shadow Hamiltonian**:

$$\widetilde{H}(u, v) = H(u, v) + \Delta t^2 H_2(u, v) + \Delta t^4 H_4(u, v) + \mathcal{O}(\Delta t^6).$$

Because $\widetilde{H}$ is an exact invariant of the numerical flow, the discrete solution cannot suffer from secular energy growth or artificial numerical damping:
- **Zero Numerical Dissipation:** Unlike dissipative schemes (such as implicit backward Euler, Runge–Kutta with negative dissipation, or theta-schemes with $\theta > 1/4$), the leapfrog integrator possesses **identically zero numerical dissipation**. Wave packet amplitudes remain undiminished over thousands of periods.
- **Physical Damping Verification:** When $\gamma > 0$, the energy decay observed in [`WaveSolverMatFree`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverMatFree.cpp) is $100\%$ physical: the numerical decay rate $\frac{\Delta E}{\Delta t}$ matches the theoretical dissipation rate $D(t) = 2\gamma E_{\text{kin}}(t)$ to high accuracy.

---

### 9.6 Code References: Implementation Across the Solver Suite

The energy and dissipation diagnostics are unified across all three solvers in the repository:

| Concept / Diagnostic | Math Notation | Implementation File & Line Range | Key Code Construct |
| :--- | :--- | :--- | :--- |
| **Energy Data Structure** | $E_{\text{nat}}, E_{\text{stag}}, D$ | [`src/WaveSolverBase.hpp:L38-L66`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverBase.hpp#L38-L66) | `struct EnergyData` |
| **CSV Exporter** | $E(t)$ history | [`src/WaveSolverBase.hpp:L73-L95`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverBase.hpp#L73-L95) | `write_energy_history_csv` (17 decimal digits) |
| **CG Energy Evaluation** | $E_{\text{stag}} = \frac{1}{2}\|v\|_{\mathbf{M}}^2 + \frac{1}{2} a_h$ | [`src/WaveSolverMatFree.cpp:L360-L474`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverMatFree.cpp#L360-L474) | `WaveSolverMatFree::compute_energy()` |
| **CG Staggered Kinetic** | $\frac{1}{2} \sum_i m_i (\frac{u^{n+1}_i - u^n_i}{\Delta t})^2$ | [`src/WaveSolverMatFree.cpp:L439-L448`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverMatFree.cpp#L439-L448) | `lumped_mass_.local_element(i) * diff * diff` |
| **DG Energy Evaluation** | SIPG face cross-terms | [`src/WaveSolverDG.cpp:L445-L665`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverDG.cpp#L445-L665) | `WaveSolverDG::compute_energy()` |
| **DG Face Integrals** | $\int_F (\{\nabla u\} [w] + \sigma [u][w])$ | [`src/WaveSolverDG.cpp:L550-L645`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverDG.cpp#L550-L645) | Interior & boundary face jump integrals |
| **Theta Energy Evaluation** | Consistent mass & stiffness | [`src/WaveSolverTheta.cpp:L375-L442`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverTheta.cpp#L375-L442) | `WaveSolverTheta::compute_energy()` |

---

## 10. Numerical Dispersion Analysis

While energy conservation guarantees amplitude stability, high-frequency wave propagation is equally susceptible to **numerical dispersion** — the artificial variation of propagation speed as a function of wavenumber $k$, mesh size $h$, polynomial order $p$, and mass matrix formulation.

### 10.1 Physics of Numerical Dispersion in High-Frequency Waves

In the continuous wave equation $\partial_{tt} u - c^2 \Delta u = 0$, plane wave solutions $e^{i(\mathbf{k} \cdot \mathbf{x} - \omega t)}$ satisfy the linear dispersion relation:

$$\omega = c \|\mathbf{k}\| \implies v_p \equiv \frac{\omega}{\|\mathbf{k}\|} = c, \quad v_g \equiv \nabla_{\mathbf{k}} \omega = c \frac{\mathbf{k}}{\|\mathbf{k}\|}.$$

Phase velocity $v_p$ and group velocity $v_g$ are identical, isotropic, and independent of frequency: the continuous system is completely **dispersionless**.

Upon spatial and temporal discretisation, the discrete wave operator yields a discrete dispersion relation $\omega_h(\mathbf{k}) \ne c \|\mathbf{k}\|$. The numerical phase speed $c_{\text{num}}(\mathbf{k}) \equiv \frac{\omega_h(\mathbf{k})}{\|\mathbf{k}\|}$ deviates from $c$:
- If $c_{\text{num}} < c$, the numerical wave propagates **slower** than the physical wave (phase lag, subluminal dispersion).
- If $c_{\text{num}} > c$, the numerical wave propagates **faster** than the physical wave (phase lead, superluminal dispersion).

For high-order polynomial elements of degree $p$, the asymptotic relative phase error scales as:

$$\left| \frac{c_{\text{num}} - c}{c} \right| \le C_p (k h)^{2p}.$$

For linear elements ($p = 1$), the dispersion error decays as $\mathcal{O}((kh)^2)$, requiring many grid points per wavelength ($N_{\lambda} = \frac{\lambda}{h} \ge 15\text{--}20$) to prevent phase decoherence. For high-order elements ($p = 4, 6$), the error decays as $\mathcal{O}((kh)^8)$ and $\mathcal{O}((kh)^{12})$, enabling accurate wave packet propagation over hundreds of wavelengths with only $3\text{--}4$ points per wavelength.

---

### 10.2 Consistent Mass vs. Lumped Mass: Superluminal vs. Subluminal Dispersion

A profound mathematical distinction exists between consistent mass formulations and lumped mass formulations, governed by the classic spectral theorem of Hughes (1987) and Ainsworth (2004):

```
                                 DISPERSION MODES
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
  Consistent Mass Matrix                                   Lumped Mass Matrix
  (Full Galerkin / Theta)                              (Gauss–Lobatto / MatFree)
  ───────────────────────                              ─────────────────────────
  • Rayleigh quotient upper bound:                     • Under-integration lowers eigenvalues:
      λ_h ≥ λ  ⟹  ω_h ≥ c·k                               λ_h ≤ λ  ⟹  ω_h ≤ c·k
  • Numerical speed c_num > c                          • Numerical speed c_num < c
  • SUPERLUMINAL (Phase Lead)                          • SUBLUMINAL (Phase Lag)
  • Wave packet arrives EARLY (Δt < 0)                 • Wave packet arrives LATE (Δt > 0)
```

1. **Consistent Mass Matrix (Full Galerkin, `WaveSolverTheta`):**
   When the continuous eigenvalue problem $-\Delta \phi = \lambda \phi$ is discretised using the full, un-lumped mass matrix $\mathbf{M}_{\text{cons}}$:
   $$\mathbf{K} \mathbf{\Phi} = \lambda_h \mathbf{M}_{\text{cons}} \mathbf{\Phi}.$$
   By the Rayleigh–Ritz min-max principle, the subspace approximation provides an **upper bound** on all discrete eigenvalues: $\lambda_{h, j} \ge \lambda_j \implies \omega_{h, j} \ge \omega_j$.
   Consequently, the spatial discretization accelerates high frequencies:
   $$c_{\text{num}} > c \implies \Delta t_{\text{shift}} < 0 \quad (\text{Superluminal Phase Lead}).$$
   This was observed in the benchmark: `WaveSolverTheta` produced negative time shifts ($\Delta t \approx -1.26\text{ s}$ to $-1.67\text{ s}$, with the wave peak at $x \approx 11.3\text{--}11.6$ ahead of the physical target $x = 10.0$).

2. **Lumped Mass Matrix (Gauss–Lobatto Quadrature, `WaveSolverMatFree`, `WaveSolverDG`):**
   Collocating the quadrature points at the Gauss–Lobatto support points evaluates the mass matrix via numerical quadrature:
   $$(\mathbf{M}_{\text{lump}})_{ii} = \int_K N_i(\mathbf{x}) d\Omega \approx w_i |J_K|.$$
   Because the Gauss–Lobatto quadrature under-integrates polynomials of degree $2p$ (exact only up to degree $2p-1$), it systematically depresses the kinetic energy and lowers the discrete eigenvalues: $\lambda_{h, j} \le \lambda_j \implies \omega_{h, j} \le \omega_j$.
   Consequently, mass lumping decelerates the wave:
   $$c_{\text{num}} < c \implies \Delta t_{\text{shift}} > 0 \quad (\text{Subluminal Phase Lag}).$$
   This was observed in the benchmark: `WaveSolverMatFree` produced a small positive time shift ($\Delta t = +0.07638\text{ s}$, with the wave peak at $x = 9.924 < 10.0$).

---

### 10.3 Direct Peak Tracking vs. Carrier Aliasing in Global L2 Minimization

To measure dispersion in a propagating wave packet:

$$u(\mathbf{x}, t) = A \exp\left( -\frac{(x - x_0 - c_{\text{eff}} t)^2}{2\sigma^2} \right) \cos(k x - \omega t) \sin\left(\frac{\pi (y - y_{\min})}{L_y}\right),$$

a standard approach is to minimise the global $L_2$ error over time shifts:

$$\min_{s \in \mathbb{R}} \| u_h(\mathbf{x}, T) - u_{\text{exact}}(\mathbf{x}, T - s) \|_{L^2(\Omega)}.$$

**The Carrier Aliasing Problem:**
For high carrier wavenumbers ($k = 2\pi \implies \lambda = 1.0$), the objective function $J(s) \equiv \|u_h(T) - u_{\text{exact}}(T - s)\|_{L^2}$ exhibits sharp local minima spaced periodically at every carrier period $\Delta s = \frac{\lambda}{c_{\text{eff}}} \approx 1.0\text{ s}$. Any minor envelope distortion or boundary residue causes global optimization algorithms (e.g. ternary search or gradient descent) to jump into a neighboring cycle, producing false aliased shifts of $\Delta t = \pm 1.0\text{ s}, \pm 2.0\text{ s}$.

**The 3-Stage Direct Peak Tracking Solution (`locate_peak_1d`):**
To eliminate carrier aliasing entirely, [`locate_peak_1d`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverBase.hpp#L140-L205) tracks the physical envelope crest directly along the centerline $y = 0$:
1. **Stage 1 (Uniform Coarse Scan):** Evaluate the numerical solution at 500 points across the window $[x_{\text{exact}} - 2.5, x_{\text{exact}} + 2.5]$ to identify the global maximum crest index.
2. **Stage 2 (Local Sub-Grid Bracketing):** Construct an ultra-fine local sampling grid ($\delta x = 0.001$) around the candidate crest.
3. **Stage 3 (Parabolic 3-Point Interpolation):** Fit a quadratic polynomial through the three highest sampled points $(x_{-1}, u_{-1}), (x_0, u_0), (x_1, u_1)$ to pinpoint the true sub-grid peak coordinate $x_{\text{num}}^*$:
   $$x_{\text{num}}^* = x_0 - \frac{1}{2} \frac{(u_1 - u_{-1})\delta x}{u_1 - 2 u_0 + u_{-1}}.$$
4. **Physical Observables:**
   - Physical shift: $\Delta x = x_{\text{exact}}^* - x_{\text{num}}^*$,
   - Equivalent time delay: $\Delta t = \frac{\Delta x}{c_{\text{eff}}}$,
   - Phase error: $\Delta\phi = k \Delta x \pmod{2\pi}$ (radians),
   - Aligned $L_2$ error: $\|u_h(T) - u_{\text{exact}}(T - \Delta t)\|_{L^2(\Omega)}$ (isolating pure envelope/amplitude deformation from spatial translation).

---

### 10.4 Parametric Study: Polynomial Degree p = 1, 2, 4, 6 at Matched DOFs

The numerical dispersion analysis was executed using [`run_dispersion`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/dispersion_analysis.hpp#L195-L368) on a 2D domain $[-15, 15]^2$ with wave parameters:
- Wavenumber $k = 2\pi \approx 6.28319$ ($\lambda = 1.0$),
- Gaussian envelope width $\sigma = 1.0$,
- Initial center $x_0 = -8.0$, Final time $T = 18.0 \implies$ Exact target peak at $x = 10.0$,
- Matched degrees of freedom $\approx 16,000$.

#### Part 1: Polynomial Degree Study (`WaveSolverTheta` Crank–Nicolson, Consistent Mass)
| Degree $p$ | Refinement | Active Cells | Total DOFs | $L_2$ (Raw) | $L_2$ (Aligned) | Time Shift $\Delta t$ (s) | Phase Shift $\Delta\phi$ (rad) | Dispersion Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$p = 1$** | 7 | 16,384 | 16,641 | $5.506 \times 10^0$ | $5.289 \times 10^0$ | $-1.351$ | $-8.489$ | Superluminal lead |
| **$p = 2$** | 6 | 4,096 | 16,641 | $6.203 \times 10^0$ | $2.686 \times 10^0$ | $-1.518$ | $-9.538$ | Superluminal lead |
| **$p = 4$** | 5 | 1,024 | 16,641 | $4.680 \times 10^0$ | $2.536 \times 10^0$ | $-1.266$ | $-7.957$ | Superluminal lead |
| **$p = 6$** | 4 | 256 | 9,409 | $5.608 \times 10^0$ | $3.895 \times 10^0$ | $-1.672$ | $-10.504$ | Superluminal lead |

**Observations:**
- In all cases, $\Delta t < 0$, confirming superluminal wave speed as predicted by the Rayleigh–Ritz upper bound for consistent mass matrices.
- The raw $L_2$ error is large ($\approx 5.0$) entirely because a phase shift of $\approx 1.3\text{ s}$ shifts the wave packet by more than a full carrier wavelength ($\lambda = 1.0$).
- When the phase shift is compensated ($L_2^{\text{aligned}}$), the error drops by more than $55\%$ for $p \ge 2$, revealing that the underlying wave packet geometry is preserved.

---

### 10.5 Continuous Galerkin (CG) vs. Discontinuous Galerkin (DG) Dispersion

#### Part 2: Matched Comparison at $p = 4$, Refinement 5 ($T = 18.0\text{ s}$, 18 Wavelengths Propagated)
| Solver Formulation | DOFs | Raw $L_2$ Error | Aligned $L_2$ Error | Peak Position $x_{\text{num}}$ | Time Delay $\Delta t$ | Phase Lag $\Delta\phi$ | Relative Lag $\Delta t / T$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CG (Matrix-Free, Leapfrog)** | 16,641 | **$1.042 \times 10^0$** | **$8.530 \times 10^{-1}$** | **$9.924$** | **$+0.07638\text{ s}$** | **$+0.4799\text{ rad}$** | **$0.42\%$** |
| **DG (SIPG, Leapfrog)** | 25,600 | **$1.358 \times 10^0$** | **$8.874 \times 10^{-1}$** | **$9.878$** | **$+0.12248\text{ s}$** | **$+0.7696\text{ rad}$** | **$0.68\%$** |

**Key Physical Insights:**
1. **Lumped Mass Subluminal Lag:** Both CG and DG exhibit positive time delays ($\Delta t > 0$, $x_{\text{num}} < 10.0$), perfectly confirming the Gauss–Lobatto mass-lumping subluminal property.
2. **Spectral Fidelity of Degree $p = 4$:** Over an extended propagation distance of 18 carrier wavelengths ($18.0\text{ length units}$), the relative phase delay is less than **$0.42\%$** for CG and **$0.68\%$** for DG!
3. **CG vs. DG Phase Lag Mechanism:** DG exhibits a slightly larger phase delay ($\Delta t = 0.122\text{ s}$ vs $0.076\text{ s}$) due to the penalty terms $\frac{\sigma_F}{h_F} [u][w]$ across the $1,024$ element faces. The interior penalty fluxes introduce a microscopic numerical impedance on element interfaces that slightly retards the wave front.
4. **Shape Invariance:** Both methods achieve an aligned $L_2$ error of $\approx 0.85\text{--}0.88$, demonstrating that the wave packet preserves its Gaussian envelope shape without numerical dissipation or dispersion tail artifacts.

---

### 10.6 Code References & Benchmark Execution

The numerical dispersion diagnostics are accessible via the `WaveBenchmark` executable:

```bash
# Execute the full 2D dispersion benchmark (Part 1: p-study, Part 2: CG vs DG)
./WaveBenchmark --mode dispersion --dim 2 --refine 5
```

Key source references:
- **Dispersion Measurement Function:** [`measure_dispersion`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/dispersion_analysis.hpp#L130-L190)
- **Dispersion Benchmark Harness:** [`run_dispersion`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/dispersion_analysis.hpp#L195-L368)
- **Exact Waveguide Solution:** [`GaussianSinusoidExact`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveFunctions.hpp#L210-L295) (accounting for waveguide effective phase speed $c_{\text{eff}} = \sqrt{c^2 + \frac{\pi^2}{k^2 L_y^2}}$)
- **Peak Finders:**
  - CG: [`WaveSolverMatFree::find_peak`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverMatFree.cpp#L476-L495)
  - DG: [`WaveSolverDG::find_peak`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverDG.cpp#L670-L708)
  - Theta: [`WaveSolverTheta::find_peak`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverTheta.cpp#L480-L534)

---

## 11. Summary & Architectural Comparison Table

The three wave solvers in this benchmark suite embody complementary design points across the spectrum of numerical partial differential equations:

| Feature / Metric | `WaveSolverMatFree` | `WaveSolverDG` | `WaveSolverTheta` |
| :--- | :--- | :--- | :--- |
| **Spatial Discretization** | Continuous Galerkin ($C^0$, Conforming) | Discontinuous Galerkin (SIPG) | Continuous Galerkin ($C^0$, Conforming) |
| **FE Basis & Nodes** | Gauss–Lobatto tensor product ($\text{FE\_Q}$) | Gauss–Lobatto tensor product ($\text{FE\_DGQ}$) | Standard Gauss–Lobatto ($\text{FE\_Q}$) |
| **Mass Matrix Treatment** | Lumped diagonal ($M_{ii} = w_i \|J\|$) | Block diagonal (diagonal with GL) | Full consistent sparse matrix |
| **Time Integrator** | Explicit Leapfrog (Störmer–Verlet) | Explicit Leapfrog (Störmer–Verlet) | Implicit Crank–Nicolson ($\theta = 1/4$) |
| **Stability Regime** | Conditionally stable ($\text{CFL} \le C / p^2$) | Conditionally stable ($\text{CFL} \le C_{\text{DG}} / p^2$) | Unconditionally stable ($\theta \ge 1/4$) |
| **Matrix Storage** | **$0$ bytes** (Matrix-Free operator) | **$0$ bytes** (Element-local face fluxes) | Global sparse matrices ($\mathbf{M}, \mathbf{K}, \mathbf{A}$) |
| **Linear Solvers** | None (Trivial vector division) | None (Trivial vector division) | Parallel GMRES / CG with AMG |
| **Energy Conservation** | **Exact machine precision** ($E_{\text{stag}}^{n+1/2}$) | **Exact machine precision** ($E_{\text{stag}}^{n+1/2}$) | Symplectic at $\theta=1/4$, dissipative at $\theta > 1/4$ |
| **Numerical Dissipation** | **Identically zero** | **Identically zero** | Zero for $\theta=1/4$; $>0$ for $\theta > 1/4$ |
| **Numerical Dispersion** | Subluminal lag ($\Delta t > 0$, $0.42\%$ at $p=4$) | Subluminal lag ($\Delta t > 0$, $0.68\%$ at $p=4$) | Superluminal lead ($\Delta t < 0$, consistent mass) |
| **MPI Scalability** | Excellent (Ghost DOF exchanges only) | Near-perfect (Face communication only) | Limited by global linear solver communication |
| **SIMD Vectorisation** | Full AVX-512 / AVX2 batching across cells | Element-local SIMD | Standard sparse BLAS vector operations |

### Overall Algorithmic Pipeline

```
                              Continuous Wave PDE:
                  ü + γ·u̇ - c²·Δu = f(x, t)    in  Ω × (0, T]
                                       │
                                       ▼
                         Weak Variational Formulation:
                 (ü, ϕ) + γ(u̇, ϕ) + c²(∇u, ∇ϕ) = (f, ϕ)   ∀ϕ ∈ V
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        Continuous Galerkin (CG)               Discontinuous Galerkin (DG)
        • u_h ∈ C^0, conforming                • u_h ∈ L^2, discontinuous
        • a_h(u, ϕ) = c²(∇u, ∇ϕ)_Ω             • a_h(u, ϕ) = (∇u, ∇ϕ) - face terms + penalties
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                          Gauss–Lobatto Mass Lumping:
               N_i(x_q) = δ_iq  ⟹  M_ij = w_i |J_K| δ_ij  (Diagonal Mass)
                                       │
                                       ▼
                       Second-Order Staggered Leapfrog:
        (1 + ½dtγ) M·u^{n+1} = 2 M·u^n - (1 - ½dtγ) M·u^{n-1} - dt² a_h(u^n, ϕ) + dt² f^n
                                       │
                                       ▼
                          Hardware-Accelerated Kernel:
         dealii::MatrixFree  ⟹  FEEvaluation::submit_gradient  ⟹  SIMD Vectorisation
                                       │
                                       ▼
                           Diagnostic Verification:
          ┌────────────────────────────┴────────────────────────────┐
          ▼                                                         ▼
  Exact Staggered Energy:                                 Numerical Dispersion:
  E_stag^{n+1/2} ≡ E_stag^{n-1/2}                         Direct 3-stage peak tracking
  Machine precision roundoff (10^{-14})                   Subluminal vs Superluminal separation
```

---

## 12. Convergence Analysis: Method of Manufactured Solutions (MMS) & Error Scaling

To verify the mathematical correctness and asymptotic convergence rates of the spatial and temporal discretizations, the solver suite implements the **Method of Manufactured Solutions (MMS)** via the `--mode convergence` driver.

### 12.1 Analytical Standing Wave Solution & Dirichlet Boundary Alignment

We construct a multi-dimensional standing wave on the hypercube domain $\Omega = (0, \pi)^d$ with wave speed $c = 1$:

$$\boxed{ u_{\text{exact}}(\mathbf{x}, t) = \cos(\sqrt{d}\, t) \prod_{j=1}^d \sin(x_j) }$$

**1. Boundary Conditions:**
For any coordinate $x_k \in \{0, \pi\}$, $\sin(x_k) = 0$. Consequently:
$$u_{\text{exact}}(\mathbf{x}, t) = 0 \quad \forall \mathbf{x} \in \partial\Omega, \; \forall t \ge 0,$$
matching the homogeneous Dirichlet boundary conditions identically.

**2. Initial Conditions:**
At $t = 0$:
$$u_0(\mathbf{x}) \equiv u_{\text{exact}}(\mathbf{x}, 0) = \prod_{j=1}^d \sin(x_j),$$
$$v_0(\mathbf{x}) \equiv \partial_t u_{\text{exact}}(\mathbf{x}, 0) = -\sqrt{d} \sin(0) \prod_{j=1}^d \sin(x_j) \equiv 0.$$
The system starts from a pure modal displacement at rest.

**3. Governing PDE Satisfaction:**
Differentiating twice in space:
$$\frac{\partial^2 u}{\partial x_k^2} = -\cos(\sqrt{d}\, t) \sin(x_k) \prod_{j \ne k} \sin(x_j) = - u_{\text{exact}}(\mathbf{x}, t),$$
$$\Delta u_{\text{exact}} = \sum_{k=1}^d \frac{\partial^2 u}{\partial x_k^2} = - d\, u_{\text{exact}}(\mathbf{x}, t).$$
Differentiating twice in time:
$$\frac{\partial^2 u_{\text{exact}}}{\partial t^2} = -(\sqrt{d})^2 \cos(\sqrt{d}\, t) \prod_{j=1}^d \sin(x_j) = - d\, u_{\text{exact}}(\mathbf{x}, t).$$
Substituting into the unforced wave operator ($c = 1, \gamma = 0$):
$$\partial_{tt} u_{\text{exact}} - \Delta u_{\text{exact}} = -d\, u_{\text{exact}} - (-d\, u_{\text{exact}}) \equiv 0.$$
The manufactured solution is an **exact, unforced eigenmode** of the continuous wave equation ($f(\mathbf{x}, t) \equiv 0$).

---

### 12.2 Discretization Error Decomposition: Spatial vs. Temporal Balance

For a conforming/non-conforming finite element discretization of polynomial degree $p$ combined with a second-order time integrator (Leapfrog or Crank–Nicolson), the global error at final time $T$ satisfies the a priori bound:

$$\|u(T) - u_h^n\|_{L^2(\Omega)} \le C_{\text{space}} h^{p+1} + C_{\text{time}} \Delta t^2,$$
$$|u(T) - u_h^n|_{H^1(\Omega)} \le C_{\text{space}}' h^p + C_{\text{time}}' \Delta t^2.$$

For our degree $p = 4$ elements:
- Spatial rate in $L^2$: $\mathcal{O}(h^5)$ ($\text{EOC} = 5.0$).
- Spatial rate in $H^1$: $\mathcal{O}(h^4)$ ($\text{EOC} = 4.0$).
- Temporal rate: $\mathcal{O}(\Delta t^2)$ ($\text{EOC} = 2.0$).

---

### 12.3 Scaling Laws: Strict Spatial Scaling ($\Delta t \propto h^{2.5}$) vs. CFL Scaling ($\Delta t \propto h$)

Because the spatial error decays as $\mathcal{O}(h^5)$ while the temporal error decays only as $\mathcal{O}(\Delta t^2)$, measuring convergence requires distinguishing between two scaling regimes:

#### Regime A: Standard CFL Scaling ($\Delta t = \text{CFL} \cdot h$)
When $\Delta t$ is coupled linearly to $h$:
$$\|u(T) - u_h^n\|_{L^2} \approx C_{\text{space}} h^5 + C_{\text{time}} (\text{CFL} \cdot h)^2 = C_{\text{space}} h^5 + \widetilde{C}_{\text{time}} h^2.$$
On coarse meshes ($h \approx 0.8$), the high-order spatial error $h^5$ may still contribute. However, on refined meshes ($h \le 0.2$), $h^5 \ll h^2$, and the temporal error **completely dominates**. The observed convergence rate asymptotes to:
$$\text{EOC}_{L^2} \to 2.00, \quad \text{EOC}_{H^1} \to 2.00.$$

#### Regime B: Strict Spatial Scaling ($\Delta t \propto h^{2.5}$)
To isolate the true spatial convergence order $\mathcal{O}(h^5)$, the temporal error must shrink at least as fast as the spatial error:
$$\Delta t^2 \sim h^5 \implies \mathbf{\Delta t \propto h^{2.5}}.$$
In the code (`src/main.cpp`):
```cpp
const double h_ref = numbers::PI / 4.0;       // Level 2 reference
const double dt_ref = cfl * h_ref;
dt = dt_ref * std::pow(h_cell / h_ref, 2.5);  // dt ∝ h^2.5
```
Under this scaling, halving the mesh size ($h \to h/2$) reduces $\Delta t$ by $2^{2.5} \approx 5.657$, which reduces $\Delta t^2$ by $32 = 2^5$. Both spatial and temporal errors decay by $32\times$, allowing the optimal rates to be observed:
$$\mathbf{\text{EOC}_{L^2} \approx 5.00}, \quad \mathbf{\text{EOC}_{H^1} \approx 4.00 - 5.00}.$$

---

### 12.4 Startup Formulation: Fictitious Step $u^{-1} = u_{\text{exact}}(-\Delta t)$

Explicit Leapfrog requires a fictitious past level $u^{-1}$ to start. Standard Taylor expansion:
$$u^{-1} = u_0 - \Delta t\, v_0 + \frac{\Delta t^2}{2} a_0$$
incurs a local truncation error of $\mathcal{O}(\Delta t^3)$. When testing high-order spatial convergence ($p=4$), even an $\mathcal{O}(\Delta t^3)$ startup error can pollute the $\mathcal{O}(h^5)$ spatial error on fine grids.

To guarantee that startup error is zero, the solver interface accepts an optional exact past solution:
```cpp
StandingWaveExact<dim> u_prev(-dt);
solver->set_initial_conditions(u0, v0, &u_prev);
```
Inside the solvers (`WaveSolverMatFree`, `WaveSolverDG`), `u_prev` directly initializes `old_solution_` via $L^2$ projection:
```cpp
if (u_prev != nullptr) {
    VectorTools::project(dof_handler_, constraints_,
                         QGaussLobatto<dim>(fe_degree_ + 1),
                         *u_prev, old_solution_);
}
```

---

### 12.5 Fine-Grid Roundoff Saturation Barrier

In explicit second-order Leapfrog time-stepping:
$$u^{n+1} = 2u^n - u^{n-1} - \Delta t^2 \mathbf{M}^{-1} \mathbf{K} u^n,$$
local roundoff errors $\epsilon_{\text{mach}} \approx 10^{-16}$ accumulate over $N = T/\Delta t$ steps. The global roundoff error scales as:
$$E_{\text{roundoff}} \sim \frac{\epsilon_{\text{mach}}}{\Delta t^2}.$$

Under strict spatial scaling ($\Delta t \propto h^{2.5}$):
- **Level 2** ($h = \pi/4$): $\Delta t \approx 2 \times 10^{-2}$ ($50$ steps), error $\approx 7 \times 10^{-5} \gg E_{\text{roundoff}}$.
- **Level 3** ($h = \pi/8$): $\Delta t \approx 3.5 \times 10^{-3}$ ($289$ steps), error $\approx 2 \times 10^{-6} \gg E_{\text{roundoff}}$.
- **Level 4** ($h = \pi/16$): $\Delta t \approx 6 \times 10^{-4}$ ($1,600$ steps), error $\approx 7 \times 10^{-8} \gg E_{\text{roundoff}}$.
- **Level 5** ($h = \pi/32$): $\Delta t \approx 10^{-4}$ ($9,000$ steps), theoretical error $\approx 2 \times 10^{-9}$, while $E_{\text{roundoff}} \approx 10^{-8}$.
- **Level 6** ($h = \pi/64$): $\Delta t \approx 1.8 \times 10^{-5}$ ($55,000$ steps for CG, $>650,000$ for DG), $E_{\text{roundoff}} \approx 3 \times 10^{-7}$.

At Level 5 and 6, the theoretical spatial error drops below the hardware floating-point roundoff floor, causing the apparent EOC to saturate. Therefore:
- **Spatial scaling** defaults to refinements `{2, 3, 4}` where errors are cleanly above roundoff.
- **CFL scaling** safely includes refinements `{2, 3, 4, 5}`, where step counts remain modest and $\text{EOC} = 2.00$ is achieved.

---

### 12.6 Verification Results Across the Suite (CG, DG, Theta)

#### 1. Strict Spatial Scaling ($\Delta t \propto h^{2.5}$, $p = 4$, $T = 1.0$)
| Solver | Refinement | $h$ | DOFs | $L^2$ Error | $L^2$ EOC | $H^1$ Error | $H^1$ EOC | Theoretical Spatial Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Matrix-Free CG** | 2 | $0.7854$ | 289 | $7.170 \times 10^{-5}$ | — | $1.048 \times 10^{-4}$ | — | — |
| | 3 | $0.3927$ | 1,089 | $2.197 \times 10^{-6}$ | **5.03** | $3.516 \times 10^{-6}$ | **4.90** | $\mathcal{O}(h^5) / \mathcal{O}(h^4)$ |
| | 4 | $0.1963$ | 4,225 | $6.882 \times 10^{-8}$ | **5.00** | $1.411 \times 10^{-7}$ | **4.64** | $\mathcal{O}(h^5) / \mathcal{O}(h^4)$ |
| **Matrix-Free DG (SIPG)** | 2 | $0.7854$ | 400 | $2.735 \times 10^{-6}$ | — | $4.774 \times 10^{-5}$ | — | — |
| | 3 | $0.3927$ | 1,600 | $1.037 \times 10^{-7}$ | **4.72** | $4.891 \times 10^{-6}$ | **3.29** | $\mathcal{O}(h^5) / \mathcal{O}(h^4)$ |
| | 4 | $0.1963$ | 6,400 | $3.921 \times 10^{-9}$ | **4.73** | $1.598 \times 10^{-7}$ | **4.94** | $\mathcal{O}(h^5) / \mathcal{O}(h^4)$ |
| **Theta-Scheme (CN)** | 2 | $0.7854$ | 289 | $1.007 \times 10^{-2}$ | — | $1.424 \times 10^{-2}$ | — | — |
| | 3 | $0.3927$ | 1,089 | $4.347 \times 10^{-4}$ | **4.53** | $6.147 \times 10^{-4}$ | **4.53** | $\mathcal{O}(h^5) / \mathcal{O}(h^4)$ |
| | 4 | $0.1963$ | 4,225 | $1.376 \times 10^{-5}$ | **4.98** | $2.004 \times 10^{-5}$ | **4.94** | $\mathcal{O}(h^5) / \mathcal{O}(h^4)$ |

#### 2. Standard CFL Scaling ($\Delta t \propto h$, $p = 4$, $T = 1.0$)
| Solver | Refinement | $h$ | DOFs | $L^2$ Error | $L^2$ EOC | $H^1$ Error | $H^1$ EOC | Theoretical Temporal Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Matrix-Free CG** | 3 | $0.3927$ | 1,089 | $1.775 \times 10^{-5}$ | **2.01** | $2.515 \times 10^{-5}$ | **2.06** | $\mathcal{O}(\Delta t^2)$ |
| | 4 | $0.1963$ | 4,225 | $4.415 \times 10^{-6}$ | **2.01** | $6.245 \times 10^{-6}$ | **2.01** | $\mathcal{O}(\Delta t^2)$ |
| | 5 | $0.0982$ | 16,641 | $1.101 \times 10^{-6}$ | **2.00** | $1.557 \times 10^{-6}$ | **2.00** | $\mathcal{O}(\Delta t^2)$ |
| **Matrix-Free DG (SIPG)** | 3 | $0.3927$ | 1,600 | $2.920 \times 10^{-7}$ | **3.23** | $5.003 \times 10^{-6}$ | **3.25** | $\mathcal{O}(\Delta t^2)$ |
| | 4 | $0.1963$ | 6,400 | $6.888 \times 10^{-8}$ | **2.08** | $1.415 \times 10^{-7}$ | **5.14** | $\mathcal{O}(\Delta t^2)$ |
| | 5 | $0.0982$ | 25,600 | $1.700 \times 10^{-8}$ | **2.02** | $2.784 \times 10^{-8}$ | **2.35** | $\mathcal{O}(\Delta t^2)$ |
| **Theta-Scheme (CN)** | 3 | $0.3927$ | 1,089 | $3.014 \times 10^{-3}$ | **1.74** | $4.263 \times 10^{-3}$ | **1.74** | $\mathcal{O}(\Delta t^2)$ |
| | 4 | $0.1963$ | 4,225 | $8.287 \times 10^{-4}$ | **1.86** | $1.172 \times 10^{-3}$ | **1.86** | $\mathcal{O}(\Delta t^2)$ |
| | 5 | $0.0982$ | 16,641 | $2.175 \times 10^{-4}$ | **1.93** | $3.076 \times 10^{-4}$ | **1.93** | $\mathcal{O}(\Delta t^2)$ |

---

## 13. Neumann Boundary Conditions & Acoustic Sound Waves (DG Alignment)

### 13.1 Physics of Neumann Boundary Conditions: Sound-Hard Acoustic Walls

In linear acoustics, the acoustic pressure perturbation $p(\mathbf{x}, t)$ satisfies the scalar wave equation:
$$\frac{\partial^2 p}{\partial t^2} - c^2 \Delta p = 0 \quad \text{in } \Omega \times (0, T].$$
The acoustic fluid velocity $\mathbf{v}_{\text{fluid}}$ is coupled to the pressure gradient through Euler's linearized momentum equation:
$$\rho_0 \frac{\partial \mathbf{v}_{\text{fluid}}}{\partial t} = -\nabla p.$$
At a rigid, impermeable acoustic boundary (such as the rigid walls of an organ pipe, a concert hall, or an acoustic resonator), the normal component of fluid velocity must vanish identically:
$$\mathbf{v}_{\text{fluid}} \cdot \mathbf{n} = 0 \quad \implies \quad \nabla p \cdot \mathbf{n} = 0 \quad \text{on } \partial\Omega.$$
This boundary condition is known as a **sound-hard wall** (homogeneous Neumann boundary condition). Unlike homogeneous Dirichlet boundaries ($p = 0$, representing an open pressure-release surface), sound-hard walls reflect acoustic waves with no phase inversion ($+1$ reflection coefficient), causing constructive acoustic interference, standing pressure antinodes, and resonance phenomena.

---

### 13.2 Mathematical Formulation Across Solvers (CG, DG, Theta)

#### 1. Continuous Galerkin (`WaveSolverMatFree` and `WaveSolverTheta`)
In conforming finite element methods ($V_h \subset H^1(\Omega)$), integration by parts of the Laplacian yields:
$$\int_\Omega (-\Delta u) v \, d\mathbf{x} = \int_\Omega \nabla u \cdot \nabla v \, d\mathbf{x} - \int_{\partial\Omega} (\nabla u \cdot \mathbf{n}) v \, ds.$$
When homogeneous Neumann conditions $\nabla u \cdot \mathbf{n} = 0$ are prescribed:
- The boundary integral vanishes identically: $-\int_{\partial\Omega} (\nabla u \cdot \mathbf{n}) v \, ds = 0$.
- Homogeneous Neumann is therefore a **natural boundary condition** for conforming Galerkin methods.
- In implementation, boundary DoFs are left unconstrained (no calls to `VectorTools::interpolate_boundary_values` on $\partial\Omega$). The spatial operator $\mathbf{K}$ naturally computes the correct weak Laplacian without any boundary penalty or modification.

#### 2. Discontinuous Galerkin (`WaveSolverDG`, SIPG Formulation)
In Discontinuous Galerkin discretizations ($V_h \not\subset H^1(\Omega)$), functions are discontinuous across all faces. The global SIPG bilinear form partitions into volume terms, interior face terms, and boundary face terms:
$$a_h(u, v) = \sum_{K} \int_K \nabla u \cdot \nabla v \, d\mathbf{x} + \sum_{F \in \mathcal{F}_I} a_h^F(u, v) + \sum_{F \in \mathcal{F}_B} a_h^{\partial K}(u, v).$$
On interior faces $F \in \mathcal{F}_I$:
$$a_h^F(u, v) = \int_F \left( -\{\nabla u\} \cdot [v] - \{\nabla v\} \cdot [u] + \sigma [u][v] \right) ds.$$
On boundary faces $F \in \mathcal{F}_B$:
- **Dirichlet Boundary** ($\Gamma_D, u = 0$): Enforced weakly by penalizing deviations from $0$:
  $$a_h^{\partial K}(u, v) = \int_F \left( -(\nabla u \cdot \mathbf{n}) v - (\nabla v \cdot \mathbf{n}) u + 2\sigma u v \right) ds.$$
- **Neumann Boundary** ($\Gamma_N, \nabla u \cdot \mathbf{n} = 0$): By standard variational calculus, the integration by parts on cell $K$ gives:
  $$\int_K (-\Delta u) v \, d\mathbf{x} = \int_K \nabla u \cdot \nabla v \, d\mathbf{x} - \int_{\partial K \cap \Gamma_N} (\nabla u \cdot \mathbf{n}) v \, ds.$$
  Since $\nabla u \cdot \mathbf{n} = 0$, the boundary flux is **identically zero**. Furthermore, because $u$ is unrestricted on $\Gamma_N$, **no penalty term is required**.
- **Implementation in `WaveOperationDG`**:
  ```cpp
  template <int dim, int fe_degree>
  void WaveOperationDG<dim, fe_degree>::local_apply_boundary_face(...) const
  {
    if (boundary_type_ == WaveSolverBase<dim>::BoundaryType::Neumann)
      return; // Boundary flux is identically zero!
    // ... Dirichlet SIPG penalty evaluation ...
  }
  ```
  Similarly, in `compute_energy()`, boundary face penalty integrals are skipped when `boundary_type_ == BoundaryType::Neumann`, preserving the exact discrete Hamiltonian structure.

---

### 13.3 Exact Acoustic Sound Wave Mode (MMS for Neumann BCs)

To perform rigorous Method of Manufactured Solutions (MMS) verification for Neumann boundary conditions, we introduce the **acoustic cavity standing wave mode** on $\Omega = (0, \pi)^d$:
$$u_{\text{exact}}(\mathbf{x}, t) = \cos(\omega t) \prod_{j=0}^{d-1} \cos(x_j), \quad \omega = \sqrt{d}, \quad c = 1.$$

#### Mathematical Verification:
1. **Governing Equation ($f \equiv 0$)**:
   $$\frac{\partial^2 u}{\partial t^2} = -\omega^2 \cos(\omega t) \prod_{j=0}^{d-1} \cos(x_j) = -d \, u(\mathbf{x}, t),$$
   $$\Delta u = \sum_{k=0}^{d-1} \frac{\partial^2 u}{\partial x_k^2} = \sum_{k=0}^{d-1} \left( -\cos(x_k) \cos(\omega t) \prod_{j \neq k} \cos(x_j) \right) = -d \, u(\mathbf{x}, t).$$
   Therefore:
   $$\frac{\partial^2 u}{\partial t^2} - \Delta u = -d u - (-d u) = 0 \quad (\text{exact solution with } f = 0).$$

2. **Neumann Boundary Condition**:
   The gradient is given by:
   $$\frac{\partial u}{\partial x_k} = -\sin(x_k) \cos(\omega t) \prod_{j \neq k} \cos(x_j).$$
   On the boundary planes $x_k = 0$ and $x_k = \pi$:
   $$\left. \frac{\partial u}{\partial x_k} \right|_{x_k = 0} = 0 \quad \text{and} \quad \left. \frac{\partial u}{\partial x_k} \right|_{x_k = \pi} = 0.$$
   Because the normal vector $\mathbf{n}$ is parallel to the coordinate axis $\mathbf{e}_k$ on each facet of $\Omega = (0, \pi)^d$, we have:
   $$\nabla u \cdot \mathbf{n} \equiv 0 \quad \text{on all faces of } \partial\Omega.$$

3. **Initial Conditions**:
   $$u(\mathbf{x}, 0) = \prod_{j=0}^{d-1} \cos(x_j), \quad \frac{\partial u}{\partial t}(\mathbf{x}, 0) = 0.$$

This exact mode allows running the full MMS convergence suite (`--bc neumann`) across all three solvers to verify both $\mathcal{O}(h^5)$ spatial EOC and $\mathcal{O}(\Delta t^2)$ temporal EOC under sound-hard wall boundary conditions.

---

### 13.4 Acoustic Sound Pulse & Discontinuous Galerkin Advantages

In addition to resonant standing waves, the benchmark provides an acoustic sound pulse initial condition (`--wave pulse`):
$$u_0(\mathbf{x}) = A \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}_0\|^2}{2\sigma_0^2}\right), \quad v_0(\mathbf{x}) = 0.$$

#### Why This Problem Highlights Discontinuous Galerkin (DG):
1. **Acoustic Shock and Sharp Gradient Tracking**:
   High-frequency sound waves and acoustic pulses exhibit steep wavefronts. Conforming CG methods can suffer from spurious Gibbs oscillations near steep gradients unless artificial stabilization is introduced. The Discontinuous Galerkin formulation (`WaveSolverDG`) permits jumps across element boundaries and dissipates unresolvable high frequencies via interior penalty fluxes without polluting smooth regions.
2. **Compact Domain and Rigid Wall Reflections**:
   When the acoustic pulse expands and impinges upon the Neumann boundaries, it reflects back into the domain. DG handles reflecting boundary faces naturally with zero flux, cleanly resolving the reflected wavefronts.
3. **High-Order Polynomial Efficiency ($p=4$)**:
   The tensor-product Gauss-Lobatto element with $p=4$ provides $(p+1)^d = 25$ DoFs per quad (125 DoFs per hex), achieving minimal numerical dispersion over hundreds of wave periods.

---

### 13.5 Command-Line Interface & Usage Examples

The benchmark executable and test script provide flexible flags for boundary conditions and wave selection:

```bash
# 1. MMS Convergence Study with Neumann Boundary Conditions (dim=2, all solvers)
./build/WaveBenchmark --mode convergence --solver all --bc neumann --dim 2 --time 1.0

# 2. Benchmark with Acoustic Standing Wave and Sound-Hard Walls
./build/WaveBenchmark --mode bench --solver all --bc neumann --wave acoustic --dim 2 --refine 6 --time 5.0

# 3. Benchmark with Acoustic Sound Pulse in a Cavity (Observing Wall Reflections)
./build/WaveBenchmark --mode bench --solver all --bc neumann --wave pulse --dim 2 --refine 6 --time 5.0 --output

# 4. Automated Verification via Unified Test Script
./test_script.sh --step convergence --bc neumann --time 1.0
./test_script.sh --step bench --bc neumann --wave acoustic --time 5.0
```

---

## 14. Non-Homogeneous (Inhomogeneous) Boundary Conditions

To rigorously test boundary handling under dynamic external excitation, the codebase supports **non-homogeneous (time-dependent)** boundary conditions (`--non-homogeneous` / `--inhomogeneous`):
1. **Non-Homogeneous Dirichlet Boundary Condition**:
   $$u(\mathbf{x}, t) = g_D(\mathbf{x}, t) \quad \text{on } \partial\Omega \times (0, T].$$
2. **Non-Homogeneous Neumann Boundary Condition**:
   $$\nabla u(\mathbf{x}, t) \cdot \mathbf{n} = g_N(\mathbf{x}, t) \quad \text{on } \partial\Omega \times (0, T].$$

### 14.1 Variational Formulations

#### 1. Discontinuous Galerkin (SIPG Formulation)
In Discontinuous Galerkin, boundary conditions are incorporated purely through weak boundary face integrals without altering nodal degrees of freedom or changing sparsity patterns.

- **Non-Homogeneous Dirichlet**:
  When $u|_{\partial\Omega} = g_D$, the standard SIPG boundary face operator penalizes deviations $(u - g_D)$ from the prescribed trace:
  $$a_h^{\partial K}(u, v) = \int_{\partial K \cap \Gamma_D} \left( -(\nabla u \cdot \mathbf{n}) v - (\nabla v \cdot \mathbf{n}) (u - g_D) + 2\sigma (u - g_D) v \right) ds.$$
  In [`WaveOperationDG::local_apply_boundary_face`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverDG.cpp#L150-L195), evaluating $g_D(\mathbf{x}_q, t^n)$ at face quadrature points produces:
  ```cpp
  const auto u_val_diff = u_val - g_val;
  fe_eval.submit_value(2.0 * penalty * u_val_diff - u_grad_n, q);
  fe_eval.submit_normal_derivative(-u_val_diff, q);
  ```
- **Non-Homogeneous Neumann**:
  Integration by parts on element $K$ yields $-\int_{\partial K \cap \Gamma_N} (\nabla u \cdot \mathbf{n}) v \, ds = -\int_{\partial K \cap \Gamma_N} g_N v \, ds$.
  Because $u$ is unconstrained, the penalty is identically zero:
  $$\text{Flux}(v) = -g_N(\mathbf{x}, t) v.$$
  In [`WaveOperationDG::local_apply_boundary_face`](file:///Users/feliceferrara/amsc_mk_25-shared-folder/pde/Wave-function-nmpde-projects/src/WaveSolverDG.cpp#L196-L225):
  ```cpp
  const auto normal_grad_exact = exact_solution_->gradient(q_point, current_time_);
  const auto g_N = normal_grad_exact * normal;
  fe_eval.submit_value(-g_N, q);
  fe_eval.submit_normal_derivative(VectorizedArray<double>(), q);
  ```

#### 2. Continuous Galerkin (CG Formulation)
- **Non-Homogeneous Dirichlet**:
  In conforming Continuous Galerkin (`WaveSolverMatFree`), Dirichlet boundary DoFs are not locked to zero. At each time step $t^n$, the boundary DoFs of $u^n$ are explicitly updated with the exact trace $g_D(\mathbf{x}, t^n)$ using `VectorTools::interpolate_boundary_values`. The interior nodes are then updated via the leapfrog stencil:
  $$u^{n+1} = 2u^n - u^{n-1} - \Delta t^2 \mathbf{M}_L^{-1} \mathbf{K} u^n.$$
- **Non-Homogeneous Neumann**:
  In conforming CG, non-homogeneous Neumann conditions require assembling the boundary surface load vector $\int_{\partial\Omega} g_N v \, ds$. Since the matrix-free CG operator uses a cell-only integration loop (`cell_loop`), CG with non-homogeneous Neumann is skipped with a descriptive message.

---

### 14.2 Complementary Manufactured Solutions for MMS

To test non-homogeneous boundaries without trivial zero traces, the convergence harness selects exact solutions whose boundary values are non-zero:

1. **For Non-Homogeneous Dirichlet MMS**:
   Uses the acoustic standing wave $u(\mathbf{x}, t) = \cos(\sqrt{d} t) \prod_{j=0}^{d-1} \cos(x_j)$.
   On $\partial(0, \pi)^d$, $\cos(0) = 1$ and $\cos(\pi) = -1$, giving a non-trivial time-dependent boundary trace $g_D(\mathbf{x}, t) \neq 0$.

2. **For Non-Homogeneous Neumann MMS**:
   Uses the standing wave $u(\mathbf{x}, t) = \cos(\sqrt{d} t) \prod_{j=0}^{d-1} \sin(x_j)$.
   The normal gradient satisfies:
   $$\left. \frac{\partial u}{\partial x_k} \right|_{x_k = 0} = \cos(\sqrt{d} t) \prod_{j \neq k} \sin(x_j) \neq 0, \quad \left. \frac{\partial u}{\partial x_k} \right|_{x_k = \pi} = -\cos(\sqrt{d} t) \prod_{j \neq k} \sin(x_j) \neq 0,$$
   generating a non-trivial boundary flux $g_N(\mathbf{x}, t) \neq 0$.

---

### 14.3 Empirical Verification Results ($p=4$, $\Omega = (0, \pi)^2$, $T=1.0$)

#### Non-Homogeneous Dirichlet MMS ($u|_{\partial\Omega} = \cos(\sqrt{2}t)\cos(x)\cos(y)$)

- **Continuous Galerkin (`WaveSolverMatFree`, $p=4$)**:
  | Refinement | $h$ | DOFs | $L^2$ Error | $L^2$ EOC | $H^1$ Error | $H^1$ EOC |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | 2 | $7.854 \times 10^{-1}$ | 289 | $2.703 \times 10^{-5}$ | — | $8.162 \times 10^{-5}$ | — |
  | 3 | $3.927 \times 10^{-1}$ | 1,089 | $8.384 \times 10^{-7}$ | **5.01** | $2.900 \times 10^{-6}$ | **4.81** |
  | 4 | $1.964 \times 10^{-1}$ | 4,225 | $2.631 \times 10^{-8}$ | **4.99** | $1.268 \times 10^{-7}$ | **4.52** |

- **Discontinuous Galerkin (`WaveSolverDG`, SIPG, $p=4$)**:
  | Refinement | $h$ | DOFs | $L^2$ Error | $L^2$ EOC | $H^1$ Error | $H^1$ EOC |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | 2 | $7.854 \times 10^{-1}$ | 400 | $2.252 \times 10^{-6}$ | — | $5.060 \times 10^{-5}$ | — |
  | 3 | $3.927 \times 10^{-1}$ | 1,600 | $1.007 \times 10^{-7}$ | **4.48** | $4.550 \times 10^{-6}$ | **3.48** |
  | 4 | $1.964 \times 10^{-1}$ | 6,400 | $3.319 \times 10^{-9}$ | **4.92** | $1.585 \times 10^{-7}$ | **4.84** |

#### Non-Homogeneous Neumann MMS ($\nabla u \cdot \mathbf{n}|_{\partial\Omega} = \pm \cos(\sqrt{2}t)\sin(x)\sin(y)$)

- **Discontinuous Galerkin (`WaveSolverDG`, SIPG, $p=4$)**:
  | Refinement | $h$ | DOFs | $L^2$ Error | $L^2$ EOC | $H^1$ Error | $H^1$ EOC |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | 2 | $7.854 \times 10^{-1}$ | 400 | $3.641 \times 10^{-6}$ | — | $4.287 \times 10^{-5}$ | — |
  | 3 | $3.927 \times 10^{-1}$ | 1,600 | $1.060 \times 10^{-7}$ | **5.10** | $3.643 \times 10^{-6}$ | **3.56** |
  | 4 | $1.964 \times 10^{-1}$ | 6,400 | $4.377 \times 10^{-9}$ | **4.60** | $1.425 \times 10^{-7}$ | **4.68** |

---

### 14.4 Command-Line Usage

```bash
# 1. Non-homogeneous Dirichlet convergence study
./build/WaveBenchmark --mode convergence --solver all --bc dirichlet --non-homogeneous --dim 2 --time 1.0

# 2. Non-homogeneous Neumann convergence study (DG solver)
./build/WaveBenchmark --mode convergence --solver dg --bc neumann --non-homogeneous --dim 2 --time 1.0

# 3. Via unified test script
./test_script.sh --step convergence --bc dirichlet --non-homogeneous --time 1.0
./test_script.sh --step convergence --bc neumann --non-homogeneous --time 1.0
```

