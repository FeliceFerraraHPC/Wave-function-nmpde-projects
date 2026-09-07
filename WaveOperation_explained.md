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

---
### TO DO:
**Numerical Dissipation: Energy Conservation Analysis**

The continuous wave equation preserves the total Hamiltonian energy:

$$E(t) = \frac{1}{2} \Vert{}\partial_t u\Vert{}_{L^2(\Omega)}^2 + \frac{1}{2} \Vert{}\nabla u\Vert{}_{L^2(\Omega)}^2 = \text{const}$$

- Implement an energy calculation routine at each time step $t^n$:
    
      $$E^n = \frac{1}{2} \int_\Omega \left(\frac{u^{n+1} - u^n}{\Delta t}\right)^2 d\mathbf{x} + \frac{1}{2} a_h(u^n, u^{n+1})$$
    
- **Compare CG vs. DG:** Plot $E(t) / E(0)$ across several hundred wave periods.
    
- **Finding to Discuss:** Both the Leapfrog time-stepper and the symmetric spatial operators (CG and SIPG) are symplectic/conservative, exhibiting **zero numerical dissipation** (energy remains bounded and fluctuates stably around a constant without decaying).
    
**Numerical Dispersion Analysis**

Dispersion is the primary spatial error mechanism in high-frequency wave propagation.

- Run a traveling wave packet or sinusoidal pulse over a long simulation time ($T_{\text{final}} \gg 1$).
        
- **Phase Error:** Measure the shift between numerical wave peaks and analytical peaks.
    
- **Parametric Study:**
      
    - Compare the phase lag of low-order ($p=1, 2$) vs. high-order ($p=4, 6$) elements under a matched total number of DOFs.
            
    - Compare the dispersion behavior of CG vs. DG. High-order DG should show significantly reduced phase dispersion over long distances.
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

## Summary

```
          PDE (strong form)
              ↓  multiply by ϕ, integrate by parts
          Weak form:  (ü, ϕ) + c²(∇u, ∇ϕ) + γ(u̇, ϕ) = (f, ϕ)
              ↓  FE expansion  u_h = Σ uⱼ Nⱼ
          ODE system:  M·ü + c²·K·u + γ·M·u̇ = f(t)
              ↓  central differences for ü and u̇ (leapfrog + Crank–Nicolson damping)
          Algebraic update:
              (1 + ½dtγ)M·u^{n+1} = 2M·u^n - (1-½dtγ)M·u^{n-1} - dt²c²K·u^n + dt²f^n
              ↓  Gauss–Lobatto mass lumping:
                   N_i(x_q) = δ_iq  →  M_ij = w_i·δ_ij  →  M diagonal  →  trivial inversion
              ↓  K·u^n evaluated matrix-free via submit_gradient
          Code:  local_apply → cell_loop → dst.scale(inv_effective_mass)  →  u^{n+1}
```
