# posteriori4pseudoproblems
A Posteriori Error Bounds for the BDF-2 Method Applied to Pseudo-Parabolic Partial Differential Equations

This repository contains the code used to compute the numerical results of [1]. Its
purpose is to make these results reproducible. The a posteriori error bounds of [1] hold
for time discretizations of second order in general; the numerical experiments, and thus
this code, use the BDF-2 method.

## The numerical example

The code solves the pseudo-parabolic problem

```math
\begin{align*}
-u_{xxt} + (5x+6)\,u_t - u_{xx} - \mathrm{e}^{-x}\,u &= \mathrm{e}^{2t} + \cos\bigl(\pi (x+t)^2\bigr) \quad &&\text{on } (-1,1) \times (0,2] \\
u(x,0) &= \sin(x\pi) \quad &&\text{on } [-1,1] \\
u(x,t) &= 0 \quad &&\text{for } (x,t) \in \{-1, 1\} \times [0,2],
\end{align*}
```

i.e. $L u_t + M u = F$ with $Lu = -u_{xx} + a u$, $Mu = -u_{xx} + c u$, $a(x) = 5x+6$ and
$c(x) = -\mathrm{e}^{-x}$.

> [!NOTE]
> The problem stated in the current arXiv version of [1] does not coincide with the problem
> above, which is the one the numerical results were computed for. The arXiv version will
> be updated accordingly.

The problem is discretized by

- the BDF-2 method in time, where the starting value $u^1_h$ is computed by the backward
  Euler method,
- $P_1$ finite elements in space for the $L^2$ estimator and $P_2$ finite elements for the
  $H^1$ estimator, on uniform meshes with $M = N = 2^p$, $p = 6, \dots, 13$,
- the Galerkin projection of $u_0$ with respect to $L$ ($L^2$ estimator) or the $L^2$
  projection of $u_0$ ($H^1$ estimator) as initial approximation $u^0_h$.

The elliptic part of the estimator is computed by means of a higher-order approximation,
using $P_2$ ($L^2$ estimator) or $P_3$ ($H^1$ estimator) finite elements. All spatial
integrals are approximated by the Gauss–Lobatto formula with 4 nodes. Since the exact
solution is unknown, the error is measured against a reference solution computed by the
dG(2) method in time and a spectral Galerkin method in space.

## Project Structure
```text
.
├── results                             # The computed results as LaTeX tables
└── src
    └── posteriori4pseudoproblems       # Main Python package
        ├── pseudo_parabolic_pde_class  # The PDE and its generic constants
        ├── discretization_dataclasses  # Spatial and temporal discretization parameters
        ├── quadrature                  # Gauss–Lobatto quadrature
        ├── interval_transformation     # Affine maps between intervals
        ├── legendre_galerkin           # P_k-FEM and spectral reference functions
        ├── matrix_assembly             # Coefficient matrices of L and M
        ├── projection_operators        # L^2 and Galerkin projections
        ├── time_discretization_methods # BDF-FEM and dG(2)-spectral time stepping
        ├── reference_solution          # dG(2)-spectral reference solution
        ├── norms                       # L^2 and H^1 norms
        ├── utilities                   # Divided differences and Horner scheme
        ├── error_bound                 # The a posteriori error estimator
        └── numerical_examples          # The numerical example of [1]
```

## Installation

**Requirements**
- Python 3.13 or newer
- Docker (optional)

The project can be used in two different ways.

**Installation on your system or in a virtual environment:**
Install the required packages via
```
pip3 install -r requirements.txt
```

**Using Docker or Devcontainer:**
A Dockerfile as well as a .devcontainer.json file are provided. To run the code inside a
Docker container, build a Docker image via

```
docker build -t posteriori4pseudoproblems:latest .
```

If you use Visual Studio Code, you can run the code inside a Devcontainer environment:

```
code posteriori4pseudoproblems.code-workspace
```

## Reproducing the results

From the root of the repository, run

```
PYTHONPATH=src python3 -m posteriori4pseudoproblems.numerical_examples.exact_solution_unknown_example
```

This computes the errors, the estimators and their components for both the $L^2$ and the
$H^1$ estimator and writes them as LaTeX tables to

- `results/exact_solution_unknown_example/bdf2_l2.txt`
- `results/exact_solution_unknown_example/bdf2_h1.txt`

The runs on the finest meshes take a while.

## Tests

This repository deliberately contains no test suite: its sole purpose is to reproduce the
results of [1]. The code was validated by reproducing the results of the original
implementation used for [1].

## Acknowledgments
The code was restructured and documented with the assistance of Claude (Anthropic), an AI
assistant. The research question, the experiment design, the mathematical analysis, and
all final decisions are my own.

## Literature
[1] Ossadnik, M. and Linß, T.
    A posteriori error bounds for pseudo-parabolic equations using $C_0$ semigroups.
    [arXiv:2606.20073](https://arxiv.org/abs/2606.20073), 2026.
