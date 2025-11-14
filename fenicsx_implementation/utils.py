"""
Generates test data for a time-dependent convection-diffusion-reaction problem.

This script defines a function `generate_data_bump` that simulates the
evolution of a scalar concentration 'c' for a single time step. The initial
condition 'c0' is a Gaussian-like bump, which is then advected by a
constant velocity field 'phi' and subject to diffusion 'D' and reaction 'r'.

The primary use is to create a "ground truth" dataset (c0, c1, phi) for
inverse problems, such as optical flow or parameter estimation.

The generated data is saved to disk in ADIOS (for checkpointing/reloading) and
VTX (for visualization) formats.
"""

from pathlib import Path
import numpy as np
import dolfinx
import dolfinx.fem.petsc
import adios4dolfinx
import ufl


def generate_data_bump(
    mesh, g, output: Path, D=0.0, r=0.0, Function=dolfinx.fem.Function
):
    """
    Generates and saves data for a convection-diffusion-reaction test case.

    This function creates an initial state 'c0' (a Gaussian-like bump) and
    a constant convection field 'phi' = (g, g). It then solves a single
    implicit time step (Backward Euler) of the convection-diffusion-reaction
    equation to find the evolved state 'c1' at t=tau.

    Equation:
      c_t + div(phi * c) - D * div(grad(c)) + r * c = 0

    The function is idempotent: if the output file 'data.bp' already exists
    in the 'output' directory, it loads and returns the data from the file
    instead of re-computing it.

    Args:
        mesh: The dolfinx.mesh.Mesh object.
        g: The magnitude of the constant convection field (velocity).
        output: A pathlib.Path to the output directory.
        D: The diffusion coefficient (default: 0.0).
        r: The reaction coefficient (default: 0.0).
        Function: The dolfinx.fem.Function class (used for type hinting).

    Returns:
        A tuple (c0, c1, phi, tau):
        - c0: dolfinx.fem.Function (CG1), the initial state (t=0).
        - c1: dolfinx.fem.Function (CG1), the final state (t=tau).
        - phi: dolfinx.fem.Function (CG1 Vector), the convection field.
        - tau: float, the time step used (hard-coded to 1.0).
    """
    # --- 1. Setup Function Spaces and Time Step ---
    Q = dolfinx.fem.functionspace(mesh, ("CG", 1))  # Scalar space for concentration 'c'
    V = dolfinx.fem.functionspace(
        mesh, ("CG", 1, (mesh.topology.dim,))
    )  # Vector space for velocity 'phi'
    tau = 1.0  # Time step

    # --- 2. Caching: Check if data already exists ---
    # If the ADIOS file exists, load data from it instead of re-running.
    # This is useful for re-running subsequent inversion scripts.
    path = output / "data.bp"
    if path.is_file():
        # Allocate functions to hold the loaded data
        phi = Function(V)
        c0 = Function(Q)
        c1 = Function(Q)

        # Read data from the ADIOS file
        adios4dolfinx.read_function(path, phi, time=0.0, name="phi")
        adios4dolfinx.read_function(path, c0, time=0.0, name="c0")
        adios4dolfinx.read_function(path, c1, time=0.0, name="c1")
        return (c0, c1, phi, tau)

    # --- 3. Generate Initial State (c0) and Velocity (phi) ---

    # Define initial condition c0 as a Gaussian-like bump
    x = ufl.SpatialCoordinate(mesh)
    mu = 0.5  # Center of the bump
    sigma2 = 0.1  # Width (variance) of the bump
    bump = 2 * ufl.exp(-((x[0] - mu) ** 2 + (x[1] - mu) ** 2) / sigma2)

    # Create and interpolate the function
    c0 = Function(Q)
    c0.interpolate(dolfinx.fem.Expression(bump, Q.element.interpolation_points))

    # Define the constant convection field phi = (g, g)
    phi = Function(V, name="phi")
    phi.interpolate(
        lambda x: np.array([g * np.ones_like(x[0]), g * np.ones_like(x[0])])
    )

    # --- 4. Define Variational Problem (Backward Euler time step) ---
    # We solve for c1 (as 'c') using c0 as the previous state.
    #
    # Equation: (c - c0)/tau + div(phi*c) - D*div(grad(c)) + r*c = 0
    #
    # Weak Form (multiply by d*tau and integrate):
    # a(c, d) = L(d)
    # a(c, d) = integral[ c*d + tau*div(phi*c)*d + tau*D*grad(c)*grad(d) + tau*r*c*d ] dx
    # L(d)     = integral[ c0*d ] dx
    # (Note: Integration by parts is used on the diffusion term)

    c = ufl.TrialFunction(Q)
    d = ufl.TestFunction(Q)

    # Left-hand side 'a':
    # Mass term (c*d) + Convection term (tau*div(phi*c)*d)
    a = (ufl.inner(c, d) + tau * ufl.inner(ufl.div(c * phi), d)) * ufl.dx()

    assert D is not None, "Diffusion coefficient D must be provided"
    # Add diffusion term
    a += tau * ufl.inner(D * ufl.grad(c), ufl.grad(d)) * ufl.dx()
    # Add reaction term
    if r is not None and abs(r) > 0.0:
        a += tau * r * c * d * ufl.dx()

    # Right-hand side 'L':
    L = ufl.inner(c0, d) * ufl.dx()

    # --- 5. Define Boundary Conditions ---
    # Apply Dirichlet BC: c1 = c0 on the entire exterior boundary.
    # This forces the new state to match the old state at the boundary,
    # effectively handling inflow/outflow.
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    # Locate DOFs on the boundary facets for the *scalar* space Q
    boundary_dofs = dolfinx.fem.locate_dofs_topological(Q, fdim, boundary_facets)

    # Apply the *initial state* c0 as the boundary value for c1
    bc = dolfinx.fem.dirichletbc(c0, boundary_dofs)

    # --- 6. Solve the Linear Problem ---
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},  # Use a direct solver
        petsc_options_prefix="Poisson",  # (Note: prefix is a misnomer, this is C-D-R)
    )

    # Solve for c1 (the state at t=tau)
    c1 = problem.solve()

    # --- 7. Save Data to Files ---

    # Save data for checkpointing/reloading using ADIOS
    # This is the file that the caching logic checks for
    adios4dolfinx.write_function_on_input_mesh(path, c0, time=0.0, name="c0")
    adios4dolfinx.write_function_on_input_mesh(path, c1, time=0.0, name="c1")
    adios4dolfinx.write_function_on_input_mesh(path, phi, time=0.0, name="phi")

    # Save data for visualization (e.g., ParaView) using VTX/BP5
    with dolfinx.io.VTXWriter(
        mesh.comm, output / "data_viz.bp", [c0, c1, phi], engine="BP5"
    ) as vtx:
        vtx.write(0.0)

    # Return the computed functions
    return (c0, c1, phi, tau)
