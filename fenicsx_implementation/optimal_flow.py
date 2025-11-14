# Install dependencies withing an environment with FEniCSx installed.
# pip install -r fenicsx_implementation/requirements.txt
import json
import logging
from pathlib import Path
import numpy as np
from mpi4py import MPI
import ufl
import adios4dolfinx
import dolfinx
import dolfinx_adjoint  # Import the dolfinx_adjoint library
import utils
import pyadjoint  # Import the core pyadjoint library for optimization

# --- 1. Basic Setup ---

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
for lib in ["matplotlib"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# --- 2. Define Problem Parameters ---
n = 40  # Mesh resolution (n x n grid)
g = 2e-2  # Velocity field magnitude, phi = (g, g)
noise_variance = 0.01  # Variance (sigma^2) of the noise to add to data
comm = MPI.COMM_WORLD  # MPI communicator
mesh = dolfinx.mesh.create_unit_square(comm, n, n)  # The computational mesh
reg = "H1"  # Type of regularization to use ("H1" or other)
alpha = dolfinx.fem.Constant(mesh, 1e-3)  # Regularization parameter
D = dolfinx.fem.Constant(mesh, 1e-2)  # Diffusion coefficient
output = Path("results_velocity_dolfin_adjoint") / f"method_n_{n}_g_{g}"
output.mkdir(exist_ok=True, parents=True)

# Define the function space for the concentration 'c'
V_D = dolfinx.fem.functionspace(mesh, ("CG", 1))

# --- 3. Generate or Load Synthetic Data ---
# Use the utility function to create the "ground truth" data.
# This simulates the forward problem with a *known* velocity field 'phi'
# to get an initial state c0 and a final state c_T.
# We use dolfinx_adjoint.Function to ensure compatibility with adjoint operations.
(c0_data, c_T_data, phi, tau) = utils.generate_data_bump(
    mesh,
    g,
    D=D,
    output=output,
    Function=dolfinx_adjoint.Function,
)
N = c0_data.x.array.size
r0 = np.random.normal(0, noise_variance, N)
r1 = np.random.normal(0, noise_variance, N)

# --- 4. Create Noisy Observation Data ---
# We create two sets of data:
# 1. "_true" functions: Store the clean, noise-free ground truth for comparison.
# 2. "data" functions: The actual data used in the inversion (ground truth + noise).

# Store clean initial condition
c0_true = dolfinx.fem.Function(V_D, name="c0_true")
c0_true.x.array[:] = c0_data.x.array[:]
# Store clean final condition
c_T_true = dolfinx.fem.Function(V_D, name="c_T_true")
c_T_true.x.array[:] = c_T_data.x.array[:]

# Add noise to the data to create the synthetic observations
c0_data.x.array[:] += r0
c_T_data.x.array[:] += r1

# Get the state space from the data
V_state = c0_data.function_space

# NOTE: This block appears to be a duplicate and adds noise a second time.
# This is likely an error, but we document what the code does.
N = c0_data.x.array.size
r0 = np.random.normal(0, noise_variance, N)
r1 = np.random.normal(0, noise_variance, N)

c0_data_true = dolfinx.fem.Function(V_state, name="c0_data_true")
c0_data_true.x.array[:] = c0_data.x.array[
    :
]  # (This is a copy of the *already noisy* data)
c2_true = dolfinx.fem.Function(V_state, name="c2_true")
c2_true.x.array[:] = c_T_data.x.array[
    :
]  # (This overwrites c2_true with the *noisy* data)
c0_data.x.array[:] += r0  # Add noise *again*
c_T_data.x.array[:] += r1  # Add noise *again*
c0_data.name = "c0_data"
c_T_data.name = "c_T_data"


print("Computing OCD via reduced approach")

# --- 6. Set up the Inverse Problem ---

# Define mesh and function space for the (unknown) concentration 'c'
C = dolfinx.fem.functionspace(mesh, ("CG", 1))

# Space for the convective velocity field 'phi' (the control variable)
Q = dolfinx.fem.functionspace(mesh, ("CG", 1, (mesh.topology.dim,)))

# Define the control variable 'phi' as a dolfinx_adjoint.Function.
# This tells dolfinx_adjoint to track operations involving this function.
phi = dolfinx_adjoint.Function(Q, name="Control")
# We start with an initial guess of phi = 0 (the default).

# Define the previous solution 'c_' (initial condition)
c_ = dolfinx_adjoint.Function(C)
c_.x.array[:] = c0_data.x.array[:]  # Use the noisy initial data

# Define the target observation 'c2'
c2 = dolfinx_adjoint.Function(C)
c2.x.array[:] = c_T_data.x.array[:]  # Use the noisy final data
dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": 4})

# --- 7. Define the Forward PDE (State Equation) ---
# This is the weak form of the convection-diffusion-reaction equation,
# solved with Backward Euler:
# (c - c_)/tau + div(c*phi) - div(D*grad(c)) = 0
c = ufl.TrialFunction(C)
d = ufl.TestFunction(C)
F = (
    1.0 / tau * (c - c_) * d  # Time derivative: (c - c_)/tau * d
    + ufl.div(c * phi) * d  # Convection: div(c*phi) * d
    + ufl.inner(D * ufl.grad(c), ufl.grad(d))  # Diffusion
) * dx
a, L = ufl.lhs(F), ufl.rhs(F)  # Split into bilinear (a) and linear (L) forms


# Define the regularization functional R(phi)
def R(phi, alpha):
    """Defines the regularization cost functional."""
    if reg == "H1":
        # H1 regularization: 0.5 * alpha * (||phi||^2 + ||grad(phi)||^2)
        form = (
            0.5
            * alpha
            * (ufl.inner(phi, phi) + ufl.inner(ufl.grad(phi), ufl.grad(phi)))
            * dx
        )
    else:
        # H(div) regularization: 0.5 * alpha * (||phi||^2 + ||div(phi)||^2)
        form = (
            0.5
            * alpha
            * (ufl.inner(phi, phi) + ufl.inner(ufl.div(phi), ufl.div(phi)))
            * dx
        )
    return form


# --- 8. Set up Boundary Conditions for the Forward Problem ---
tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V_state, fdim, boundary_facets)

# Set the BC for the forward solve to match the *target data* on the boundary.
# This strongly enforces that the final state matches the observation data
# at the boundary.
bc = dolfinx.fem.dirichletbc(c2, boundary_dofs)

# --- 9. Define the Adjoint-Tracked Forward Solver ---
# 'c' is the state variable. dolfinx_adjoint will compute its derivative w.r.t. 'phi'.
c = dolfinx_adjoint.Function(C, name="State")
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",  # Use a direct solver (LU)
    "pc_factor_mat_solver_type": "mumps",  # Use MUMPS for factorization
}

# Create the LinearProblem. dolfinx_adjoint overloads this class
# to automatically record the solve on its "tape".
problem = dolfinx_adjoint.LinearProblem(
    a,
    L,
    u=c,
    bcs=[bc],
    petsc_options=petsc_options,
    adjoint_petsc_options=petsc_options,  # Options for the adjoint solve
)

# Solve the forward problem once with the initial guess (phi=0)
problem.solve()

# Output max values for comparison
print("max c_1 = %f" % c2.x.array.max())  # Target data
print("max c = %f" % c.x.array.max())  # State from initial guess

# --- 10. Define the Objective Functional ---
# The goal is to minimize J = J_misfit + J_regularization
j = 0.5 * (c - c2) ** 2 * dx + R(phi, alpha)
J = dolfinx_adjoint.assemble_scalar(j)
print("J (initial) = %f" % J)

# --- 11. Set up the Optimization ---
# Define the control variable that pyadjoint will optimize
m = pyadjoint.Control(phi)


class CallBack:
    """A callback class to monitor the optimization progress."""

    def __init__(self, output_dir, comm, print_freq=30):
        self.counter = 0
        self.output_dir = output_dir
        self.mesh = mesh
        self.resfile = output_dir / "opts.json"  # File for logging JSON results
        self.fname = output_dir / "D_opt.bp"  # File for saving intermediate phi
        self.results = []
        self.comm = comm
        self.print_freq = print_freq

    def __call__(self, j, phij):
        """This method is called by the optimizer at each iteration."""
        phi_max = mesh.comm.allreduce(phij.x.array.max(), op=MPI.MAX)

        if self.counter % self.print_freq == 0:
            # Print progress
            print(rf"j = {j}, max phi = {phi_max} (mm/h)")
            self.results.append({"count": self.counter, "j": j, "phi_max": phi_max})
            # Save JSON log
            self.resfile.write_text(json.dumps(self.results))

            # Save the current control variable 'phi' to an ADIOS file
            adios4dolfinx.write_function_on_input_mesh(
                self.fname, phij, time=self.counter, name="phi"
            )

        self.counter += 1


# Create the ReducedFunctional. This is the core of dolfinx_adjoint.
# It wraps the objective functional 'J' and the control 'm'.
# When the optimizer (e.g., L-BFGS) asks Jhat for its value, pyadjoint
# solves the forward PDE and computes J.
# When the optimizer asks for the gradient, pyadjoint solves the
# adjoint PDE and computes the gradient dJ/dphi.
Jhat = pyadjoint.ReducedFunctional(J, m, eval_cb_post=CallBack(output, mesh.comm))

# --- 12. Run the Minimization ---
# This calls the L-BFGS-B minimizer from scipy.optimize
tol = 1.0e-12
phi_opt = pyadjoint.minimize(
    Jhat, tol=tol, options={"gtol": tol, "maxiter": 500, "disp": True}
)

# --- 13. Post-Processing: Get Final Results ---
# The optimization is done. Update the 'phi' function with the optimal values.
phi.x.array[:] = phi_opt.x.array[:]

# Solve the forward problem one last time with the optimal 'phi'
# to get the final state 'c' that corresponds to it.
problem.solve()

# Assemble the final cost components
J = dolfinx_adjoint.assemble_scalar(j)
j0 = 0.5 * (c - c2) ** 2 * dx  # Final misfit
jr = R(phi, alpha)  # Final regularization
# We document the code as written.
J0 = dolfinx_adjoint.assemble_scalar(j0)
Jr = dolfinx_adjoint.assemble_scalar(jr)
print("J  = %f" % J)
print("J0 = %f" % J0)
print("Jr = %f" % Jr)


# Save final state and parameter to ADIOS file for checkpointing
adios4dolfinx.write_function_on_input_mesh(output / "results.bp", c, time=0.0, name="c")
adios4dolfinx.write_function_on_input_mesh(
    output / "results.bp", phi, time=0.0, name="phi"
)

# Save final state and parameter to VTX file for visualization
with dolfinx.io.VTXWriter(
    mesh.comm,
    output / "results_viz.bp",
    [c, phi, c0_data, c0_true, c_T_data, c_T_true],
    engine="BP5",
) as vtx:
    vtx.write(0.0)
