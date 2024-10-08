from time import time

import numpy as np

from nonzero_madm_functions import *


# Time range and steps
start = 0 / time_conversion  # convert from fs to AU
end = 50 / time_conversion  # convert from fs to AU
num_steps = 50
t_values = np.linspace(start, end, num_steps)

# Constants
q = 10  # any less than 2 gets wonky
print(f"-------- q = {q} --------")

K = 2
Q = 0
S = 0
J_0 = 0

# Electronic states
elec1 = "X"
elec2 = "Y"

# Initialize lists to hold MADM elements
madms_coherences = []
on_diagonal_madms = []

# Find nonzero angular momentum pairs
nonzero_pairs = find_nonzero_angular_momentum_MADM_pairs(J_0, K, S)

# Get possible quantum states
states = get_possible_states(q)

# Calculate Hamiltonian matrix
hamiltonian = build_hamiltonian(states)

# Get eigenvectors and eigenvalues
T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, Kt, indices_of_good_eigenvalues)

# Calculate normalization constant for the wavefunction (may already be normalized)
psi_norm_constant = psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0)

# Start timer
t0 = time()

# Calculate MADM for each nonzero angular momentum pair
for nonzero_term in nonzero_pairs:
    J, k, Jp, kp = nonzero_term

    # Get MADM elements for the specified angular momenta
    madms_nonzero = MADM_for_angular_pairs(J, k, Jp, kp, T2_eigenvectors_in_BO, T2_BO_eigenvalues, elec1, elec2,
                                           M_0, J_0, K, Q, S, t_values, psi_norm_constant)

    # If on the diagonal, separate the diagonal terms
    if J == Jp and k == kp:
        diagonal_indices = find_diagonal_indices(int(np.sqrt(len(madms_nonzero))))

        # Create a boolean mask to exclude diagonal indices
        exclude_mask = np.ones(len(madms_nonzero), dtype=bool)
        exclude_mask[diagonal_indices] = False

        # Extract elements that are not diagonal
        coherences = madms_nonzero[exclude_mask]

        # Extract elements that are diagonal
        on_diagonal = madms_nonzero[diagonal_indices]

        madms_coherences.extend(coherences)
        on_diagonal_madms.extend(on_diagonal)
    else:
        madms_coherences.extend(madms_nonzero)

# Print execution time and number of MADM elements
print(f"Time to execute: {round(time() - t0, 4)}s for {len(madms_coherences) + len(on_diagonal_madms)} MADM elements")
print("Coherences:", len(madms_coherences))
print("On-Diagonal:", len(on_diagonal_madms))

# Convert MADM elements to real parts
real = np.real(madms_coherences)
real_diag = np.real(on_diagonal_madms)
trace = np.sum(real_diag, axis=0)

# Determine tolerance for plotting
tol = 0
for r in real:
    diff = abs(np.max(r) - np.min(r))
    if diff > tol:
        tol = diff

tol /= 2
print(f"Tolerance = {tol}")

# Plot results
plot_fun_stuff(t_values, real, trace, tol, elec1, elec2, K, Q, S, q, trace=True, do_tolerance=True)
