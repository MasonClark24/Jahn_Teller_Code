from time import time

import numpy as np

from nonzero_madm_functions import *


def plot_selected_coherences_and_trace(t_values, coherences_grid, trace_grid, K, Q, S, q):
    """
    Plots the selected coherences (XX, YY, ZZ) in a 2x2 grid and the trace of diagonals below.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(f"Rovibronic Coherences and Trace K,Q,S={K},{Q},{S} over time, quanta={q}", fontsize=16)

    # Iterate through the selected coherences and plot
    selected_keys = ['XX', 'YY', 'ZZ']
    for i, (ax, key) in enumerate(zip(axes.flat, selected_keys)):

        coherences = coherences_grid[key]

        # Determine tolerance for plotting
        tol = 0
        for r in coherences:
            diff = abs(np.max(r) - np.min(r))
            if diff > tol:
                tol = diff
        tol /= 2
        if tol == 0:
            tol = 1

        for element in coherences:
            if abs(np.max(element) - np.min(element)) >= tol:
                ax.plot(t_values * time_conversion, element, linewidth=0.5)

        ax.set_title(f"{key} Coherences")
        ax.set_ylabel('Value')
        ax.set_xlabel('fs')

    # Plot the trace of diagonal elements
    trace_ax = axes[1, 1]
    for trace, key in zip(trace_grid, selected_keys):
        trace_ax.plot(t_values * time_conversion, trace, linewidth=0.5, label=key)

    trace = np.sum(trace_grid, axis=0)
    trace_ax.plot(t_values * time_conversion, trace, linewidth=0.5, label=f"Trace={np.average(trace)}", color='black')

    trace_ax.set_title("Trace of XX, YY, ZZ Diagonals")
    trace_ax.set_ylabel('Value')
    trace_ax.set_xlabel('fs')
    trace_ax.legend()


    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.show()


# Time range and steps
start = 0 / time_conversion  # convert from fs to AU
end = 150 / time_conversion  # convert from fs to AU
num_steps = 50
t_values = np.linspace(start, end, num_steps)

# Constants
q = 10  # any less than 2 gets wonky
print(f"-------- q = {q} --------")

K = 2
Q = 0
S = 0
J_0 = 0

# Focused electronic state combinations
selected_combinations = [('X', 'X'), ('Y', 'Y'), ('Z', 'Z')]

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

print(f"Norm = {psi_norm_constant}")

# Start timer
t0 = time()

# Initialize dictionaries to hold coherence and trace results
coherences_grid = {}
trace_grid = []

# Iterate over each selected electronic state combination
for elec1, elec2 in selected_combinations:
    final_madm = np.zeros((len(states) * len(states) // 9, num_steps), dtype=np.complex128)
    madms_coherences = []

    # Calculate MADM for each nonzero angular momentum pair
    for nonzero_term in nonzero_pairs:
        J, k, Jp, kp = nonzero_term

        # Get MADM elements for the specified angular momenta
        ans = MADM_for_angular_pairs(J, k, Jp, kp, T2_eigenvectors_in_BO, T2_BO_eigenvalues, elec1, elec2,
                                     M_0, J_0, K, Q, S, t_values, psi_norm_constant)

        final_madm += ans

    # Create a boolean mask to exclude diagonal indices
    diagonal_indices = find_diagonal_indices(int(np.sqrt(len(final_madm))))
    exclude_mask = np.ones(len(final_madm), dtype=bool)
    exclude_mask[diagonal_indices] = False

    # Extract elements that are not diagonal
    coherences = final_madm[exclude_mask]
    madms_coherences.extend(coherences)

    # Store coherences in the grid
    coherences_grid[f"{elec1}{elec2}"] = np.real(madms_coherences)

    # Calculate and store trace of diagonal elements
    on_diagonal = final_madm[diagonal_indices]
    real_diag = np.real(on_diagonal)
    trace = np.sum(real_diag, axis=0)
    trace_grid.append(trace)


# Plot the results in a 2x2 grid with the trace below
plot_selected_coherences_and_trace(t_values, coherences_grid, trace_grid, K, Q, S, q)

# Print execution time
print(f"Time to execute: {round(time() - t0, 4)}s")
