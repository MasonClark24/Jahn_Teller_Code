from time import time
from nonzero_madm_functions import *


def plot_coherences_grid(t_values, coherences_grid, K, Q, S, q):
    """
    Plots the coherences in a 3x3 grid for different electronic state combinations.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    fig.suptitle(f"Rovibronic Coherences K,Q,S={K},{Q},{S} over time, quanta={q}, J_0={J_0}", fontsize=16)

    # Iterate through the grid and plot
    for i, (ax, (elec1, elec2)) in enumerate(zip(axes.flat, coherences_grid.keys())):
        coherences = coherences_grid[(elec1, elec2)]

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

        ax.set_title(f"{elec1}{elec2} Coherences")
        ax.set_ylabel('Value')
        ax.set_xlabel('fs')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.show()


# Time range and steps
start = 0 / time_conversion  # convert from fs to AU
end = 150*2 / time_conversion  # convert from fs to AU
num_steps = 50
t_values = np.linspace(start, end, num_steps)

# Constants
q = 5   # any less than 2 gets wonky
print(f"-------- q = {q} --------")

K = 2
Q = 0
S = 0
J_0 = 0

# Electronic state combinations
electronic_combinations = [('X', 'X'), ('X', 'Y'), ('X', 'Z'), ('Y', 'X'),
                           ('Y', 'Y'), ('Y', 'Z'), ('Z', 'X'), ('Z', 'Y'),
                           ('Z', 'Z')]

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

# Initialize dictionary to hold coherence results for each combination
coherences_grid = {}

# Iterate over each electronic state combination
for elec1, elec2 in electronic_combinations:
    final_madm = np.zeros((len(states) * len(states) // 9, num_steps), dtype=np.complex128)

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

    # Store coherences in the grid
    coherences_grid[(elec1, elec2)] = np.real(coherences)

# Plot the results in a 3x3 grid
plot_coherences_grid(t_values, coherences_grid, K, Q, S, q)

# Print execution time
print(f"Time to execute: {round(time() - t0, 4)}s")
