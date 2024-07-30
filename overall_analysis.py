import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nonzero_madm_functions import *


def full_function(q, J_0):
    # Time range and steps
    start = 0 / time_conversion  # convert from fs to AU
    end = 50 / time_conversion  # convert from fs to AU
    num_steps = 50
    t_values = np.linspace(start, end, num_steps)

    K = 2
    Q = 0
    S = 0

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


# Range of q and J_0 values
q_values = range(3, 13)
J_0_values = range(0, 4)

# Initialize lists to store results
time_taken = {J_0: [] for J_0 in J_0_values}

# Measure time taken for each combination of q and J_0
for J_0 in J_0_values:
    print(J_0, J_0_values)
    for q in q_values:
        start_time = time.time()
        full_function(q, J_0)
        end_time = time.time()
        time_taken[J_0].append(end_time - start_time)

print()
# Print the data in Desmos-friendly format
for J_0 in J_0_values:
    print(f'J_0 = {J_0}')
    for q, time in zip(q_values, time_taken[J_0]):
        print(f'({q}, {time})')
    print()

# Curve fitting
def fit_function(q, a, b):
    return a * q ** b


# Fit and plot with formulas
plt.figure(figsize=(10, 6))
plt.grid(True)

colors = plt.cm.viridis(np.linspace(0, 1, len(J_0_values)))

for J_0 , color in zip(J_0_values, colors):
    popt, pcov = curve_fit(fit_function, q_values, time_taken[J_0])
    fitted_times = fit_function(q_values, *popt)

    residuals = np.array(time_taken[J_0]) - fitted_times
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.array(time_taken[J_0]) - np.mean(time_taken[J_0])) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    plt.plot(q_values, time_taken[J_0], 'o', color=color,
             label=f'J_0 = {J_0}      R² = {r_squared:.3f}      {popt[0]:.2e} * q^({popt[1]:.2e})')
    plt.plot(q_values, fitted_times, '-', color=color)

plt.xlabel('q')
plt.ylabel('Time Taken (s)')
plt.legend()
plt.title('Time Taken vs. q with Fitted Curves')
plt.show()


