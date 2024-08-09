from time import time
from nonzero_madm_functions import *

# Time range and steps
start = 0 / time_conversion  # convert from fs to AU
end = 50 / time_conversion  # convert from fs to AU
num_steps = 50
t_values = np.linspace(start, end, num_steps)

# Constants
q = 30  # any less than 2 gets wonky
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

timing_data = {}
t0 = time()     # total timer

# Find nonzero angular momentum pairs
nonzero_pairs = find_nonzero_angular_momentum_MADM_pairs(J_0, K, S)

# Get possible quantum states
states = get_possible_states(q)

# Calculate Hamiltonian matrix
start_time = time()
hamiltonian = build_hamiltonian(states)
timing_data['Make Hamiltonian'] = time() - start_time

# Get eigenvectors and eigenvalues
start_time = time()
T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, Kt, indices_of_good_eigenvalues)
timing_data['Eigenvectors/values'] = time() - start_time

# Calculate normalization constant for the wavefunction (may already be normalized)
start_time = time()
psi_norm_constant = psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0)
print(f"normalization constant: {psi_norm_constant}")
timing_data['Normalization'] = time() - start_time

# Calculate MADM for each nonzero angular momentum pair
start_time = time()
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


timing_data[f'All {len(madms_coherences) + len(on_diagonal_madms)} MADMs over {num_steps} steps'] = time() - start_time

full_time = time() - t0

# Create pie chart with timing information
labels = [f"{key}: {value:.2f}s ({(value / full_time) * 100:.1f}%)" for key, value in timing_data.items()]
times = list(timing_data.values())
plt.figure(figsize=(10, 7))
plt.pie(times, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title(f'Time taken by different parts of the function q={q}, J_0={J_0}, ({round(full_time,1)}s)')
plt.show()
