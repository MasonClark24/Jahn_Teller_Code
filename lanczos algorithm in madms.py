from time import time
import numpy as np
from nonzero_madm_functions import *

def Lanczos(H, eigenvectors_I_want):
    """
    Perform the Lanczos algorithm to find eigenvalues and eigenvectors of the Hamiltonian matrix H.

    :param H: Original Hamiltonian matrix.
    :param eigenvectors_I_want: List of indices of desired eigenvectors.
    :return: Tuple (T2_eigenvectors_in_BO, T2_BO_eigenvalues)
             where T2_eigenvectors_in_BO are the selected eigenvectors in the original basis,
             and T2_BO_eigenvalues are the corresponding eigenvalues.
    """
    vg = np.zeros(H.shape[0])
    vg[1] = -1 / np.sqrt(2)
    vg[2] = 1 / np.sqrt(2)

    on_diag = np.zeros(H.shape[0], dtype=complex)
    sub_diag = np.zeros(H.shape[0] - 1, dtype=complex)

    P = np.zeros((H.shape[0], H.shape[0]), dtype=complex)
    P[:, 0] = vg / la.norm(vg)

    w = np.dot(H, P[:, 0])
    a = np.dot(np.conj(w), P[:, 0])
    w = w - a * P[:, 0]
    on_diag[0] = a

    for j in range(1, H.shape[0]):
        b = la.norm(w)
        if b < 1e-4:
            #print(f"Break Lanczos algorithm due to small b: {b:.2e} at iteration {j} of {H.shape[0]}")
            on_diag = on_diag[:j]
            sub_diag = sub_diag[:j-1]
            P = P[:, :j]

            break

        P[:, j] = w / b
        for k in range(j):
            P[:, j] -= np.dot(np.conj(P[:, j]), P[:, k]) * P[:, k]
        P[:, j] /= la.norm(P[:, j])

        w = np.dot(H, P[:, j])
        a = np.dot(np.conj(w), P[:, j])
        w = w - a * P[:, j] - b * P[:, j - 1]

        on_diag[j] = a
        sub_diag[j - 1] = b

    eigenvalues, eigenvectors = eigh_tridiagonal(np.real(on_diag), np.real(sub_diag))

    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    eigenvalues += (2 * Kt ** 2 / 3)
    eigenvalues *= 0.03967 / 27.211396

    T2_eigenvectors_in_BO = []
    T2_BO_eigenvalues = []

    for i in eigenvectors_I_want:
        T2_eigenvectors_in_BO.append(np.dot(np.real(P), eigenvectors[:, i]))
        T2_BO_eigenvalues.append(eigenvalues[i])

    return np.array(T2_eigenvectors_in_BO), np.array(T2_BO_eigenvalues)

def build_hamiltonian(states, Kt):
    hamil = np.zeros((len(states), len(states)))
    for i, s in enumerate(states):
        hamil[i][i] = int(s[0]) + int(s[1]) + int(s[2]) + 1.5

    for i in range(3):

        for index, state in enumerate(states):
            electronic1 = state[3]
            electronic2 = None

            if electronic1 == 'X' and i == 0:
                continue
            elif electronic1 == 'Y' and i == 1:
                continue
            elif electronic1 == 'Z' and i == 2:
                continue
            elif electronic1 == 'X' and i == 2:
                electronic2 = 'Y'
            elif electronic1 == 'Y' and i == 0:
                electronic2 = 'Z'
            elif electronic1 == 'Z' and i == 1:
                electronic2 = 'X'
            else:
                continue

            high_state = state.copy()
            high_state[i] += 1
            high_state[3] = electronic2


            try:
                other_state_index = states.index(high_state)
                hamil[index][other_state_index] = np.real(sqrt(high_state[i]) / sqrt(2)) * Kt
                hamil[other_state_index][index] = np.real(sqrt(high_state[i]) / sqrt(2)) * Kt

            except ValueError:
                pass

            low_state = state.copy()
            low_state[i] -= 1
            low_state[3] = electronic2

            try:
                other_state_index = states.index(low_state)
                hamil[index][other_state_index] = np.real(sqrt(state[i]) / sqrt(2)) * Kt
                hamil[other_state_index][index] = np.real(sqrt(state[i]) / sqrt(2)) * Kt
            except ValueError:
                pass

    return hamil


# Time range and steps
start = 0 / time_conversion  # convert from fs to AU
end = 50 / time_conversion  # convert from fs to AU
num_steps = 50
t_values = np.linspace(start, end, num_steps)

# Constants
q = 15  # any less than 2 gets wonky
print(f"-------- q = {q} --------")

K = 2
Q = 0
S = 0
J_0 = 0

timing_data = {}
t0 = time()

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
hamiltonian = build_hamiltonian(states, Kt)

# Get eigenvectors and eigenvalues
start_time = time()
T2_eigenvectors_in_BO, T2_BO_eigenvalues = Lanczos(hamiltonian, indices_of_good_eigenvalues)
# T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, 1, indices_of_good_eigenvalues)
timing_data['Lanczos Eigenvectors/values'] = time() - start_time


# Calculate normalization constant for the wavefunction (may already be normalized)
start_time = time()
psi_norm_constant = psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0)
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





