import matplotlib.pyplot as plt
from time import time
from nonzero_madm_functions import *
import itertools

"""
Massive improvement by considering what wigner 3j symbols are zero for J, k, J', k'
"""

start = 0 / time_conversion  # convert from fs to AU
end = 50 / time_conversion  # convert from fs to AU
num_steps = 50
t0 = time()
t_values = np.linspace(start, end, num_steps)

K = 2
Q = 0
S = 0
J_0 = 0

elec1 = "X"
elec2 = "Y"

madms_coherences = []
on_diagonal_madms = []

# find the angular momentum pairs that are nonzero (ie. worth calculating)
nonzero_pairs = find_nonzero_angular_momentum_MADM_pairs(J_0, K, S)

# Get possible quantum states
states = get_possible_states(q)

# Calculate Hamiltonian matrix
hamiltonian = np.array([[matrix_elem(state1, state2).real for state2 in states] for state1 in states])

# Get eigenvectors and eigenvalues
T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, Kt, indices_of_good_eigenvalues)

# full vector includes the excitation amplitude into J and k, and this affects the normalization constant
# however, the full PSI_0 isn't needed, as we only need to explore specific J, k, J', k' pairs, so just get the constant
# note: psi may already be normalized with the math of the excitation constants
psi_norm = psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0)


# for every nonzero pair of states, calculate that MADM over time
for nonzero_term in nonzero_pairs:
    J, k, Jp, kp = nonzero_term[0], nonzero_term[1], nonzero_term[2], nonzero_term[3]

    madms_nonzero = MADM_for_angular_pairs(J, k, Jp, kp, T2_eigenvectors_in_BO, T2_BO_eigenvalues, elec1, elec2, M_0, J_0, K, Q, S)

    if J == Jp and k == kp:
        diagonal_indices = find_diagonal_indices(int(np.sqrt(len(madms_nonzero))))

        # Create a boolean mask to exclude diagonal indices
        exclude_mask = np.ones(len(madms_nonzero), dtype=bool)
        exclude_mask[diagonal_indices] = False

        # Extract elements that are not diagonal
        coherences = madms_nonzero[exclude_mask]

        # extract elements that are diagonal
        on_diagonal = madms_nonzero[diagonal_indices]

        for elem in coherences:
            madms_coherences.append(elem)

        for elem in on_diagonal:
            on_diagonal_madms.append(elem)
    else:
        for elem in madms_nonzero:
            madms_coherences.append(elem)


print(f"Time to execute: {round(time() - t0, 2)}s for {len(madms_coherences)+len(on_diagonal_madms)} MADM elements")
print("Coherences:", len(madms_coherences))
print("On-Diagonal:", len(on_diagonal_madms))

real = np.real(madms_coherences)
real_diag = np.real(on_diagonal_madms)
trace = np.sum(real_diag, axis=0)

# if something doesn't change at 1/Nth of the highest changing one, don't plot it b/c it hardly changes with time
tol = 0
for r in real:
    diff = abs(np.max(r) - np.min(r))
    if diff > tol:
        tol = diff

tol /= 2

print(f"tolerance = {tol}")
plot_fun_stuff(t_values, real, trace, tol, elec1, elec2, K, Q, S, trace=True, do_tolerance=True)
# plot_fun_stuff(t_values, plotting, on_diagonal_madms, tol, elec1, elec2, K, Q, S)
