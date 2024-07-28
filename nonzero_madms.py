import matplotlib.pyplot as plt
from time import time
from functions import *
import itertools


"""
To do:
Gaussian to filter around specific energy
"""


def nonzero_madm(J_0, J, Jp, k, kp, K, S):
    #valid_Js = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]

    #if AMCOs(J, Jp, k, kp, M_0, M_0, K, 0, S) != 0 and a('X', J, J_0, M_0, k) != 0 and a('X', Jp, J_0, M_0, kp) != 0:
        #print(J_0, J, Jp, k, kp, K, S, "section 1")

    #if J not in valid_Js or Jp not in valid_Js:
        #return False

    if (J + Jp + K) % 2 != 0:
        return False

    if not abs(J - K) <= Jp <= J + K:
        return False

    if kp - k != S:
        return False

    #print(J_0, J, Jp, k, kp, K, S, " section 2")
    #print()
    return True


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
elec2 = "X"

madms = []

# Function to generate (J, k) permutations
def generate_permutations(J_0):
    J_values = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]
    return [(J, k) for J in J_values for k in range(-J, J + 1)]

# Generate all (J, k) pairs
permutations = generate_permutations(J_0)

# Generate all combinations of these pairs taken 2 at a time
combined_permutations = itertools.product(permutations, repeat=2)

# Filter based on nonzero_madm condition
permutations_of_perms = [
    [J, k, Jp, kp] for (J, k), (Jp, kp) in combined_permutations
    if nonzero_madm(J_0, J, Jp, k, kp, 2, 0)
]

def psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0):
    """
    return normalization constant for psi
    """

    # what J values are possible, given J_0
    J_possibilities = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]    # By wigner rules, J = +- J_0

    psi = []

    # find psi (without the e^it term) and each Born-Oppenheimer state's information
    for i, vector in enumerate(T2_eigenvectors_in_BO):
        psi_n = []

        for i, ground_state_component in enumerate(vector):
            for J in J_possibilities:
                for K in range(-J, J + 1):
                    psi_n = np.append(psi_n, a(states[i][3], J, J_0, M_0, K) * ground_state_component)

        psi.append(psi_n)

    psi_0 = np.sum(np.array(psi), axis=0)

    return np.sqrt(np.conj(psi_0).dot(psi_0))

# Get possible quantum states
states = get_possible_states(q)

# Calculate Hamiltonian matrix
hamiltonian = np.array([[matrix_elem(state1, state2).real for state2 in states] for state1 in states])

# Get eigenvectors and eigenvalues
T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, Kt, indices_of_good_eigenvalues)

psi_norm = psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0)

for nonzero_term in permutations_of_perms:
    J, k, Jp, kp = nonzero_term[0], nonzero_term[1], nonzero_term[2], nonzero_term[3]

    amco = AMCOs(J, Jp, k, kp, M_0, M_0, K, Q, S)

    exp_1 = np.array([np.exp(-1j * (energy + E_jkm(B, J))) for energy in T2_BO_eigenvalues])
    exp_2 = np.array([np.exp(-1j * (energy + E_jkm(B, Jp))) for energy in T2_BO_eigenvalues])

    # Create a dictionary with keys for the 'X', 'Y', and 'Z' electronic components
    elec_dict = {"X": 0,
                 "Y": 1,
                 "Z": 2}

    electronically_seperated = [np.array_split(np.array(v), 3) for v in T2_eigenvectors_in_BO / len(T2_eigenvectors_in_BO)]

    x = a(elec1, J, J_0, M_0, k) * np.array([vector[elec_dict[elec1]] for vector in electronically_seperated]) / psi_norm
    xp = a(elec2, Jp, J_0, M_0, kp) * np.array([vector[elec_dict[elec2]] for vector in electronically_seperated]) / psi_norm


    def madm(amco, x, xp, exp_1, exp_2, t):
        exponent_1 = np.power(exp_1, t)
        term1 = np.sum([vector * constant for vector, constant in zip(x, exponent_1)], axis=0)

        exponent_2 = np.power(exp_2, t)
        term2 = np.sum([vector * constant for vector, constant in zip(xp, exponent_2)], axis=0)

        return amco * np.outer(np.conj(term2), term1).flatten()

    results = np.array([madm(amco, x, xp, exp_1, exp_2, t) for t in t_values]).T

    for ans in results:
        madms.append(ans)



print(f"Time to execute: {round(time() - t0, 2)}s for {len(madms)} MADM elements")

real = np.real(madms)
imag = np.imag(madms)


def plot_fun_stuff(plotting):

    # Plot the traced result and the off-diagonal ones separately
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i, element in enumerate(plotting):
        condition = abs(np.min(element) - np.max(element)) > 1e-5
        if condition:
            ax1.plot(t_values * time_conversion, element, linewidth=0.5)


    ax1.set_title(f"{elec1}{elec2} Full MADM K=2 with time, quanta={q}")
    ax1.set_ylabel('Value')
    ax2.set_xlabel('Time (fs)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# plot_fun_stuff(real)
# plot_fun_stuff(imag)
