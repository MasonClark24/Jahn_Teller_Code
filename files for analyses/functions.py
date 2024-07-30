from cmath import sqrt
from itertools import product
import numpy as np
from scipy.linalg import eigh
import math
import numpy.linalg as la
from scipy.linalg import eigh_tridiagonal, inv
from wigners import wigner_3j
import itertools
import matplotlib.pyplot as plt




class Vector_Part:
    """
    Defining a state | psi >
    """

    def __init__(self, J, K, vib, elec, E):
        """
        Saves relevant information.

        J, K are rotational, vib is the vibrational, elec is the electronic
        E is the energy of its eigenstate with the energy from rotation, too
        """
        self.J = J
        self.K = K
        self.vib = vib
        self.elec = elec
        self.E = E

    def __str__(self):
        """
        Print all information
        """
        return f"|{str(self.vib):6} {self.elec}   J: {self.J}, K: {self.K}>"


def get_possible_states(quanta):
    """
    Return the list of states, given that you have a certain number of quanta to go around
    for vibration and three possible electronic states

    :param quanta: How many quanta can go into vibration?
    :return: List of possible states. Ex. [ 000, X ] is zero quanta in vibrational and 'X' for electronic
    """

    def distribute_balls(q):
        """
        :param q: Distribute the quanta into three buckets
        :return: combinations of these
        """

        combinations = []
        for distribution in product(range(q + 1), repeat=3):
            if sum(distribution) == q:
                combinations.append(list(distribution))
        return combinations

    quanta_permutations = []

    # include permutations for all quanta ranging from zero to the max amount
    for i in range(0, int(quanta + 1)):
        quanta_permutations += distribute_balls(i)

    electronic = ["X", "Y", "Z"]
    states = []

    # provide all states
    for vib in electronic:
        for item in quanta_permutations:
            state = item + [vib]
            states.append(state)

    return states


def matrix_elem(state1, state2):
    """
    Calculate hamiltonian matrix elements for given quantum states (indices)

    || doesn't include the Kt part, which is added later for easier manipulation ||

    Parameters:
    - state1: Quantum state 1
    - state2: Quantum state 2

    Returns:
    - Matrix element
    """

    # If the states are the same, its the simple harmonic oscillator solution
    if state1 == state2:
        return int(state1[0]) + int(state1[1]) + int(state1[2]) + 1.5
    elif state1[3] == state2[3]:
        return 0

    # which N does the matrix allow coupling for
    electronic_jahn_teller = {
        "XY": 2,
        "YX": 2,
        "XZ": 1,
        "ZX": 1,
        "YZ": 0,
        "ZY": 0,
    }

    index_of_coupling = electronic_jahn_teller[state1[3] + state2[3]]

    raised_state1 = state1.copy()
    raised_state1[index_of_coupling] += 1

    lowered_state1 = state1.copy()
    lowered_state1[index_of_coupling] -= 1

    # if they couple, give them a value.

    if raised_state1[:3] == state2[:3] or lowered_state1[:3] == state2[:3]:

        # Raising/lowering constant
        biggest_value = max(state1[index_of_coupling], state2[index_of_coupling])

        return sqrt(biggest_value) / sqrt(2)
    else:
        return 0


def compute_eigenvalues_and_vectors_eigh(Matrix, Kt):
    """
    Compute eigenvalues and eigenvectors for a given Kt.

    Parameters:
    - Matrix: Matrix of matrix elements
    - Kt: Parameter

    Returns:
    - Eigenvalues and eigenvectors
    """

    # multiply all non-diagonal elements by Kt
    Matrix2 = Matrix.copy()
    mask = ~np.eye(Matrix2.shape[0], dtype=bool)
    Matrix2[mask] *= Kt

    # find eigenvalues/vectors
    eigenvalues, eigenvectors = eigh(Matrix2)

    # Sort eigenvalues and eigenvectors in ascending order
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors


def eigenvector_and_eigenvalues(hamiltonian, Kt, eigenvectors_I_want):
    """
    Return the eigenvalues / vectors for the eigenvectors_I_want, given the hamiltonian and Kt
    """

    eigenvalues, eigenvectors = compute_eigenvalues_and_vectors_eigh(hamiltonian, Kt)
    eigenvalues += (2 * Kt ** 2 / 3)  # correction factor. See sakamoto

    eigenvalues *= 0.03967 / 27.211396  # in units of Hbar OmegaT

    # eigenvalues += 9/27.211396

    # OmegaT here is T2 Stretching mode. So this converts the values to AU, not hbarW

    T2_eigenvectors_in_BO = []
    T2_BO_eigenvalues = []

    for i in eigenvectors_I_want:
        T2_eigenvectors_in_BO.append(eigenvectors[i])
        T2_BO_eigenvalues.append(eigenvalues[i])

    return np.array(T2_eigenvectors_in_BO), np.array(T2_BO_eigenvalues)


def AMCOs(Jn, Jpnp, Kn, Kpnp, Mn, Mpnp, K, Q, S):
    """
    Given the J, K, M, for the row/col and KQS, returns the amco value

    """

    line0 = ((2 * K + 1) / (8 * math.pi ** 2)) * math.sqrt((2 * Jn + 1) * (2 * Jpnp + 1))
    line0 *= pow(-1, abs(Mn - Kn))  # -1^-1 is the same as -1^1
    line1 = wigner_3j(Jn, Jpnp, K, -Mn, Mpnp, Q) * wigner_3j(Jn, Jpnp, K, -Kn, Kpnp, S)
    A = line0 * line1


    return A


def full_amco(J_0, K, Q, S, vector_parts):
    """
    The amco is largely a repeated matrix. For different J, K in the rows/columns its unique, but this is
    repeated many times for every psi element. Thus, given J_0, we know how many unique J,K pairs there are.
    This is how many times the matrix is repeated

    Ex. A chess board is a repeated matrix. We know it has a 2x2 of black-white-black-white.
    So just copy that many times over :)


    :param vector_parts: contains information on the orders of J, K for the rows/cols
    :return: full amco
    """
    # Gets the time-independent AMCOs
    amco_list = []

    # find how many unique indices there are (J, K)
    num_unique_indices = 0

    # for every J_possibility, there are 2j+1 Ks that go with it.
    J_possibilities = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]     # By wigner rules, J = +- J_0
    for j in J_possibilities:
        num_unique_indices += 2 * j + 1

    # calculate for the portion of unique values of J,K
    # assumes they are in (abc, abc, abc) order as opposed to (aaa, bbb, ccc)
    for row in vector_parts[:num_unique_indices]:
        amcos = []

        for col in vector_parts[:num_unique_indices]:
            # Extract values from row and col
            J, Jp, k, kp = row.J, col.J, row.K, col.K

            # Append AMCO to the amco_list
            amcos.append(AMCOs(J, Jp, k, kp, M_0, M_0, K, Q, S))


        amco_list.append(amcos)

    # the full amco is something like this for every psi element (as they each have the full J,K options added to them)
    size_of_electronic_section = len(vector_parts) // 3  # there is X, Y, Z
    how_many_amco_copies = size_of_electronic_section // num_unique_indices
    amco_list = np.tile(amco_list, (how_many_amco_copies, how_many_amco_copies))

    return amco_list  # makes an amco matrix the size of the density matrix, but only had to calculate a portion of it


def psi_t_function(psi_0, constant, t, num_psis):
    """
    Given a base psi(0), turn it into psi(t)
    """

    PSI_T = psi_0.copy() * np.power(constant, t)

    # for each eigenvector, there is a full psi (since each has a unique eigenvalue)
    # Ex. 9 eigenvectors chosen means that there are 9 psis. This makes calculations for psi(t) easier
    # Add each of these psis to achieve the total psi(t)
    separate_psis = np.array_split(PSI_T, num_psis)

    # add them together
    PSI_T = np.sum(separate_psis, axis=0)

    # interestingly, adding these sub-vectors together keeps the normalization of psi_0
    # But it's good to do it, still
    PSI_T /= np.sqrt(np.conj(PSI_T).dot(PSI_T))

    return PSI_T


def psi(T2_eigenvectors_in_BO, T2_BO_eigenvalues, states, B, J_0, M_0):
    """
    Given the hamiltonian, Kt, the indices of the eigenvectors_I_want,
    and other relevant constants like {B, J_0, M_0},

    - return psi_0 (ie. no e^it) and each Born-Oppenheimer state info
    """

    # what J values are possible, given J_0
    J_possibilities = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]    # By wigner rules, J = +- J_0

    psi = np.array([])
    psi_t_vector_parts = []

    # find psi (without the e^it term) and each Born-Oppenheimer state's information
    for i, vector in enumerate(T2_eigenvectors_in_BO):
        E_vibration = T2_BO_eigenvalues[i]

        for i, ground_state_component in enumerate(vector):
            for J in J_possibilities:

                En = E_vibration + E_jkm(B, J)  # units?

                # print(E_vibration, E_jkm(B, J))

                for K in range(-J, J + 1):
                    # elec state is states[i][3]

                    psi = np.append(psi, a(states[i][3], J, J_0, M_0, K) * ground_state_component)

                    # print(J, K)

                    info = Vector_Part(J, K, states[i][:3], states[i][3], En)
                    # print(info)
                    psi_t_vector_parts.append(info)

    psi /= np.sqrt(np.conj(psi).dot(psi))  # normalize


    return psi, np.array(psi_t_vector_parts)


def a(elec_state, J, Jin, M, k):

    mux, muy, muz = 0, 0, 0

    if elec_state == "X":
        mux = 1
    elif elec_state == "Y":
        muy = 1
    elif elec_state == "Z":
        muz = 1

    # mux, muy, muz = 1, 1, 1

    mu = [0, 0, 0]
    mu[0] = (1 / np.sqrt(2)) * (mux - 1j * muy)
    mu[1] = muz
    mu[2] = (-1 / np.sqrt(2)) * (mux + 1j * muy)

    A = 0

    for MU, q in zip(mu, [-1, 0, 1]):
        for Kin in range(-Jin, Jin + 1):

            # wigner 3j condition
            if -Kin + -q + k != 0:
                continue

            term = ((-1) ** (Kin - M + q) * np.sqrt((2 * J + 1) * (2 * Jin + 1)) *
                  wigner_3j(Jin, 1, J, -Kin, -q, k) * wigner_3j(Jin, 1, J, -M, 0, M) * MU)

            A += term

    return A


def nonzero_madm(J_0, J, Jp, k, kp, K, S):
    """
    Given J_0, J, Jp, k, kp, K, S:
    - excitation amplitude from J_0 into J, k is zero if J isn't J_0 +- 1
    - AMCOs are zero if J + Jp + K is odd
    - abs(J - K) <= Jp <= J + K
    - kp - k != S

    Nonzero MADMs are worth exploring, and given information about angular momentum,
    this returns whether the wigner 3j matrices are nonzero and this index (J, J', k, k')
    is worth calculating
    """
    valid_Js = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]

    # Q must also be zero

    if J not in valid_Js or Jp not in valid_Js:
        return False

    if (J + Jp + K) % 2 != 0:
        return False

    if not abs(J - K) <= Jp <= J + K:
        return False

    if kp - k != S:
        return False

    return True


def find_nonzero_angular_momentum_MADM_pairs(J_0, K, S):
    """
    Parameters:
        - J_0, K, S
    Returns:
        - list of [J, k, J', k'] indices where the MADM wouldn't be zero by
        wigner 3j rules
        - a majority of them are zero, so this is highly useful to find nonzero elements
    """

    # Function to generate (J, k) permutations
    def generate_permutations(J_0):
        J_values = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]
        return [(J, k) for J in J_values for k in range(-J, J + 1)]

    # Generate all (J, k) pairs
    permutations = generate_permutations(J_0)

    # Generate all combinations of these pairs (J, k, J', k')
    combined_permutations = itertools.product(permutations, repeat=2)

    # Filter based on nonzero_madm condition
    return [
        [J, k, Jp, kp] for (J, k), (Jp, kp) in combined_permutations
        if nonzero_madm(J_0, J, Jp, k, kp, K, S)
    ]


def psi_norm(T2_eigenvectors_in_BO, states, J_0, M_0):
    """
    return normalization constant for psi
    """

    # what J values are possible, given J_0
    J_possibilities = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]  # By wigner rules, J = +- J_0

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

def find_diagonal_indices(n):
    flattened_indices = np.arange(n ** 2)
    diagonal_indices = flattened_indices[::n + 1]
    return diagonal_indices


def plot_fun_stuff(t_values, plotting, on_diagonal_madms, tol, elec1, elec2, K, Q, S, trace, do_tolerance):
    """
    Plots term of regular and specifically on_diagonal cases

    Tol (tolerance) is so you can filter out MADMs don't have a peak-trough difference of at least 'tol'
    - helps remove constant terms

    Also allows you to decide whether you are plotting the individual elements of the diagonal, or just the trace
    """

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for element in plotting:
        condition = abs(np.max(element) - np.min(element)) >= tol
        if do_tolerance:
            if condition:
                ax1.plot(t_values * time_conversion, element, linewidth=0.5)
        else:
            ax1.plot(t_values * time_conversion, element, linewidth=0.5)

    if trace:
        ax2.plot(t_values * time_conversion, on_diagonal_madms, linewidth=0.5)
    else:
        for element in on_diagonal_madms:
            ax2.plot(t_values * time_conversion, element, linewidth=0.5)

    fig.suptitle(f"Rovibronic MADMs K,Q,S={K},{Q},{S} over time, quanta={q}")

    ax1.set_title(f"{elec1}{elec2} Coherences")
    ax1.set_ylabel('Value')

    ax2.set_title(f"{elec1}{elec2} On-Diagonal (Same rot & vibration), Trace={trace}")
    ax2.set_ylabel('Value')

    plt.tight_layout()
    plt.show()

def MADM_for_angular_pairs(J, k, Jp, kp, T2_eigenvectors_in_BO, T2_BO_eigenvalues, elec1, elec2, M_0, J_0, K, Q, S):

    amco = AMCOs(J, Jp, k, kp, M_0, M_0, K, Q, S)

    # holds superposition of energy states for ease of calculation
    exp_1 = np.array([np.exp(-1j * (energy + E_jkm(B, J))) for energy in T2_BO_eigenvalues])
    exp_2 = np.array([np.exp(-1j * (energy + E_jkm(B, Jp))) for energy in T2_BO_eigenvalues])

    # Born_oppenheimer eigenvectors are devided into [000] X, [001] X... [000] Y, [001] Y, ... [000] Z, [001] Z
    # splitting it into 3 parts separates that section (Useful for MADMs only concerning X and Y combos, for instance)
    electronically_seperated = [np.array_split(np.array(v), 3) for v in T2_eigenvectors_in_BO]

    # Create a dictionary with keys for the 'X', 'Y', and 'Z' electronic components
    elec_dict = {"X": 0, "Y": 1, "Z": 2}

    # create the appropriate PSI portion with excitation amplitudes
    x = a(elec1, J, J_0, M_0, k) * np.array([vector[elec_dict[elec1]] for vector in electronically_seperated]) / psi_norm
    xp = a(elec2, Jp, J_0, M_0, kp) * np.array([vector[elec_dict[elec2]] for vector in electronically_seperated]) / psi_norm

    def madm(amco, x, xp, exp_1, exp_2, t):
        exponent_1 = np.power(exp_1, t)  # (e^-iE)^t = e^-iE^t
        psi_t = np.sum([vector * constant for vector, constant in zip(x, exponent_1)], axis=0)

        exponent_2 = np.power(exp_2, t)
        psi_prime_t = np.sum([vector * constant for vector, constant in zip(xp, exponent_2)], axis=0)

        # creates the time dependent density matrix * AMCO for this J, J', k, k' set. returns all the values from it
        return amco * np.outer(np.conj(psi_prime_t), psi_t).flatten()

    # save results
    MADMs = np.array([madm(amco, x, xp, exp_1, exp_2, t) for t in t_values]).T   # transpose so its each term over time

    # essentially returns the density matrix as MADMs (flattened, and for this J, J', k, k' pairing)

    return MADMs


def E_jkm(B, J):
    return B * J * (J + 1)


time_conversion = 1 / 41.341378  # AU =  41.3 * fs
M_0 = 0
# B = 1.71270 / 4.134e16  # AU
B = 2.603010273176e-7  #1.71270 Ghz to eV then to AU

Kt = 0.1
q = 10  # any less than 2 gets wonky

print(f"-------- q = {q} --------")

# indices_of_good_eigenvalues = [4, 5, 6, 15, 16, 17, 24, 25, 26]  # indices of the first 9 T2 values (3 triply degen)
indices_of_good_eigenvalues = [4, 15, 24]
# indices_of_good_eigenvalues = [4]  # indices of the first 9 T2 values (3 triply degen)
