from nonzero_madm_functions import *
import matplotlib.pyplot as plt
from math import comb

def hamil(q, Kt, states=None):
    # set up hamiltonian

    if states is None:
        states = get_possible_states(q)

    H = np.array([[matrix_elem(state1, state2).real for state2 in states] for state1 in states])
    mask = ~np.eye(H.shape[0], dtype=bool)
    H[mask] *= Kt

    return H


def Lanczos(H, vg, num_vectors=None):
    """
    LANCZOS ALGORITHM

    :param H: Original Hamil
    :param vg: Initial Vector from Symmetry Consideration
    :param num_vectors: Can continue until the algorithm breaks or select a finite number of vectors
    :return: Tri-diagonal Matrix of order (num_vectors) with (ideally) the eigenvalues of that symmetry
    """

    if num_vectors is None:
        num_vectors = len(vg)

    P = np.zeros((H.shape[0], num_vectors), dtype=complex)  # Transformation matrix
    T = np.zeros((num_vectors, num_vectors), dtype=complex)  # New Hamil
    P[:, 0] = vg / la.norm(vg)  # Normalizes initial vector

    # Performs the first iteration step of the Lanczos algorithm
    w = np.dot(H, P[:, 0])
    a = np.dot(np.conj(w), P[:, 0])
    w = w - a * P[:, 0]
    T[0, 0] = a

    # Continue until number of desired vectors/iterations is complete
    for j in range(1, num_vectors):

        b = la.norm(w)

        # if b is near zero, the algorithm is complete and stops early
        if b < 1e-4:
            # print("Break Lanczos algorithm due to small b.")
            # print(f"b={b}, b[prior]={T[j - 2, j - 1]}")
            print(f"Break Lanczos algorithm due to small b: {b:.2e} at iteration {j} of {num_vectors}")
            # T = T[:j, :j]
            # P = P[:, :j]
            break

        P[:, j] = w / b

        # Orthogonalize the newly generated vector for extra precision
        for k in range(j):
            P[:, j] -= np.dot(np.conj(P[:, j]), P[:, k]) * P[:, k]
        P[:, j] /= la.norm(P[:, j])  # Re-normalize after orthogonalization


        w = np.dot(H, P[:, j])
        a = np.dot(np.conj(w), P[:, j])
        w = w - a * P[:, j] - b * P[:, j - 1]


        # Creates tridiagonal matrix T using a and b values
        T[j, j] = a
        T[j - 1, j] = b
        T[j, j - 1] = np.conj(b)

    # works!!!
    X = np.real(np.dot(np.conj(P.T), np.dot(H, P)).round(7))


    return T, P


def Lanczos_hamil(q, Kt, H=None, num_vectors=None):
    if H is None:
        H = hamil(q, Kt)

    # Choose an initial vector
    v = np.zeros(H.shape[0])
    # v[0] = 1      # for T1
    v[1] = -1 / np.sqrt(2)
    v[2] = 1 / np.sqrt(2)


    T, P = Lanczos(H, v, num_vectors)
    diag = np.real(np.diag(T))
    sub_diag = np.real(np.diag(T, -1))

    eigenvalues = eigh_tridiagonal(diag, sub_diag, eigvals_only=True)
    """    # has a lowest eignevalue of 0, which is not valid, so don't use those
    eigenvalues = eigh(T, eigvals_only=True)
    eigenvalues = eigenvalues[eigenvalues != 0]"""

    return eigenvalues


def states_in_chunk(state1, q):
    """
    There are four chunks in this hamiltonian, and each is found by determining which states
    interact with the basic one.

    First chunk is what interacts with 000X, next is 000Y, 3rd is 000Z, and the last is what is left over


    :param state1: Ex. |000| |x>
    :param q: Number of quanta
    :return: List of states for that chunk
    """

    states = get_possible_states(q)
    states_in_this_chunk = [state1]

    states.remove(state1)

    for state1 in states_in_this_chunk:
        for state2 in states:
            if np.real(matrix_elem(state1, state2)) > 0:
                states_in_this_chunk.append(state2)
                states.remove(state2)


    return states_in_this_chunk

def states_in_chunk(state1, q):

    states = get_possible_states(q)
    for state in [[0, 0, 0, 'X'], [0, 0, 1, 'Y'], [0, 1, 0, 'Z'], [1, 0, 1, 'Z'], [0, 0, 2, 'X'], [0, 2, 0, 'X']]:
        for n in states:
            if np.real(matrix_elem(state, n)) > 0:
                #print(state, n)
                pass

    states_in_this_chunk = [state1]

    # already contains the one 000X/y/z state, so continue from q=1
    for i in range(1, q+1):
        state_list = distribute_balls(i)

        electronic = ["X", "Y", "Z"]

        # provide all states
        for vib in electronic:
            for item in state_list:
                state = item + [vib]

                for chunk_state in states_in_this_chunk:
                    if state in states_in_this_chunk:
                        continue

                    if np.real(matrix_elem(chunk_state, state)) > 0:
                        states_in_this_chunk.append(state)

    return states_in_this_chunk

def states_in_chunk3(state1, q):


    states_in_this_chunk = [state1]

    # already contains the one 000X/y/z state, so continue from q=1
    S = []
    for i in range(1, q + 1):
        state_list = distribute_balls(i)

        electronic = ["X", "Y", "Z"]

        # provide all states
        for vib in electronic:
            for item in state_list:
                S.append(item + [vib])


    for state in states_in_this_chunk:
        for state2 in S:
            if state2 in states_in_this_chunk:
                continue
            if np.real(matrix_elem(state, state2)) > 0:
                states_in_this_chunk.append(state2)


    return states_in_this_chunk

def distribute_balls(q):
    """
    :param q: Distribute the quanta into three buckets
    :return: combinations of these
    """

    combinations = []
    for distribution in product(range(q + 1), repeat=3):
        if sum(distribution) == q:
            combinations.append(list(distribution))

    # combinations.append(combinations.pop(0))    # optional... moves first item to end

    return combinations


def plot_hamil(states, Kt):
    """
    Given the list of states and Kt, shows the hamiltonian plotted on a 2d histogram

    :param states: List of states [000X, 001Y, etc...]
    :param Kt: Number
    :return: Shows matrix visually
    """

    # create the matrix
    H = np.array([[matrix_elem(state1, state2).real for state2 in states] for state1 in states])

    # Multiply off diagonal terms by Kt to make it correct
    mask = ~np.eye(H.shape[0], dtype=bool)
    H[mask] *= Kt

    # Make elements that are zero NaNs, that way it is not plotted
    H = np.where(H == 0, np.nan, H)

    # Plotting the matrix as a 2D histogram using imshow
    plt.figure(figsize=(8, 6))
    plt.imshow(H, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Values')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Histogram of Matrix')
    plt.show()


def plot_OG_versus_Lanczos(q, Kt_list, order_of_states1, order_of_states2, num_vectors):
    eigenvalues_list = []
    lanscos_version = []


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Creating a figure with 2 subplots side by side

    H_full = hamil(q, 1, order_of_states1)
    mask_full = ~np.eye(H_full.shape[0], dtype=bool)

    H_part = hamil(q, 1, order_of_states2)
    mask_part = ~np.eye(H_part.shape[0], dtype=bool)

    print(num_vectors, "/", H_part.shape[0], "Vectors Selected in Chunk")

    for i, Kt in enumerate(Kt_list):
        print(f"{i / len(Kt_list) * 100}%")

        H_full_kt = np.copy(H_full)
        H_full_kt[mask_full] *= Kt

        H_part_kt = np.copy(H_part)
        H_part_kt[mask_part] *= Kt

        eigenvalues = eigh(H_full_kt, eigvals_only=True)
        eigenvalues[eigenvalues != 0] += (2 * Kt ** 2 / 3)
        eigenvalues_list.append(eigenvalues)

        eigenvalues = Lanczos_hamil(q, Kt, H_part_kt, num_vectors)
        eigenvalues[eigenvalues != 0] += (2 * Kt ** 2 / 3)
        lanscos_version.append(eigenvalues)

    eigenvalues_list = np.array(eigenvalues_list).T
    lanscos_version = np.array(lanscos_version).T


    for eig in lanscos_version:
        axs[0].scatter(Kt_list, eig, s=15)  # Plotting in the first subplot

    axs[0].set_title(f'Lanczos Version Eigenvalues')  # Adding title to the first subplot
    axs[0].set_xlabel('Kt')
    axs[0].set_ylabel('Eigenvalue (Units of hbar w)')
    axs[0].set_ylim(1, 10)  # Setting y-axis limits for the first subplot

    # Plotting the original eigenvalues in the second subplot
    for eigenvalues in eigenvalues_list:
        axs[1].scatter(Kt_list, eigenvalues, s=15)

    axs[1].set_title('Original Eigenvalues')  # Adding title to the second subplot
    axs[1].set_ylim(1, 10)  # Setting y-axis limits for the second subplot
    axs[1].set_xlabel('Kt')
    axs[1].set_ylabel('Eigenvalue (Units of hbar w)')