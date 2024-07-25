import numpy as np

from functions import *
import matplotlib.pyplot as plt
from math import comb


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



q = 6
total_possible_vectors = np.sum([comb(i + 3 - 1, 3 - 1) for i in range(q + 1)]) * 3  # total number of states! Cool

# get the states belonging to the first three chunks
X = states_in_chunk([0, 0, 0, 'X'], q)
Y = states_in_chunk([0, 0, 0, 'Y'], q)
Z = states_in_chunk([0, 0, 0, 'Z'], q)

# get the states left over, making up the final chunk
A_E = get_possible_states(q)
for item in X + Y + Z:
    if item in A_E:
        A_E.remove(item)

state_order = X + Y + Z + A_E

# doesn't need to be all three! Can just do X, for example
first_3 = X + Y + Z


# plot things
plot_hamil(first_3, 0.1)
# num_vectors = find_number_of_t2_vectors(q, X)
num_vectors = len(first_3)

plot_OG_versus_Lanczos(q, np.linspace(0, 3, 100),
                       order_of_states1=first_3, order_of_states2=first_3, num_vectors=num_vectors)
# plot_OG_versus_Lanczos(q, np.linspace(0.1, 3, 100), Y, num_vectors=num_vectors)
# plot_OG_versus_Lanczos(q, np.linspace(0.1, 3, 100), Z, num_vectors=num_vectors)
plt.suptitle(f"Top Left Block, q={q}    No Seperation of T1 Vs. T2")
plt.tight_layout()  # Adjusting layout to prevent overlapping
plt.show()
