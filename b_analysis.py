from Lanczos_functions import *
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


def plot_OG_versus_Lanczos_with_b(q, Kt_list, order_of_states1, order_of_states2, v, num_vectors):
    eigenvalues_list = []
    lanscos_version = []
    b_matrix = []
    lanczos_step_totals = []  # List to store the shape of matrices T

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))  # Creating a figure with 2x3 subplots


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

        T, P, b_list, num_iterations = Lanczos_track_b(H_part_kt, v, num_vectors=num_vectors)
        eigenvalues = eigh_tridiagonal(np.real(np.diag(T)), np.real(np.diag(T, -1)), eigvals_only=True)
        eigenvalues[eigenvalues != 0] += (2 * Kt ** 2 / 3)
        lanscos_version.append(eigenvalues)

        b_matrix.append(b_list)
        lanczos_step_totals.append(num_iterations)

    eigenvalues_list = np.array(eigenvalues_list).T
    lanscos_version = np.array(lanscos_version).T

    for eig in lanscos_version:
        axs[0, 0].scatter(Kt_list, eig, s=5)  # Plotting in the first subplot
    axs[0, 0].set_title(
        f'Lanczos Version Eigenvalues')  # Adding title to the first subplot
    axs[0, 0].set_xlabel('Kt')
    axs[0, 0].set_ylabel('Eigenvalue (Units of hbar w)')
    axs[0, 0].set_ylim(1, 10)  # Setting y-axis limits for the first subplot

    for eigenvalues in eigenvalues_list:
        axs[0, 1].scatter(Kt_list, eigenvalues, s=5)
    axs[0, 1].set_title('Original Eigenvalues')  # Adding title to the second subplot
    axs[0, 1].set_ylim(1, 10)  # Setting y-axis limits for the second subplot
    axs[0, 1].set_xlabel('Kt')
    axs[0, 1].set_ylabel('Eigenvalue (Units of hbar w)')

    for b_list in b_matrix:
        axs[0, 2].plot(range(1, len(b_list) + 1), b_list, marker='o', linestyle='-', color='b', markersize=1)
    axs[0, 2].axhline(y=tol, color='black', linestyle='--', label=f'y = {tol}')
    axs[0, 2].set_xlabel('Iteration Index')
    axs[0, 2].set_ylabel('b Value')
    axs[0, 2].set_title('b Values vs. Iteration Index')
    axs[0, 2].set_yscale('log')
    axs[0, 2].grid(True)

    # Plotting the histogram of T shapes in the second row
    axs[1, 1].hist(lanczos_step_totals, bins=len(lanczos_step_totals), alpha=0.5)
    axs[1, 1].set_xlabel('lanczos step totals')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Histogram of lanczos step totals')

    plt.tight_layout()  # Adjust layout to prevent overlap

    axs[1, 0].plot(Kt_list, lanczos_step_totals, marker='o', linestyle='-', color='b')
    axs[1, 0].set_xlabel('Kt')
    axs[1, 0].set_ylabel('termination number')
    axs[1, 0].set_title('termination number vs. Kt')
    axs[1, 0].grid(True)

    # Plot b_list iteration indexes for scenarios above average
    # Find scenarios where the final iteration index is above the average
    avg_final_iteration = np.mean(lanczos_step_totals)
    b_above_avg = [b_list for b_list in b_matrix if len(b_list) > avg_final_iteration]

    for b_list in b_above_avg:
        axs[1, 2].plot(range(1, len(b_list) + 1), b_list, marker='o', linestyle='-')
    axs[1, 2].axhline(y=tol, color='black', linestyle='--', label=f'y = {tol}')
    axs[1, 2].set_xlabel('Iteration Index')
    axs[1, 2].set_ylabel('b Value')
    axs[1, 2].set_title(f'b Values vs. Iteration Index ({len(b_above_avg)} are > {avg_final_iteration} iter.)')
    axs[1, 2].set_yscale('log')
    axs[1, 2].grid(True)

    print(avg_final_iteration)

def Lanczos_track_b(H, vg, num_vectors=None):
    """
    LANCZOS ALGORITHM

    :param H: Original Hamiltonian matrix
    :param vg: Initial vector from symmetry consideration
    :param num_vectors: Number of desired vectors/iterations (default is None, which means use the length of vg)
    :return: Tri-diagonal matrix T of order num_vectors with eigenvalues of the original matrix H
    """

    if num_vectors is None:
        num_vectors = len(vg)

    P = np.zeros((H.shape[0], num_vectors), dtype=complex)  # Transformation matrix
    T = np.zeros((num_vectors, num_vectors), dtype=complex)  # New Hamiltonian matrix
    P[:, 0] = vg / la.norm(vg)  # Normalizes initial vector

    b_values = []  # List to store b values at each iteration
    num_iterations = None

    # Performs the first iteration step of the Lanczos algorithm
    w = np.dot(H, P[:, 0])
    a = np.dot(np.conj(w), P[:, 0])
    w = w - a * P[:, 0]
    T[0, 0] = a

    # Continue until the number of desired vectors/iterations is complete
    # then if the next step is a lower b, include it. otherwise, end
    one_more_b = None
    final_step = True

    for j in range(1, num_vectors):

        b = la.norm(w)

        b_values.append(b)  # Store b value for this iteration

        # If b is near zero, the algorithm is complete and stops early
        if b < tol:
            print("Break Lanczos algorithm due to small b.", j, "/", len(vg))
            num_iterations = j
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

    X = np.real(np.dot(np.conj(P.T), np.dot(H, P)).round(7))

    if num_iterations is None:
        num_iterations = num_vectors

    return T, P, b_values, num_iterations


q = 6
num_vectors = 60
X = states_in_chunk([0, 0, 0, 'X'], q)  # get the states belonging to the first three chunks
v = np.zeros(len(X))
print(X)
tol = 1e-2
# v[0] = 1
v[1] = -1 / np.sqrt(2)
v[2] = 1 / np.sqrt(2)
total_possible_vectors = np.sum([comb(i + 3 - 1, 3 - 1) for i in range(q + 1)]) * 3  # total number of states! Cool
print(np.sum([comb(i + 3 - 1, 3 - 1) for i in range(200 + 1)]) * 3)  # total number of states! Cool


plot_OG_versus_Lanczos_with_b(q, np.linspace(0.1, 3, 100),
                       order_of_states1=X, order_of_states2=X, v=v, num_vectors=num_vectors)

plt.suptitle(f"Top Left Block, q={q}")
plt.tight_layout()  # Adjusting layout to prevent overlapping
plt.show()
