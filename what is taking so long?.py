from functions import *
import matplotlib.pyplot as plt
import numpy as np
from time import time


def full_matrix_madm_over_time(q, J_0, K, Q, S, elec1, elec2, Kt, indices_of_good_eigenvalues, B, M_0, t_values):
    """
    Calculate MADM over time given initial parameters.
    """
    timing_data = {}

    # Get possible quantum states
    states = get_possible_states(q)

    # Calculate Hamiltonian matrix
    start_time = time()
    hamiltonian = np.array([[matrix_elem(state1, state2).real for state2 in states] for state1 in states])
    timing_data['Calculate Hamiltonian'] = time() - start_time

    # Get eigenvectors and eigenvalues
    start_time = time()
    T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, Kt, indices_of_good_eigenvalues)
    timing_data['Eigenvectors and Eigenvalues'] = time() - start_time

    # Get psi_0
    start_time = time()
    psi_0, psi_vector_parts = psi(T2_eigenvectors_in_BO, T2_BO_eigenvalues, states, B, J_0, M_0)
    timing_data['Get psi_0'] = time() - start_time

    # Reduce vector parts
    reduced_vector_parts = psi_vector_parts[:len(psi_vector_parts) // len(indices_of_good_eigenvalues)]

    # Get the time-independent AMCOs
    start_time = time()
    amco_list = full_amco(J_0, K, Q, S, reduced_vector_parts)
    timing_data['Get AMCOs'] = time() - start_time

    # Calculate constants for time-dependent calculations
    constant = np.array([np.exp(-1j * energy.E) for energy in psi_vector_parts])

    def madm(t):
        # Get psi(t)
        num_psis = len(indices_of_good_eigenvalues)
        psi_t = psi_t_function(psi_0, constant, t, num_psis)

        # Separate the x, y, and z electronic components
        psi_x_t, psi_y_t, psi_z_t = np.array_split(psi_t, 3)

        # Create a dictionary with keys for the 'X', 'Y', and 'Z' electronic components
        elec_dict = {"X": psi_x_t, "Y": psi_y_t, "Z": psi_z_t}

        # For the correct electronic pair, calculate that MADM for this time 't'
        MADM = np.outer(np.conj(elec_dict[elec2]), elec_dict[elec1]) * np.array(amco_list)
        return MADM

    plotting_values = []

    start_time = time()
    for t in t_values:
        plotting_values.append(madm(t).flatten())
    timing_data[f'Computing {len(t_values)} MADMs'] = time() - start_time

    # Plotting values
    plotting = np.vstack(plotting_values).T

    # Calculate total time taken
    total_time = sum(timing_data.values())

    # Create pie chart with timing information
    labels = [f"{key}: {value:.2f}s ({(value / total_time) * 100:.1f}%)" for key, value in timing_data.items()]
    times = list(timing_data.values())
    plt.figure(figsize=(10, 7))
    plt.pie(times, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Time taken by different parts of the function ({round(total_time,1)}s)')
    plt.show()

    return plotting


# Example usage
start = 0 / time_conversion  # convert from fs to AU
end = 50 / time_conversion  # convert from fs to AU
num_steps = 50
t0 = time()
t_values = np.linspace(start, end, num_steps)

q = 10
K = 2
Q = 0
S = 0
J_0 = 0
elec1 = "X"
elec2 = "Y"


plotting = full_matrix_madm_over_time(q, J_0, K, Q, S, elec1, elec2, Kt=Kt,
                                      indices_of_good_eigenvalues=indices_of_good_eigenvalues, B=B, M_0=M_0,
                                      t_values=t_values)
print(f"Time to execute: {round(time() - t0, 2)}s for {len(plotting)} MADM elements")
