from functions import *
import matplotlib.pyplot as plt
from time import time

"""
To do:
Gaussian to filter around specific energy
"""


def madm(t, psi_0, constant, elec1, elec2, amco_list):
    # get psi(t)
    num_psis = len(indices_of_good_eigenvalues)
    psi_t = psi_t_function(psi_0, constant, t, num_psis)

    # separate the x, y, and z electronic components
    psi_x_t, psi_y_t, psi_z_t = np.array_split(psi_t, 3)

    # Create a dictionary with keys for the 'X', 'Y', and 'Z' electronic components
    elec_dict = {"X": psi_x_t,
                 "Y": psi_y_t,
                 "Z": psi_z_t}

    # for the correct electronic pair, calculate that MADM for this time 't'
    MADM = np.outer(np.conj(elec_dict[elec2]), elec_dict[elec1]) * np.array(amco_list)

    return MADM.flatten()   # np.where(np.real(np.round(MADM, 7))==0, np.nan, np.real(np.round(MADM, 7)))


def full_matrix_madm_over_time(q, J_0, K, Q, S, elec1, elec2, Kt=Kt,
                               indices_of_good_eigenvalues=indices_of_good_eigenvalues, B=B,
                               M_0=M_0):
    """
    Given initial information, return all elements of the (elec1, elec2) MADM over time
    """

    # Get possible quantum states
    states = get_possible_states(q)

    # Calculate Hamiltonian matrix
    hamiltonian = np.array([[matrix_elem(state1, state2).real for state2 in states] for state1 in states])

    # Get eigenvectors and eigenvalues
    T2_eigenvectors_in_BO, T2_BO_eigenvalues = eigenvector_and_eigenvalues(hamiltonian, Kt, indices_of_good_eigenvalues)

    # get psi_0
    psi_0, psi_vector_parts = psi(T2_eigenvectors_in_BO, T2_BO_eigenvalues, states, B, J_0, M_0)

    # get the order of Js and Ks
    reduced_vector_parts = psi_vector_parts[:len(psi_vector_parts) // len(indices_of_good_eigenvalues)]

    for x in reduced_vector_parts:
        print(x)



    # Gets the time-independent AMCOs
    amco_list = full_amco(J_0, K, Q, S, reduced_vector_parts)

    # get constants for time_dependent calculations. e^-i*E raised to the ^t is e^-i*Et
    constant = np.array([np.exp(-1j * energy.E) for energy in psi_vector_parts])


    plotting_values = []

    for t in t_values:
        plotting_values.append(madm(t, psi_0, constant, elec1, elec2, amco_list))

    return np.vstack(plotting_values).T


start = 0 / time_conversion  # convert from fs to AU
end = 50 / time_conversion  # convert from fs to AU
num_steps = 50
t0 = time()
t_values = np.linspace(start, end, num_steps)

K = 2
Q = 0
S = 0
J_0 = 1


elec1 = "X"
elec2 = "Y"


# get madm info for plotting
plotting = full_matrix_madm_over_time(q, J_0, K, Q, S, elec1, elec2, Kt=Kt,
                                      indices_of_good_eigenvalues=indices_of_good_eigenvalues, B=B, M_0=M_0)


print(f"Time to execute: {round(time() - t0, 2)}s for {len(plotting)} MADM elements")

real = np.real(plotting)
imag = np.imag(plotting)


def plot_fun_stuff(plotting):

    # initialize array for the traced same-vibration MADM
    on_diagonal_MADM = np.zeros_like(plotting[0], dtype=complex)

    # get indices of the on-diagonal (same vibration) elements
    diagonal_indices = find_diagonal_indices(int(np.real(sqrt(len(plotting)))))

    # Plot the traced result and the off-diagonal ones separately
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for i, element in enumerate(plotting):
        # print(i, "plotted of", len(plotting))
        # keep the diagonal one separate for comparison
        if i in diagonal_indices:
            on_diagonal_MADM += element

        # off diagonal elements in vibration. beware, higher quanta means this plot gets messy quickly
        else:
            condition = abs(np.min(element) - np.max(element)) > 1e-8
            if condition:
                ax1.plot(t_values * time_conversion, element, linewidth=0.5)

    # Plot diagonal elements
    ax2.plot(t_values * time_conversion, on_diagonal_MADM, linewidth=1, label=f"Same Vib. MADM {elec1}{elec2}",
             color="Black")

    ax1.set_title(f"{elec1}{elec2} Full MADM K=2 with time, quanta={q}")
    ax1.set_ylabel('Value')
    ax2.set_xlabel('Time (fs)')
    ax2.legend()

    plt.tight_layout()
    plt.show()



plot_fun_stuff(real)
# plot_fun_stuff(imag)


"""# Perform Fourier transform on each sublist
fft_result_real = np.fft.fft(real, axis=1)
fft_result_imag = np.fft.fft(imag, axis=1)

# Compute the average of the absolute values of the Fourier transforms
average_fft_real = np.mean(np.abs(fft_result_real), axis=0)
average_fft_imag = np.mean(np.abs(fft_result_imag), axis=0)

# Compute the corresponding frequencies
num_points = real.shape[1]
frequencies = np.fft.fftfreq(num_points)

# Plot both real and imaginary parts on the same figure
plt.figure(figsize=(10, 6))
plt.plot(frequencies, average_fft_real, label="Real")
plt.plot(frequencies, average_fft_imag, label="Imaginary")
plt.title('Average Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.legend()
plt.yscale("log")
plt.show()"""
