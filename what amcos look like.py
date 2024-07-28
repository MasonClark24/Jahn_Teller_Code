import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
from functions import *


def nonzero_madm(J_0, J, Jp, k, kp, K, S):
    # valid_Js = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]

    # if AMCOs(J, Jp, k, kp, M_0, M_0, K, 0, S) != 0 and a('X', J, J_0, M_0, k) != 0 and a('X', Jp, J_0, M_0, kp) != 0:
    # print(J_0, J, Jp, k, kp, K, S, "section 1")

    # if J not in valid_Js or Jp not in valid_Js:
    # return False

    if (J + Jp + K) % 2 != 0:
        return False

    if not abs(J - K) <= Jp <= J + K:
        return False

    if kp - k != S:
        return False

    # print(J_0, J, Jp, k, kp, K, S, " section 2")
    # print()
    return True


def generate_permutations(J_0):
    J_values = [j for j in [J_0 - 1, J_0 + 1] if j >= 0]
    return [(J, k) for J in J_values for k in range(-J, J + 1)]


# Parameters
K = 2
Q = 0
S = 0
J_0 = 8
elec1 = "X"
elec2 = "X"

full_amcos = []

for J_0 in range(0, 20):
    # Generate all (J, k) pairs
    permutations = generate_permutations(J_0)

    print(f"For J_0 = {J_0}")

    expected_nonzero = 0
    num_amcos = 0
    nonzero_amcos = 0
    full_amcos = []
    for (J, k) in permutations.copy():
        amco = []
        for (Jp, kp) in permutations.copy():
            M = AMCOs(J, Jp, k, kp, M_0, M_0, K, Q, S)
            amco.append(M)

            if nonzero_madm(J_0, J, Jp, k, kp, K, S):
                expected_nonzero += 1
            if M != 0:
                nonzero_amcos += 1

            num_amcos += 1

        full_amcos.append(amco)



    print(f"Full number of amcos is {num_amcos}")
    print(f"Expected nonzero amount is {expected_nonzero}")
    print(f"Actual nonzero amount is {nonzero_amcos}")
    print("-------\n")





# Convert AMCO list to numpy array for easier manipulation
amco_array = np.array(full_amcos).T
amco_array[amco_array == 0] = np.nan

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(amco_array[::-1, ::-1], cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label='AMCO Value')
plt.xlabel('Index')
plt.ylabel('Index')
plt.title(f'2D Plot of AMCO List for {elec1}-{elec2} Pair')
plt.show()

