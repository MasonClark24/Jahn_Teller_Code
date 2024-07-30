from functions import *
from Lanczos_functions import *
import matplotlib.pyplot as plt
from math import comb



q = 6
total_possible_vectors = np.sum([comb(i + 3 - 1, 3 - 1) for i in range(q + 1)]) * 3  # total number of states! Cool

# get the states belonging to the first three chunks
X = states_in_chunk([0, 0, 0, 'X'], q)
print(X)
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
