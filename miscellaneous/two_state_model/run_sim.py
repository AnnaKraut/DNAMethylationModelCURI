import matplotlib.pyplot as plt
import numpy as np
import two_state_sim as gillespie_longrun

generator = np.random.default_rng()

methylated, unmethylated, times = gillespie_longrun.Gillespie(5000000, 100, 100, generator)


# print(methylated)
# print(unmethylated)
# print(times)

plt.plot(times, methylated, label="M")
plt.plot(times, unmethylated, label="U")
plt.plot(times, methylated + unmethylated, label="Total")
plt.legend(loc='lower right')
plt.show()