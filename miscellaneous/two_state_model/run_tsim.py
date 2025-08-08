import matplotlib.pyplot as plt
import numpy as np
import two_state_treatment as gillespie_longrun

generator = np.random.default_rng()

treatments = np.array([[50,10], [100,10]])

methylated, unmethylated, treatment, times = gillespie_longrun.Gillespie(500000, 10, 100, treatments, generator)


# print(methylated)
# print(unmethylated)
# print(times)

plt.subplot(2,1,1)

plt.plot(times, methylated, label="M")
plt.plot(times, unmethylated, label="U")
plt.legend(loc='lower right')

plt.subplot(2,1,2)

plt.plot(times, treatment, label="treatment")
plt.legend(loc='lower right')

# plt.plot(times, methylated + unmethylated, label="Total")
plt.show()