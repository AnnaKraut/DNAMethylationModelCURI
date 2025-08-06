import numpy as np
import gillespie_time as gillespie_time
import matplotlib.pyplot as plt
from highlight_text import fig_text  # for coloring the title
import dill

# load data
dill.load_session("P8/two_way_sim_data.pkl")

plt.close()
final_label = "Two-way switching directions with Population = 100"
run_stats = (
    "Batches of "
    + str(batch_size)
    + ", running for maximum of "
    + str(trial_max_length)
    + " steps each"
)

plt.rcParams["font.size"] = 20



plt.subplot(2, 1, 1)

fig_text(
    x=0.5,
    y=.96,  # Position of the title (relative to the figure)
    s=run_stats
    + "\n"
    + "<Hyper-to-Hypomethylated>                 <Hypo-to-Hypermethylated>",
    highlight_textprops=[{"color": c} for c in ["#0072B2", "#D55E00"]],  # Apply colors
    ha="center",  # Horizontal alignment
    fontsize=20,
)

# just some magic to make the legend look nice
prop1, = plt.plot(step_array, timeouts_MtoU, color="#0072B2")
prop2, = plt.plot(step_array, timeouts_UtoM, color="#D55E00")

kserror1, = plt.plot(
    step_array, exponential_KS_MtoU, linestyle="dashed", color="#0072B2"
)
kserror2, = plt.plot(
    step_array, exponential_KS_UtoM, linestyle="dashed", color="#D55E00"
)

handles = [prop1, kserror1, prop2, kserror2]
labels = ["", "", "Proportion timed out", "Exponential KS error"]
plt.legend(handles, labels, ncol=2, columnspacing=-1, loc="upper right")

plt.ylabel("Proportion", size="small")

plt.subplot(2, 1, 2)
param1, = plt.plot(
    step_array,
    1 / np.array(exponential_parameters_MtoU),
    color="#0072B2",
)
param2, = plt.plot(
    step_array,
    1 / np.array(exponential_parameters_UtoM),
    color="#D55E00",
)

plt.legend((param1,param2), ("", "Exponential parameter"), ncol=2, columnspacing=-1, loc="upper right")
plt.ylabel("Exponential parameter of\nswitching time distribution", size="small")
plt.xlabel("Birth rate")

plt.show()
