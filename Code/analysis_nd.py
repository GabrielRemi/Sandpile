# %%
from analysis import *
from utils import *
from pathlib import Path
from typing import Any


params = [(1, 20), (2, 40), (3, 20), (4, 15), (5, 15), (6, 10)]


# %%
fig, ax = plt.subplots()

for dim, grid in params:
    path_dir = Path(f"data/d{dim}_g{grid}_c7_cl_nco/data_0.slopes.npy")
    step, slope = load_slope_values(path_dir.__str__())

    draw_slope(slope, step, ax=ax, label=f"dim {dim} grid {grid}")

fig.legend(fontsize="small", loc="lower right", bbox_to_anchor=(0.97, 0.1))


# %%
from sandpile import generate_3d_distribution_from_data_sample


def calc_plot_scaling_exponents(meta: dict[str, Any], axs, do_plot: bool) -> pd.DataFrame:
    path_dir = Path(meta["path"])
    if not path_dir.joinpath("distribution.npz").exists():
        generate_3d_distribution_from_data_sample(path_dir)

    (s, t, r), bins = load_3d_dist(path_dir)

    return plot_scaling_exponents(s, t, r, bins, axs, meta.get("limits", None), do_plot)  # type: ignore


# %%
import json
from matplotlib.lines import Line2D
import utils
from pathlib import Path

fig = plt.figure(0, figsize=(18, 15))
fig.clf()
axs = fig.subplots(3, 3)

data_dir = pathlib.Path("data")
with open("scaling_exponents.json", "r") as f:
    meta = json.load(f)

# if specified, calculate scaling exponents for only one data set
do_plot = False
if isinstance(meta[0], int):
    meta = [meta[meta[0]]]
    do_plot = True

df = pd.DataFrame()
for m in meta:
    df_new = calc_plot_scaling_exponents(m, axs, do_plot)
    df = pd.concat([df, df_new]).reset_index(drop=True)

system_desc = []
for m in meta:
    path = Path(m["path"])
    system_desc.append(utils.get_short_params(utils.get_system_params_from_name(path.name)))
df.insert(0, "Dimension", [d["dimension"] for d in system_desc])
df.insert(1, "Grid Size", [d["linear_grid_size"] for d in system_desc])

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
labels = [m.get("label", "default") for m in meta]
handles = [Line2D([0], [0], color=color_cycle[i % len(color_cycle)], lw=2, label=labels[i]) for i in range(len(labels))]
fig.legend(
    handles=handles, labels=labels, ncols=len(labels), loc="center", bbox_to_anchor=(0.5, 1.01), fontsize="large"
)
plt.tight_layout()
# plt.show()
fig.savefig("figs/scaling_exponents_cl_nco.pdf")
fig.savefig("figs/scaling_exponents_cl_nco.png", dpi=300)

# %%
# Make Latex Table

df_table = df.map(lambda x: "{:.2uS}".format(x) if isinstance(x, unc.core.AffineScalarFunc) else x)
header = [
    "Dimension",
    "Grid Size",
    r"$\tau$",
    r"$\alpha$",
    r"$\lambda$",
    r"$\gamma_1$",
    r"$1/\gamma_1$",
    r"$\gamma_2$",
    r"$1/\gamma_2$",
    r"$\gamma_3$",
    r"$1/\gamma_3$",
]
# styler = df_table.style
# caption = """
# Scaling exponents for the critical state of the sandpile model with closed boundary conditions
# and conservative perturbations."""
# label = "tab:scaling_exponents_cl_nco"
df_table.to_latex(
    "tables/scaling_exponents_cl_nco.tex",
    index=False,
    escape=False,
    header=header,
    column_format=r"rr|ccccccccc",
)
# styler.hide(axis=0)
# styler.to_latex("tables/scaling_exponents_cl_nco.tex", column_format=r"rr|ccccccccc")
# %%
# comparison table
df_comp = df.copy()

df_comp["1/gamma1 inv"] = 1 / df["1/gamma1"]
df_comp["1/gamma2 inv"] = 1 / df["1/gamma2"]
df_comp["1/gamma3 inv"] = 1 / df["1/gamma3"]
df_comp["gamma2_new"] = df["gamma1"] * df["gamma3"]
df_comp["alpha_new"] = 2 + (df["lambda"] - 2) / df["gamma3"]
df_comp["tau_new"] = 2 + (df["lambda"] - 2) / df["gamma2"]
df_comp.drop(columns=["1/gamma1", "1/gamma2", "1/gamma3", "lambda"], inplace=True)
df_comp = df_comp[
    [
        "Dimension",
        "Grid Size",
        "alpha",
        "alpha_new",
        "tau",
        "tau_new",
        "gamma1",
        "1/gamma1 inv",
        "gamma2",
        "1/gamma2 inv",
        "gamma2_new",
        "gamma3",
        "1/gamma3 inv",
    ]
]
header = [
    "Dimension",
    "Grid Size",
    r"$\alpha$",
    r"$2 + (\lambda - 2)/\gamma_3$",  # alpha_new
    r"$\tau$",
    r"$2 + (\lambda - 2)/\gamma_2$",  # tau_new
    r"$\gamma_1$",
    r"$(1/\gamma_1)^{{-1}}$",
    r"$\gamma_2$",
    r"$(1/\gamma_2)^{{-1}}$",
    r"$\gamma_1 \gamma_3$",
    r"$\gamma_3$",
    r"$(1/\gamma_3)^{{-1}}$",
]
# header = [
#     "Dimension",
#     "Grid Size",
#     r"$\alpha$",
#     r"$2 + (\lambda - 2)/\gamma_3$",  # alpha_new
#     r"$\tau$",
#     r"$2 + (\lambda - 2)/\gamma_2$",  # tau_new
#     r"$\gamma_1$",
#     r"$(1/\gamma_1)^{{-1}}$",
#     r"2",
#     r"2",
#     r"2",
#     r"2",
#     r"2",
# ]
# header = ["2"] * 13
df_comp.to_latex(
    "tables/scaling_exponents_comparison_cl_nco.tex",
    header=header,
    index=False,
    escape=False,
    column_format=r"rr|ccccccccccc",
)

# %%
