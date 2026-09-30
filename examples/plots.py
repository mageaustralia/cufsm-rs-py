"""Draws the example plots into examples/: the CUFSM tutorial C (inches, ksi) and a 200 mm
lipped C (mm, MPa). Run:  python examples/plots.py   (needs pip install cufsm-rs-py[plot])"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import cufsm_rs as fsm  # noqa: E402
from cufsm_rs.plot import plot_mode, plot_section, plot_signature  # noqa: E402

out = pathlib.Path(__file__).parent

# The CUFSM tutorial C: 9 x 5 x 1 in, t = 0.1 in, 10 nodes, pure compression
xz = [(5, 1), (5, 0), (2.5, 0), (0, 0), (0, 3), (0, 6), (0, 9), (2.5, 9), (5, 9), (5, 8)]
tut = fsm.Model(
    prop=[[100, 29500, 29500, 0.3, 0.3, 11346.15]],
    node=[[i + 1, x, z, 1, 1, 1, 1, 0] for i, (x, z) in enumerate(xz)],
    elem=[[i + 1, i + 1, i + 2, 0.1, 100] for i in range(9)],
)
tut = fsm.stress(tut, P=fsm.first_yield(tut, 50).Py)

# 200 x 76 x 15 x 1.9 mm lipped C, 3 mm inside radius, major axis bending
lc = fsm.lipped_c(200, 76, 15, 1.9, ri=3)
lc_bend = fsm.stress(lc, Mxx=fsm.first_yield(lc, 450).Mxx)
lc_comp = fsm.stress(lc, P=fsm.first_yield(lc, 450).Py)


def save(ax, name):
    ax.figure.tight_layout()
    ax.figure.savefig(out / name, dpi=110)
    plt.close(ax.figure)
    print(out / name)


save(plot_section(tut, node_numbers=True), "section_tutorial_c.png")
save(plot_section(lc_bend), "section_lipped_c_bending.png")

sig_t = fsm.signature(tut, np.logspace(0, 3, 80))
save(plot_signature(sig_t, classify=True), "signature_tutorial_c.png")
sig_c = fsm.signature(lc_comp, np.logspace(1, 4, 90))
ax = plot_signature(sig_c, classify=True)
ax.set_title("200x76x15x1.9 lipped C, compression (lengths in mm)")
save(ax, "signature_lipped_c.png")

for k, (L, _) in enumerate(sig_t.minima):
    r = fsm.strip(tut, [L], neigs=1)
    save(plot_mode(r, 0), f"mode_tutorial_c_min{k + 1}.png")
for k, (L, _) in enumerate(sig_c.minima):
    r = fsm.strip(lc_comp, [L], neigs=1)
    save(plot_mode(r, 0), f"mode_lipped_c_min{k + 1}.png")
save(plot_mode(sig_c, len(sig_c.lengths) - 1), "mode_lipped_c_long.png")
cc = fsm.strip(lc_comp, [1500.0], m_all=10, bc="C-C", neigs=1)
save(plot_mode(cc, 0), "mode_lipped_c_cc_1500.png")
