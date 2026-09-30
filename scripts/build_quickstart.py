"""Builds examples/quickstart.ipynb from the cells below and executes it in place (outputs saved).

Needs the dev extra (nbclient, nbformat, ipykernel) and matplotlib. Run: python scripts/build_quickstart.py"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parents[1] / "examples/quickstart.ipynb")
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md("""# cufsm-rs-py quickstart

Finite strip buckling of thin-walled sections (CUFSM) from Python. The engine is
[cufsm-rs](https://github.com/mageaustralia/cufsm-rs), the Rust port of CUFSM that runs CivilKit
Buckling in the browser; this package makes the same engine available here.

**Units.** The package is unit-agnostic, as CUFSM is: use any consistent set. The tutorial C
below is in inches and ksi (CUFSM's own example); the lipped C is in mm and MPa.

**Load factors** multiply the model's reference stresses. With reference stresses from the
squash load `Py`, a load factor reads as `Pcr / Py`."""),
    code("""import numpy as np
import cufsm_rs as fsm
from cufsm_rs.plot import plot_section, plot_signature, plot_mode
fsm.__version__"""),
    md("""## The CUFSM tutorial C (inches, ksi)

A model is CUFSM's arrays: `prop` `[mat#, Ex, Ey, vx, vy, G]`, `node`
`[node#, x, z, xdof, zdof, ydof, qdof, stress]`, `elem` `[elem#, nodei, nodej, t, mat#]`."""),
    code("""xz = [(5, 1), (5, 0), (2.5, 0), (0, 0), (0, 3), (0, 6), (0, 9), (2.5, 9), (5, 9), (5, 8)]
tut = fsm.Model(
    prop=[[100, 29500, 29500, 0.3, 0.3, 11346.15]],
    node=[[i + 1, x, z, 1, 1, 1, 1, 0] for i, (x, z) in enumerate(xz)],
    elem=[[i + 1, i + 1, i + 2, 0.1, 100] for i in range(9)],
)
tut"""),
    code("fsm.section_properties(tut)"),
    code("""y = fsm.first_yield(tut, fy=50)   # extreme fibre (element faces), as current CUFSM
y"""),
    code("""tut_c = fsm.stress(tut, P=y.Py)   # reference stress: the squash load
plot_section(tut_c, node_numbers=True);"""),
    code("""sig = fsm.signature(tut_c, np.logspace(0, 3, 80))
sig"""),
    code("plot_signature(sig, classify=True);"),
    code("""# G, D, L, O percentages of the lowest mode at each minimum
dict(zip(["first", "second"], np.round(sig.classify_minima(), 1).tolist()))"""),
    md("""## Mode shapes

`mode_shape(i_length, k_mode)` splits a mode into per-node arrays `u`, `v`, `w`, `theta`, each
of shape `(terms, nodes)`; the raw vector in CUFSM's DOF order stays on `.dofs`."""),
    code("""i = int(np.argmin(np.abs(sig.lengths - sig.minima[0, 0])))   # the length nearest the first minimum
ms = sig.mode_shape(i)
print(ms.length, ms.load_factor, ms.u.shape)
plot_mode(sig, i);"""),
    md("## A 200 x 76 x 15 x 1.9 mm lipped C (mm, MPa)"),
    code("""lc = fsm.lipped_c(200, 76, 15, 1.9, ri=3)   # outside dimensions, inside radius; E = 203000 MPa
p = fsm.section_properties(lc)
yl = fsm.first_yield(lc, 450)
print(f"A = {p.A:.1f} mm2, Ixx = {p.Ixx:.4g} mm4, Py = {yl.Py / 1e3:.1f} kN, Mxx = {yl.Mxx / 1e6:.2f} kNm")
lc_c = fsm.stress(lc, P=yl.Py)
sig_lc = fsm.signature(lc_c, np.logspace(1, 4, 90))
sig_lc"""),
    code("""ax = plot_signature(sig_lc, classify=True)
ax.set_ylabel("Pcr / Py");"""),
    code("""import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, (L, _) in zip(axs, sig_lc.minima):
    plot_mode(fsm.strip(lc_c, [L], neigs=1), 0, ax=ax)
fig.tight_layout()"""),
    code("""# Pure distortional buckling (cFSM, restricted to the D space) at the second minimum
fsm.strip(lc_c, [sig_lc.minima[1, 0]], spaces="D", neigs=1).load_factors"""),
    md("""## General end conditions

With end conditions other than S-S, lengths are physical member lengths and each length needs
several longitudinal terms (`m_all`). The terms must reach the short half-waves too: here, for
clamped-clamped (C-C) ends, terms 1 to 3 for global buckling plus the terms near `L / Lcr` for
each signature minimum (a list of term lists, one per length)."""),
    code("""L = np.array([500.0, 1000.0, 2000.0, 3000.0, 4000.0])
m_all = []
for Li in L:
    near = {m for Lcr in sig_lc.minima[:, 0] for m in range(round(Li / Lcr) - 2, round(Li / Lcr) + 3) if m > 0}
    m_all.append(sorted({1, 2, 3} | near))
cc = fsm.strip(lc_c, L, m_all=m_all, bc="C-C", neigs=3)
cc"""),
    code("""gdlo = cc.classify()[:, 0, :]   # lowest mode at each length
for Li, lf, c in zip(L, cc.curve, gdlo):
    print(f"L = {Li:6.0f} mm: Pcr/Py = {lf:.3f}, " + " ".join(f"{n} {v:.0f}%" for n, v in zip(fsm.MODE_CLASSES, c)))"""),
    code("plot_mode(cc, 3);"),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
nbformat.write(nb, out)
print("wrote", out)
