"""A lipped channel from end to end: section properties, first yield, the signature curve, its
minima and what kind of buckling each one is. Run:  python examples/demo.py"""
import time

import numpy as np

import cufsm_rs as fsm

# 200 x 76 x 15 x 1.9 mm lipped channel, 3 mm inside corner radius, steel
sec = fsm.lipped_c(200, 76, 15, 1.9, ri=3)
print(f"model: {len(sec.node)} nodes, {len(sec.elem)} elements")

p = fsm.section_properties(sec)
print(f"A = {p.A:.1f} mm2   Ixx = {p.Ixx / 1e6:.3f}e6 mm4   Izz = {p.Izz / 1e6:.3f}e6 mm4")
print(f"J = {p.J:.1f} mm4   Cw = {p.Cw / 1e9:.3f}e9 mm6   shear centre xs = {p.xs:.2f} mm")

fy = 450.0
y = fsm.first_yield(sec, fy)                      # extreme fibre, as current CUFSM
print(f"first yield at fy = {fy:g} MPa: Py = {y.Py / 1e3:.1f} kN, "
      f"Mxxy = {y.Mxx / 1e6:.2f} kNm, Mzzy = {y.Mzz / 1e6:.2f} kNm")

# reference load = the squash load, so each load factor reads directly as Pcr / Py
loaded = fsm.stress(sec, P=y.Py)
lengths = np.logspace(1, 4, 90)                     # 10 mm to 10 m
t0 = time.perf_counter()
sig = fsm.signature(loaded, lengths)
ms = (time.perf_counter() - t0) * 1e3
print(f"signature curve: {len(lengths)} lengths in {ms:.0f} ms")

# classify the mode at each minimum (cFSM: global, distortional, local, other)
at_min = fsm.strip(loaded, sig.minima[:, 0], bc="S-S")
cls = at_min.classify()[:, 0, :]                    # lowest mode at each minimum
names = "GDLO"
for (L, lf), c in zip(sig.minima, cls):
    kind = {"G": "global", "D": "distortional", "L": "local", "O": "other"}[names[int(np.argmax(c))]]
    print(f"  minimum at L = {L:7.1f} mm: Pcr/Py = {lf:.3f}  ({kind}; "
          + " ".join(f"{n} {v:.0f}%" for n, v in zip(names, c)) + ")")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(sig.lengths, sig.curve, color="#0e7c6b", lw=2)
    for L, lf in sig.minima:
        ax.plot(L, lf, "o", color="#d9480f")
        ax.annotate(f"{lf:.3f} @ {L:.0f}", (L, lf), textcoords="offset points", xytext=(6, -14))
    ax.set_xlabel("half-wavelength (mm)")
    ax.set_ylabel("Pcr / Py")
    ax.set_ylim(0, min(3.0, float(np.nanmax(sig.curve))))
    ax.set_title("200x76x15x1.9 lipped C: signature curve (cufsm-rs from Python)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/signature.png", dpi=120)
    print("plot: examples/signature.png")
except ImportError:
    print("(pip install matplotlib to also get the plot)")
