"""A lipped channel from end to end: section properties, first yield, the signature curve, its
minima and what kind of buckling each one is. Run:  python examples/demo.py

Units: N, mm and MPa throughout (the package itself is unit-agnostic)."""
import time

import numpy as np

import cufsm_rs as fsm

# 200 x 76 x 15 x 1.9 mm lipped channel, 3 mm inside corner radius, steel
sec = fsm.lipped_c(200, 76, 15, 1.9, ri=3)
print(sec)

p = fsm.section_properties(sec)
print(f"A = {p.A:.1f} mm2   Ixx = {p.Ixx / 1e6:.3f}e6 mm4   Izz = {p.Izz / 1e6:.3f}e6 mm4")
print(f"J = {p.J:.1f} mm4   Cw = {p.Cw / 1e9:.3f}e9 mm6   shear centre xs = {p.xs:.2f} mm")

fy = 450.0
y = fsm.first_yield(sec, fy)  # extreme fibre, as current CUFSM
print(f"first yield at fy = {fy:g} MPa: Py = {y.Py / 1e3:.1f} kN, "
      f"Mxx = {y.Mxx / 1e6:.2f} kNm, Mzz = {y.Mzz / 1e6:.2f} kNm")

# reference load = the squash load, so each load factor reads directly as Pcr / Py
loaded = fsm.stress(sec, P=y.Py)
lengths = np.logspace(1, 4, 90)  # half-wavelengths 10 mm to 10 m
t0 = time.perf_counter()
sig = fsm.signature(loaded, lengths)
ms = (time.perf_counter() - t0) * 1e3
print(f"{sig}: {ms:.0f} ms")

# classify the lowest mode at each minimum (cFSM: global, distortional, local, other)
for (L, lf), c in zip(sig.minima, sig.classify_minima()):
    kind = fsm.MODE_CLASSES[int(np.argmax(c))]
    print(f"  minimum at L = {L:7.1f} mm: Pcr/Py = {lf:.3f}  (mostly {kind}; "
          + " ".join(f"{n} {v:.0f}%" for n, v in zip(fsm.MODE_CLASSES, c)) + ")")

# the mode at the first minimum, per node: u, v, w, theta have shape (terms, nodes)
first = fsm.strip(loaded, [sig.minima[0, 0]], neigs=1).mode_shape(0)
print(f"mode at the first minimum: u {first.u.shape}, largest in-plane move at node "
      f"{int(np.argmax(np.hypot(first.u[0], first.w[0]))) + 1}")

try:
    import matplotlib

    matplotlib.use("Agg")
    from cufsm_rs.plot import plot_signature

    ax = plot_signature(sig, classify=True)
    ax.set_ylabel("Pcr / Py")
    ax.set_title("200x76x15x1.9 lipped C: signature curve (half-wavelength in mm)")
    ax.figure.tight_layout()
    ax.figure.savefig("examples/signature.png", dpi=120)
    print("plot: examples/signature.png")
except ImportError:
    print("(pip install cufsm-rs-py[plot] to also get the plot)")
