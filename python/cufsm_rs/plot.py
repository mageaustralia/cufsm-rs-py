"""Matplotlib plots of models and results: ``pip install cufsm-rs-py[plot]``.

Every function draws on ``ax`` (a new figure's axes if None) and returns the Axes. Units are
the model's own; axis labels do not name a unit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
except ImportError as e:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "cufsm_rs.plot needs matplotlib; install it with: pip install cufsm-rs-py[plot]"
    ) from e

from . import MODE_CLASSES, Model, StripResult

__all__ = ["plot_section", "plot_signature", "plot_mode"]

COMPRESSION = "#d9480f"
TENSION = "#1c7ed6"
SECTION = "#868e96"
DEFORMED = "#0e7c6b"


def _axes(ax, size=(6.5, 5.5)):
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def _index(model: Model):
    """Element end node rows (0-based) from CUFSM node numbers."""
    row = {int(n): k for k, n in enumerate(model.node[:, 0])}
    return [(row[int(e[1])], row[int(e[2])]) for e in model.elem]


def _size(model: Model) -> float:
    span = max(np.ptp(model.x), np.ptp(model.z))
    return float(span) if span > 0 else 1.0


def _strips(model: Model):
    """Each element as a quadrilateral of its thickness about the centreline."""
    x, z = model.x, model.z
    polys = []
    for (i, j), t in zip(_index(model), model.elem[:, 3]):
        d = np.array([x[j] - x[i], z[j] - z[i]])
        b = np.hypot(*d)
        if b == 0:
            continue
        n = np.array([-d[1], d[0]]) / b * t / 2
        pi, pj = np.array([x[i], z[i]]), np.array([x[j], z[j]])
        polys.append([pi + n, pj + n, pj - n, pi - n])
    return polys


def _finish(ax, model: Model, pad: float):
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.margins(pad)


def plot_section(model: Model, ax=None, node_numbers: bool = False, stress: bool = True):
    """The cross-section: elements drawn to their thickness, nodes, and (``stress=True``) the
    reference stress as a coloured band normal to each element (on its outer side), compression in orange and
    tension in blue, scaled so the largest stress spans 12% of the section size.
    ``node_numbers=True`` labels the nodes with their CUFSM node numbers. Returns the Axes.
    """
    ax = _axes(ax, (7.5, 5.5) if stress else (6.5, 5.5))
    x, z = model.x, model.z
    size = _size(model)
    s = model.stress
    smax = float(np.max(np.abs(s))) if len(s) else 0.0
    handles = []
    if stress and smax > 0:
        k = 0.12 * size / smax
        centre = np.array([(x.min() + x.max()) / 2, (z.min() + z.max()) / 2])
        comp, tens = [], []
        for i, j in _index(model):
            d = np.array([x[j] - x[i], z[j] - z[i]])
            b = np.hypot(*d)
            if b == 0:
                continue
            n = np.array([-d[1], d[0]]) / b
            pi, pj = np.array([x[i], z[i]]), np.array([x[j], z[j]])
            if n @ ((pi + pj) / 2 - centre) < 0:
                n = -n  # draw the band on the side away from the middle of the section
            si, sj = s[i], s[j]
            # split at a sign change so each piece is one colour
            pieces = [(0.0, 1.0)]
            if si * sj < 0:
                r = si / (si - sj)
                pieces = [(0.0, r), (r, 1.0)]
            for a0, a1 in pieces:
                p0, p1 = pi + a0 * (pj - pi), pi + a1 * (pj - pi)
                s0, s1 = si + a0 * (sj - si), si + a1 * (sj - si)
                poly = [p0, p1, p1 + n * k * abs(s1), p0 + n * k * abs(s0)]
                (comp if (s0 + s1) > 0 else tens).append(poly)
        ax.add_collection(PolyCollection(comp, facecolors=COMPRESSION, edgecolors=COMPRESSION, alpha=0.35, lw=0.5))
        ax.add_collection(PolyCollection(tens, facecolors=TENSION, edgecolors=TENSION, alpha=0.35, lw=0.5))
        if comp:
            handles.append(Patch(color=COMPRESSION, alpha=0.5, label="compression"))
        if tens:
            handles.append(Patch(color=TENSION, alpha=0.5, label="tension"))
    ax.add_collection(PolyCollection(_strips(model), facecolors=SECTION, edgecolors="#495057", lw=0.6, zorder=2))
    ax.plot(x, z, "o", ms=3, color="#212529", zorder=3)
    if node_numbers:
        off = 0.015 * size
        for num, xi, zi in zip(model.node[:, 0], x, z):
            ax.annotate(str(int(num)), (xi, zi), xytext=(xi + off, zi + off), fontsize=8, color="#212529", zorder=4)
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
                  title=f"reference stress (max {smax:.4g})", title_fontsize=8)
    ax.autoscale_view()
    _finish(ax, model, 0.08)
    return ax


def _class_label(c) -> str:
    """The dominant cFSM class, with the runner-up when the mode is mixed: "L 98%", "D 54% L 45%"."""
    order = np.argsort(c)[::-1]
    out = f"{MODE_CLASSES[order[0]]} {c[order[0]]:.0f}%"
    if c[order[1]] >= 25:
        out += f" {MODE_CLASSES[order[1]]} {c[order[1]]:.0f}%"
    return out


def plot_signature(result: StripResult, ax=None, classify: bool = False):
    """Load factor against length on a log x axis, with the minima marked and labelled.

    ``classify=True`` also labels each minimum with its dominant cFSM class (G, D, L or O, from
    :meth:`StripResult.classify_minima`), and the runner-up too when it is 25% or more.
    Returns the Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    L, lf = result.lengths, result.curve
    ax.semilogx(L, lf, color=DEFORMED, lw=2)
    mins = result.minima
    cls = result.classify_minima() if (classify and len(mins)) else None
    for k, (Lm, lm) in enumerate(mins):
        ax.plot(Lm, lm, "o", color=COMPRESSION, zorder=3)
        label = f"{lm:.4g} at {Lm:.4g}"
        if cls is not None:
            label = f"{_class_label(cls[k])}: " + label
        ax.annotate(label, (Lm, lm), textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    finite = lf[np.isfinite(lf)]
    if len(finite):
        top = float(np.max(finite))
        if len(mins):
            top = min(top, 2.5 * float(np.max(mins[:, 1])))
        else:
            top = min(top, 4.0 * float(np.min(finite)))
        ax.set_ylim(0, top * 1.05)
    one_term = all(len(t) == 1 for t in result.m_terms)
    ax.set_xlabel("half-wavelength" if result.bc == "S-S" and one_term else "length")
    ax.set_ylabel("load factor")
    ax.grid(True, which="both", alpha=0.3)
    return ax


def _deformed_strip(pi, pj, di, dj, thi, thj, n=16):
    """Points along one strip, deformed: linear along it, cubic (Hermite, with the end
    rotations) across it, as CUFSM draws a strip."""
    d = pj - pi
    b = np.hypot(*d)
    t = d / b
    nrm = np.array([-t[1], t[0]])
    xi = np.linspace(0, 1, n)[:, None]
    ai, aj = di @ t, dj @ t
    wi, wj = di @ nrm, dj @ nrm
    h1 = 1 - 3 * xi**2 + 2 * xi**3
    h2 = b * xi * (1 - xi) ** 2
    h3 = 3 * xi**2 - 2 * xi**3
    h4 = b * xi**2 * (xi - 1)
    along = (1 - xi) * ai + xi * aj
    across = h1 * wi + h2 * thi + h3 * wj + h4 * thj
    return pi + xi * d + along * t + across * nrm


def plot_mode(result: StripResult, i_length: int, k_mode: int = 0, ax=None, scale: Optional[float] = None,
              y: Optional[float] = None):
    """A mode's deformed cross-section (solid) over the undeformed one (grey).

    The section is cut at ``y`` along the member (default: where the in-plane displacement is
    largest, mid-length for one S-S term). ``scale`` multiplies the mode's displacements; by
    default the largest in-plane displacement is drawn at 10% of the section size, whatever
    the units. Returns the Axes.
    """
    ax = _axes(ax)
    model = result.model
    ms = result.mode_shape(i_length, k_mode)
    disp = ms.at(y)
    peak = float(np.max(np.hypot(disp.u, disp.w)))
    if scale is None:
        scale = 0.1 * _size(model) / peak if peak > 0 else 1.0
    x, z = model.x, model.z
    for i, j in _index(model):
        ax.plot([x[i], x[j]], [z[i], z[j]], color=SECTION, lw=1.2, ls="--", zorder=1)
        pi, pj = np.array([x[i], z[i]]), np.array([x[j], z[j]])
        if np.hypot(*(pj - pi)) == 0:
            continue
        di = scale * np.array([disp.u[i], disp.w[i]])
        dj = scale * np.array([disp.u[j], disp.w[j]])
        pts = _deformed_strip(pi, pj, di, dj, scale * disp.theta[i], scale * disp.theta[j])
        ax.plot(pts[:, 0], pts[:, 1], color=DEFORMED, lw=2, zorder=2)
    ax.plot(x + scale * disp.u, z + scale * disp.w, "o", ms=3, color=DEFORMED, zorder=3)
    ax.set_title(f"mode {k_mode + 1}: load factor {ms.load_factor:.4g} at length {ms.length:.4g}", fontsize=10)
    ax.autoscale_view()
    _finish(ax, model, 0.1)
    return ax
