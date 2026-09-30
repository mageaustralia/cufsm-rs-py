"""Finite strip buckling of thin-walled sections (CUFSM), backed by the cufsm-rs Rust port.

Models use CUFSM's own arrays and column order:

- ``prop``        ``[mat#, Ex, Ey, vx, vy, G]``
- ``node``        ``[node#, x, z, xdof, zdof, ydof, qdof, stress]`` (dof 1 = free, 0 = fixed)
- ``elem``        ``[elem#, nodei, nodej, t, mat#]`` (1-based node and material numbers)
- ``constraints`` ``[node#e, dofe, coeff, node#k, dofk]``
- ``springs``     ``[#, nodei, nodej, ku, kv, kw, kq, local, discrete, ys]`` (nodej 0 = ground)

This is an independent port of CUFSM (Schafer et al., Johns Hopkins University), not
affiliated with or endorsed by the CUFSM authors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from . import _native
from ._native import MechanismError

__all__ = [
    "Model",
    "SectionProperties",
    "YieldActions",
    "StressActions",
    "StripResult",
    "SignatureResult",
    "MechanismError",
    "section_properties",
    "stress",
    "first_yield",
    "stress_to_action",
    "signature",
    "strip",
    "classify",
    "template",
    "lipped_c",
    "lipped_z",
    "plain_c",
]

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _rows(a: Optional[ArrayLike], ncols: Iterable[int], name: str) -> np.ndarray:
    ncols = tuple(ncols)
    if a is None:
        return np.zeros((0, ncols[-1]))
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return np.zeros((0, ncols[-1]))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] not in ncols:
        raise ValueError(f"{name} must be a 2-D array with {' or '.join(map(str, ncols))} columns, got shape {arr.shape}")
    return arr


class Model:
    """A cross-section model held as CUFSM arrays (NumPy, editable in place).

    Build it from CUFSM arrays::

        Model(prop=[[100, 29500, 29500, .3, .3, 11346.15]], node=..., elem=...)

    or from plain Python objects with :meth:`from_dicts`, or from a template
    (:func:`lipped_c`, :func:`template`).
    """

    def __init__(
        self,
        prop: ArrayLike,
        node: ArrayLike,
        elem: ArrayLike,
        constraints: Optional[ArrayLike] = None,
        springs: Optional[ArrayLike] = None,
    ):
        self.prop = _rows(prop, (6,), "prop")
        self.node = _rows(node, (8,), "node")
        self.elem = _rows(elem, (4, 5), "elem")
        self.constraints = _rows(constraints, (5,), "constraints")
        self.springs = _rows(springs, (10,), "springs")
        if len(self.prop) == 0 or len(self.node) == 0 or len(self.elem) == 0:
            raise ValueError("prop, node and elem must each have at least one row")

    # --- alternative constructors -------------------------------------------------------
    @classmethod
    def from_dicts(
        cls,
        nodes: Sequence[Mapping[str, Any]],
        elements: Sequence[Mapping[str, Any]],
        materials: Optional[Sequence[Mapping[str, Any]]] = None,
        E: Optional[float] = None,
        nu: float = 0.3,
        constraints: Sequence[Mapping[str, Any]] = (),
        springs: Sequence[Mapping[str, Any]] = (),
    ) -> "Model":
        """A model from lists of dicts, 0-based node indices.

        ``nodes``: ``{"x", "z", "stress"=1.0, "free"=(1,1,1,1)}``;
        ``elements``: ``{"i", "j", "t", "mat"=0}``;
        ``materials``: ``{"E", "nu"}`` or ``{"Ex","Ey","vx","vy","G"}``, or give ``E``/``nu``;
        ``constraints``: ``{"node_e","dof_e","coeff","node_k","dof_k"}`` (dof 1..4 or "x","z","y","q");
        ``springs``: ``{"i", "j"=None, "ku","kv","kw","kq"=0, "local"=False, "discrete"=False, "ys"=0}``.
        """
        if materials is None:
            if E is None:
                raise ValueError("give materials or E")
            materials = [{"E": E, "nu": nu}]
        prop = []
        for k, m in enumerate(materials):
            if "E" in m:
                e, v = float(m["E"]), float(m.get("nu", 0.3))
                prop.append([k + 1, e, e, v, v, e / (2 * (1 + v))])
            else:
                prop.append([k + 1, m["Ex"], m["Ey"], m["vx"], m["vy"], m["G"]])
        node = []
        for k, n in enumerate(nodes):
            f = tuple(n.get("free", (1, 1, 1, 1)))
            if len(f) != 4:
                raise ValueError(f"nodes[{k}]['free'] needs 4 flags (x, z, y, theta)")
            node.append([k + 1, n["x"], n["z"], *[1.0 if b else 0.0 for b in f], n.get("stress", 1.0)])
        elem = [[k + 1, e["i"] + 1, e["j"] + 1, e["t"], e.get("mat", 0) + 1] for k, e in enumerate(elements)]
        dofs = {"x": 1, "z": 2, "y": 3, "q": 4, "theta": 4}
        d = lambda v: dofs[v] if isinstance(v, str) else int(v)  # noqa: E731
        cons = [[c["node_e"] + 1, d(c["dof_e"]), c.get("coeff", 1.0), c["node_k"] + 1, d(c["dof_k"])] for c in constraints]
        sprs = [
            [
                k + 1, s["i"] + 1, 0 if s.get("j") is None else s["j"] + 1,
                s.get("ku", 0.0), s.get("kv", 0.0), s.get("kw", 0.0), s.get("kq", 0.0),
                float(bool(s.get("local", False))), float(bool(s.get("discrete", False))), s.get("ys", 0.0),
            ]
            for k, s in enumerate(springs)
        ]
        return cls(prop, node, elem, cons, sprs)

    # --- conveniences -------------------------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return self.node[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.node[:, 2]

    @property
    def stress(self) -> np.ndarray:
        """Nodal reference stresses (node column 8)."""
        return self.node[:, 7]

    def copy(self) -> "Model":
        return Model(self.prop.copy(), self.node.copy(), self.elem.copy(), self.constraints.copy(), self.springs.copy())

    def with_stress(self, s: ArrayLike) -> "Model":
        m = self.copy()
        s = np.asarray(s, dtype=float)
        if s.shape != (len(m.node),):
            raise ValueError(f"stress needs one value per node ({len(m.node)}), got shape {s.shape}")
        m.node[:, 7] = s
        return m

    def _arrays(self):
        return (
            self.prop.tolist(),
            self.node.tolist(),
            self.elem.tolist(),
            self.constraints.tolist(),
            self.springs.tolist(),
        )

    def __repr__(self) -> str:
        return (
            f"Model({len(self.node)} nodes, {len(self.elem)} elements, {len(self.prop)} materials, "
            f"{len(self.constraints)} constraints, {len(self.springs)} springs)"
        )

    # method forms of the module functions
    def section_properties(self) -> "SectionProperties":
        return section_properties(self)

    def first_yield(self, fy: float, restrained: bool = False, extreme_fibre: bool = True) -> "YieldActions":
        return first_yield(self, fy, restrained=restrained, extreme_fibre=extreme_fibre)

    def signature(self, lengths=None, neigs: int = 1) -> "SignatureResult":
        return signature(self, lengths, neigs=neigs)


# --- results ---------------------------------------------------------------------------


class _DictLike:
    def __getitem__(self, k):
        return getattr(self, k)

    def keys(self):
        return [f for f in self.__dataclass_fields__]  # type: ignore[attr-defined]

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}


@dataclass(frozen=True)
class SectionProperties(_DictLike):
    """grosprop + cutwp_prop2. ``thetap`` in degrees; ``wn`` is the normalised warping per node."""

    A: float
    xcg: float
    zcg: float
    Ixx: float
    Izz: float
    Ixz: float
    thetap: float
    I11: float
    I22: float
    J: float
    xs: float
    zs: float
    Cw: float
    B1: float
    B2: float
    wn: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class YieldActions(_DictLike):
    """First-yield actions. ``B`` is the first-yield bimoment (yieldB)."""

    fy: float
    Py: float
    Mxx: float
    Mzz: float
    M11: float
    M22: float
    B: float


@dataclass(frozen=True)
class StressActions(_DictLike):
    """Actions fitted to the nodal stresses; ``err`` is the residual norm."""

    P: float
    M11: float
    M22: float
    B: float
    err: float


def _stack(rows):
    """(lengths, m_terms list, load factors [nl, neigs] NaN-padded, modes [nl, neigs, ndof])."""
    lengths = np.array([r[0] for r in rows], dtype=float)
    m_terms = [np.asarray(r[1], dtype=float) for r in rows]
    ne = max((len(r[2]) for r in rows), default=0)
    lf = np.full((len(rows), ne), np.nan)
    ndof = max((len(m) for r in rows for m in r[3]), default=0)
    modes = np.full((len(rows), ne, ndof), np.nan)
    for i, r in enumerate(rows):
        lf[i, : len(r[2])] = r[2]
        for q, md in enumerate(r[3]):
            modes[i, q, : len(md)] = md
    return lengths, m_terms, lf, modes


@dataclass
class StripResult:
    """A finite strip analysis.

    ``load_factors[i, k]`` is the k-th load factor at ``lengths[i]`` (NaN where fewer were found);
    ``modes[i, k]`` its mode over every DOF in CUFSM's order, scaled so its largest entry is +1.
    """

    model: Model
    bc: str
    lengths: np.ndarray
    m_terms: List[np.ndarray]
    load_factors: np.ndarray
    modes: np.ndarray = field(repr=False)

    @property
    def curve(self) -> np.ndarray:
        """The lowest load factor at each length."""
        return self.load_factors[:, 0]

    def _rows(self):
        out = []
        for i, L in enumerate(self.lengths):
            lf = self.load_factors[i]
            n = int(np.sum(~np.isnan(lf)))
            out.append((float(L), self.m_terms[i].tolist(), lf[:n].tolist(), self.modes[i, :n].tolist()))
        return out

    def classify(self, orth: str = "axial", norm: str = "vector", ospace: str = "st") -> np.ndarray:
        return classify(self, orth=orth, norm=norm, ospace=ospace)


@dataclass
class SignatureResult(StripResult):
    """A signature curve (S-S, one half-wave). ``minima`` is an ``(n, 2)`` array of
    ``[half-wavelength, load factor]`` at each interior local minimum, shortest first."""

    minima: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


# --- functions -------------------------------------------------------------------------


def _model(m) -> Model:
    if not isinstance(m, Model):
        raise TypeError(f"expected a cufsm.Model, got {type(m).__name__}")
    return m


def section_properties(model: Model) -> SectionProperties:
    """Gross and thin-walled section properties (CUFSM grosprop and cutwp_prop2)."""
    d = _native.section_properties(_model(model)._arrays())
    d["wn"] = np.asarray(d["wn"])
    return SectionProperties(**d)


def stress(
    model: Model,
    P: float = 0.0,
    Mxx: float = 0.0,
    Mzz: float = 0.0,
    M11: float = 0.0,
    M22: float = 0.0,
    B: float = 0.0,
    restrained: bool = False,
    as_array: bool = False,
):
    """Reference stresses from actions (CUFSM stresgen, plus warp_stress for a bimoment B).

    ``restrained=True`` is CUFSM's restrained bending (``unsymm = 0``: Ixz ignored).
    Returns a new Model with the stress column set, or the stress array if ``as_array``.
    """
    s = np.asarray(_native.stress(_model(model)._arrays(), P, Mxx, Mzz, M11, M22, B, not restrained))
    return s if as_array else model.with_stress(s)


def first_yield(model: Model, fy: float, restrained: bool = False, extreme_fibre: bool = True) -> YieldActions:
    """First-yield actions: ``yieldMP_extfiber`` (element faces, default) or ``yieldMP``
    (centreline nodes, ``extreme_fibre=False``), and the bimoment ``yieldB``."""
    if not (math.isfinite(fy) and fy > 0):
        raise ValueError(f"fy = {fy} must be positive")
    return YieldActions(**_native.first_yield(_model(model)._arrays(), float(fy), not restrained, extreme_fibre))


def stress_to_action(model: Model) -> StressActions:
    """The P, M11, M22, B whose stresses best fit the model's nodal stresses."""
    return StressActions(**_native.stress_to_action(_model(model)._arrays()))


def signature(model: Model, lengths: Optional[ArrayLike] = None, neigs: int = 1) -> SignatureResult:
    """Signature curve: S-S, one half-wave per length. With ``lengths=None``, CUFSM's
    ``signature_ss`` 100 log-spaced lengths. Local minima are refined by a parabola in log L."""
    ls = None if lengths is None else np.asarray(lengths, dtype=float).ravel().tolist()
    rows, minima = _native.signature(_model(model)._arrays(), ls, int(neigs))
    L, mt, lf, modes = _stack(rows)
    return SignatureResult(model, "S-S", L, mt, lf, modes, minima=np.asarray(minima, dtype=float).reshape(-1, 2))


def strip(
    model: Model,
    lengths: ArrayLike,
    m_all: Union[None, int, Sequence[Any]] = None,
    bc: str = "S-S",
    neigs: int = 20,
    spaces: Optional[str] = None,
) -> StripResult:
    """CUFSM stripmain: load factors and modes at each length.

    ``m_all``: longitudinal terms per length - a list of lists, a single list used for every
    length, an int ``n`` for ``1..n``, or None for ``[1]``. ``bc``: 'S-S', 'C-C', 'S-C',
    'C-F', 'C-G'. ``spaces``: restrict to cFSM spaces, e.g. ``"D"`` or ``"GD"`` (None = all).
    """
    ls = np.asarray(lengths, dtype=float).ravel().tolist()
    if m_all is None:
        ma = [[1.0]] * len(ls)
    elif isinstance(m_all, (int, np.integer)):
        ma = [list(map(float, range(1, int(m_all) + 1)))] * len(ls)
    else:
        seq = list(m_all)
        if seq and all(np.isscalar(v) for v in seq):
            ma = [[float(v) for v in seq]] * len(ls)
        else:
            ma = [np.asarray(v, dtype=float).ravel().tolist() for v in seq]
    if len(ma) != len(ls):
        raise ValueError(f"{len(ls)} lengths but {len(ma)} sets of longitudinal terms")
    rows = _native.strip(_model(model)._arrays(), ls, ma, bc, int(neigs), spaces)
    L, mt, lf, modes = _stack(rows)
    return StripResult(model, bc, L, mt, lf, modes)


def classify(result: StripResult, orth: str = "axial", norm: str = "vector", ospace: str = "st") -> np.ndarray:
    """cFSM modal classification (CUFSM classify.m, uncoupled basis).

    Returns ``[nlengths, neigs, 4]`` percentages ``[G, D, L, O]`` (NaN-padded).
    ``orth``: natural | axial | load; ``norm``: none | vector | strain_energy | work;
    ``ospace``: st | k | kg | vector.
    """
    out = _native.classify(result.model._arrays(), result._rows(), result.bc, orth, norm, ospace)
    arr = np.full(result.load_factors.shape + (4,), np.nan)
    for i, per in enumerate(out):
        if per:
            arr[i, : len(per)] = per
    return arr


def _prop(E: float, nu: float) -> List[List[float]]:
    return [[1, E, E, nu, nu, E / (2 * (1 + nu))]]


def template(
    shape: str = "C",
    h: float = 9.0,
    b1: float = 5.0,
    b2: Optional[float] = None,
    d1: float = 1.0,
    d2: Optional[float] = None,
    r1: float = 0.0,
    r2: Optional[float] = None,
    r3: Optional[float] = None,
    r4: Optional[float] = None,
    q1: float = 90.0,
    q2: Optional[float] = None,
    t: float = 0.1,
    nh: int = 4,
    nb1: int = 2,
    nb2: Optional[int] = None,
    nd1: Optional[int] = None,
    nd2: Optional[int] = None,
    nr1: Optional[int] = None,
    nr2: Optional[int] = None,
    nr3: Optional[int] = None,
    nr4: Optional[int] = None,
    centerline: bool = True,
    E: float = 29500.0,
    nu: float = 0.3,
) -> Model:
    """CUFSM's C/Z template (templatecalc). Unset second-side values mirror the first; corner and
    lip strip counts default to 2 where the radius/lip is non-zero. Reference stress is 1.0."""
    b2 = b1 if b2 is None else b2
    d2 = d1 if d2 is None else d2
    r2 = r1 if r2 is None else r2
    r3 = r1 if r3 is None else r3
    r4 = r1 if r4 is None else r4
    q2 = q1 if q2 is None else q2
    nb2 = nb1 if nb2 is None else nb2
    dflt = lambda n, v: (2 if v > 0 else 0) if n is None else n  # noqa: E731
    node, elem = _native.template(
        shape, h, b1, b2, d1, d2, r1, r2, r3, r4, q1, q2, t, nh, nb1, nb2,
        dflt(nd1, d1), dflt(nd2, d2), dflt(nr1, r1), dflt(nr2, r2), dflt(nr3, r3), dflt(nr4, r4),
        centerline,
    )
    return Model(_prop(E, nu), node, elem)


def _outside(shape, depth, flange, lip, t, ri, mesh, E, nu):
    nr = 2 if ri > 0 else 0
    nd = 2 if lip > 0 else 0
    return template(
        shape, depth, flange, flange, lip, lip, ri, ri, ri, ri, 90.0, 90.0, t,
        max(mesh, 2), max(mesh // 2, 2), max(mesh // 2, 2), nd, nd, nr, nr, nr, nr,
        centerline=False, E=E, nu=nu,
    )


def lipped_c(depth: float, flange: float, lip: float, t: float, ri: float = 0.0, mesh: int = 12,
             E: float = 203000.0, nu: float = 0.3) -> Model:
    """A lipped channel from OUTSIDE dimensions and inside radius (cufsm-rs ``Template::outside``)."""
    return _outside("C", depth, flange, lip, t, ri, mesh, E, nu)


def lipped_z(depth: float, flange: float, lip: float, t: float, ri: float = 0.0, mesh: int = 12,
             E: float = 203000.0, nu: float = 0.3) -> Model:
    """A lipped Z from outside dimensions and inside radius."""
    return _outside("Z", depth, flange, lip, t, ri, mesh, E, nu)


def plain_c(depth: float, flange: float, t: float, ri: float = 0.0, mesh: int = 12,
            E: float = 203000.0, nu: float = 0.3) -> Model:
    """A plain (unlipped) channel from outside dimensions and inside radius."""
    return _outside("C", depth, flange, 0.0, t, ri, mesh, E, nu)
