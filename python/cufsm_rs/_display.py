"""Dependency-free HTML for Jupyter (``_repr_html_``). Small tables, truncated for big models."""

from __future__ import annotations

import html
from typing import Any, Iterable, List, Sequence

import numpy as np

_STYLE = (
    "<style>.cufsm-rs table{border-collapse:collapse;margin:2px 0 8px 0;font-size:12px}"
    ".cufsm-rs th,.cufsm-rs td{padding:1px 8px;text-align:right;border-bottom:1px solid #ddd}"
    ".cufsm-rs th{font-weight:600}.cufsm-rs td.l,.cufsm-rs th.l{text-align:left}"
    ".cufsm-rs .note{color:#666;font-size:11px}</style>"
)

#: rows shown at each end of a truncated table
_EDGE = 5


def _num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if np.isnan(f):
        return "NaN"
    if f == int(f) and abs(f) < 1e7:
        return str(int(f))
    return f"{f:.6g}"


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], left: int = 0) -> str:
    rows = list(rows)
    out: List[str] = ["<table><tr>"]
    out += [f"<th{' class=l' if k < left else ''}>{html.escape(h)}</th>" for k, h in enumerate(header)]
    out.append("</tr>")
    for r in rows:
        if r is None:
            out.append(f"<tr><td colspan={len(header)} class=l>...</td></tr>")
            continue
        out.append("<tr>")
        out += [f"<td{' class=l' if k < left else ''}>{_num(v)}</td>" for k, v in enumerate(r)]
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _truncate(a: np.ndarray) -> list:
    rows = a.tolist()
    if len(rows) <= 2 * _EDGE + 2:
        return rows
    return rows[:_EDGE] + [None] + rows[-_EDGE:]


def _wrap(title: str, body: str) -> str:
    return f"{_STYLE}<div class='cufsm-rs'><b>{html.escape(title)}</b>{body}</div>"


def model_html(m) -> str:
    n = len(m.node)
    head = (
        f"<div>{n} nodes, {len(m.elem)} elements, {len(m.prop)} materials, "
        f"{len(m.constraints)} constraints, {len(m.springs)} springs</div>"
    )
    nodes = _table(["node#", "x", "z", "xdof", "zdof", "ydof", "qdof", "stress"], _truncate(m.node))
    elems = _table(["elem#", "nodei", "nodej", "t", "mat#"], _truncate(m.elem))
    mats = _table(["mat#", "Ex", "Ey", "vx", "vy", "G"], _truncate(m.prop))
    note = "<div class=note>Units are the model's own consistent set; positive stress is compression.</div>"
    return _wrap("cufsm_rs.Model", head + "<div>node</div>" + nodes + "<div>elem</div>" + elems
                 + "<div>prop</div>" + mats + note)


_PROP_DESC = {
    "A": "area",
    "xcg": "centroid x",
    "zcg": "centroid z",
    "Ixx": "second moment about x",
    "Izz": "second moment about z",
    "Ixz": "product moment",
    "thetap": "principal axis angle (deg)",
    "I11": "major principal second moment",
    "I22": "minor principal second moment",
    "J": "St Venant torsion constant",
    "xs": "shear centre x",
    "zs": "shear centre z",
    "Cw": "warping constant",
    "B1": "monosymmetry parameter, axis 1",
    "B2": "monosymmetry parameter, axis 2",
}


def props_html(p) -> str:
    rows = [(k, getattr(p, k), d) for k, d in _PROP_DESC.items()]
    return _wrap("Section properties", _table(["", "value", ""], rows, left=1)
                 + "<div class=note>In the model's units; wn (unit warping per node) is on the object.</div>")


def actions_html(a, title: str) -> str:
    rows = [(k, getattr(a, k)) for k in a.keys()]
    return _wrap(title, _table(["", "value"], rows, left=1)
                 + "<div class=note>In the model's units.</div>")


def result_html(r) -> str:
    L = r.lengths
    rng = f"{_num(L.min())} to {_num(L.max())}" if len(L) else "none"
    kind = "Signature curve" if r.kind == "signature" else "Strip analysis"
    nterms = sorted({len(t) for t in r.m_terms})
    head = (
        f"<div>bc {html.escape(r.bc)}, {len(L)} lengths from {rng}, "
        f"{r.neigs} load factors per length, terms per length: "
        f"{', '.join(map(str, nterms))}</div>"
    )
    body = head
    if r.kind == "signature":
        if len(r.minima):
            body += _table(["minimum", "length", "load factor"],
                           [(k + 1, Lm, lf) for k, (Lm, lf) in enumerate(r.minima)])
        else:
            body += "<div>no interior minima</div>"
    else:
        body += _table(["length", "lowest load factor"], _truncate(np.column_stack([L, r.curve])))
    body += "<div class=note>Load factors multiply the model's reference stresses.</div>"
    return _wrap(kind, body)
