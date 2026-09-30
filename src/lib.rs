//! Native layer of the `cufsm_rs` Python package (`pip install cufsm-rs-py`): thin PyO3 wrappers over cufsm-rs.
//!
//! The Python side (`python/cufsm_rs/__init__.py`) owns the CUFSM-style arrays and the NumPy
//! conversion; everything here takes and returns plain lists so the boundary stays simple.
//! Models arrive as CUFSM arrays (prop, node, elem, constraints, springs) with CUFSM's own
//! 1-based node and material numbers, and are mapped to the crate's 0-based model here.

use std::collections::HashMap;

use cufsm::cfsm::{classify_with, stripmain_constrained, Norm, OSpace, Orth, Spaces};
use cufsm::template::{templatecalc, Shape, Template};
use cufsm::{
    add_bimoment_stress, cutwp_prop2, grosprop, signature_minima, signature_ss, stresgen,
    stress_to_action, stripmain, yield_b, yield_mp, yield_mp_extfiber, Actions, BoundaryCondition,
    Constraint, Dof, Element, LengthResult, Material, Model, Node, Spring,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(
    cufsm_rs,
    MechanismError,
    PyValueError,
    "The elastic stiffness is not positive definite: the section has a mechanism at this length."
);

fn to_py(e: cufsm::Error) -> PyErr {
    match e {
        cufsm::Error::InvalidModel(s) => PyValueError::new_err(format!("invalid model: {s}")),
        e @ cufsm::Error::NotPositiveDefinite { .. } => MechanismError::new_err(e.to_string()),
    }
}

fn bad(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn dof(code: f64, what: &str) -> PyResult<Dof> {
    match code as i64 {
        1 => Ok(Dof::X),
        2 => Ok(Dof::Z),
        3 => Ok(Dof::Y),
        4 => Ok(Dof::Theta),
        c => Err(bad(format!(
            "{what}: DOF code {c} is not 1 (x), 2 (z), 3 (y) or 4 (theta)"
        ))),
    }
}

fn finite(v: f64, what: &str) -> PyResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(bad(format!("{what} is not a finite number ({v})")))
    }
}

/// Builds a crate model from CUFSM arrays.
///
/// prop: [mat#, Ex, Ey, vx, vy, G]; node: [node#, x, z, xdof, zdof, ydof, qdof, stress];
/// elem: [elem#, nodei, nodej, t, mat#] (mat# optional, first material if absent);
/// constraints: [node#e, dofe, coeff, node#k, dofk]; springs (v4.3):
/// [#, nodei, nodej, ku, kv, kw, kq, local, discrete, ys], nodej 0 = ground.
fn build(
    prop: &[Vec<f64>],
    node: &[Vec<f64>],
    elem: &[Vec<f64>],
    constraints: &[Vec<f64>],
    springs: &[Vec<f64>],
) -> PyResult<Model> {
    if prop.is_empty() {
        return Err(bad(
            "prop is empty: at least one material row [mat#, Ex, Ey, vx, vy, G] is needed",
        ));
    }
    let mut mat_index = HashMap::new();
    let mut materials = Vec::with_capacity(prop.len());
    for (i, p) in prop.iter().enumerate() {
        if p.len() != 6 {
            return Err(bad(format!(
                "prop row {i} has {} columns; CUFSM's prop is [mat#, Ex, Ey, vx, vy, G]",
                p.len()
            )));
        }
        for (k, v) in p.iter().enumerate() {
            finite(*v, &format!("prop row {i} column {k}"))?;
        }
        if mat_index.insert(p[0] as i64, i).is_some() {
            return Err(bad(format!("prop: material number {} appears twice", p[0])));
        }
        materials.push(Material {
            ex: p[1],
            ey: p[2],
            vx: p[3],
            vy: p[4],
            g: p[5],
        });
    }
    let mut node_index = HashMap::new();
    let mut nodes = Vec::with_capacity(node.len());
    for (i, n) in node.iter().enumerate() {
        if n.len() != 8 {
            return Err(bad(format!(
                "node row {i} has {} columns; CUFSM's node is [node#, x, z, xdof, zdof, ydof, qdof, stress]",
                n.len()
            )));
        }
        for (k, v) in n.iter().enumerate() {
            finite(*v, &format!("node row {i} column {k}"))?;
        }
        if node_index.insert(n[0] as i64, i).is_some() {
            return Err(bad(format!("node: node number {} appears twice", n[0])));
        }
        nodes.push(Node {
            x: n[1],
            z: n[2],
            free: [n[3] != 0.0, n[4] != 0.0, n[5] != 0.0, n[6] != 0.0],
            stress: n[7],
        });
    }
    let node_of = |num: f64, what: &str| -> PyResult<usize> {
        node_index
            .get(&(num as i64))
            .copied()
            .ok_or_else(|| bad(format!("{what} refers to node {num}, which is not in node")))
    };
    let mut elements = Vec::with_capacity(elem.len());
    for (i, e) in elem.iter().enumerate() {
        if e.len() != 4 && e.len() != 5 {
            return Err(bad(format!(
                "elem row {i} has {} columns; CUFSM's elem is [elem#, nodei, nodej, t, mat#]",
                e.len()
            )));
        }
        let what = format!("elem row {i}");
        let mat = if e.len() == 5 {
            *mat_index.get(&(e[4] as i64)).ok_or_else(|| {
                bad(format!(
                    "{what} refers to material {}, which is not in prop",
                    e[4]
                ))
            })?
        } else {
            0
        };
        elements.push(Element {
            ni: node_of(e[1], &what)?,
            nj: node_of(e[2], &what)?,
            t: finite(e[3], &format!("{what} thickness"))?,
            mat,
        });
    }
    let mut cons = Vec::with_capacity(constraints.len());
    for (i, c) in constraints.iter().enumerate() {
        let what = format!("constraints row {i}");
        if c.len() != 5 {
            return Err(bad(format!(
                "{what} has {} columns; CUFSM's constraints are [node#e, dofe, coeff, node#k, dofk]",
                c.len()
            )));
        }
        cons.push(Constraint {
            node_e: node_of(c[0], &what)?,
            dof_e: dof(c[1], &what)?,
            coeff: finite(c[2], &what)?,
            node_k: node_of(c[3], &what)?,
            dof_k: dof(c[4], &what)?,
        });
    }
    let mut sprs = Vec::with_capacity(springs.len());
    for (i, s) in springs.iter().enumerate() {
        let what = format!("springs row {i}");
        if s.len() != 10 {
            return Err(bad(format!(
                "{what} has {} columns; CUFSM's (v4.3) springs are [#, nodei, nodej, ku, kv, kw, kq, local, discrete, ys]",
                s.len()
            )));
        }
        for (k, v) in s.iter().enumerate() {
            finite(*v, &format!("{what} column {k}"))?;
        }
        sprs.push(Spring {
            ni: node_of(s[1], &what)?,
            nj: if s[2] == 0.0 {
                None
            } else {
                Some(node_of(s[2], &what)?)
            },
            ku: s[3],
            kv: s[4],
            kw: s[5],
            kq: s[6],
            local: s[7] != 0.0,
            discrete: s[8] != 0.0,
            ys_fraction: s[9],
        });
    }
    let m = Model {
        materials,
        nodes,
        elements,
        constraints: cons,
        springs: sprs,
    };
    m.validate().map_err(to_py)?;
    Ok(m)
}

/// A CUFSM table: one `Vec` per row.
type Table = Vec<Vec<f64>>;

/// prop, node, elem, constraints, springs.
type Arrays = (Table, Table, Table, Table, Table);

fn model_of(a: &Arrays) -> PyResult<Model> {
    build(&a.0, &a.1, &a.2, &a.3, &a.4)
}

fn parse_bc(bc: &str) -> PyResult<BoundaryCondition> {
    BoundaryCondition::parse(&bc.to_ascii_uppercase()).ok_or_else(|| {
        bad(format!(
            "boundary condition {bc:?} is not one of S-S, C-C, S-C, C-F, C-G"
        ))
    })
}

/// Gross properties (grosprop) and thin-walled properties (cutwp_prop2).
#[pyfunction]
fn section_properties<'py>(py: Python<'py>, arrays: Arrays) -> PyResult<Bound<'py, PyDict>> {
    let m = model_of(&arrays)?;
    let g = grosprop(&m);
    let c = cutwp_prop2(&m);
    let d = PyDict::new(py);
    for (k, v) in [
        ("A", g.a),
        ("xcg", g.xcg),
        ("zcg", g.zcg),
        ("Ixx", g.ixx),
        ("Izz", g.izz),
        ("Ixz", g.ixz),
        ("thetap", g.thetap),
        ("I11", g.i11),
        ("I22", g.i22),
        ("J", c.j),
        ("xs", c.xs),
        ("zs", c.zs),
        ("Cw", c.cw),
        ("B1", c.b1),
        ("B2", c.b2),
    ] {
        d.set_item(k, v)?;
    }
    d.set_item("wn", c.wn)?;
    Ok(d)
}

/// Nodal reference stresses from member actions (stresgen, plus warp_stress for B).
#[pyfunction]
#[pyo3(signature = (arrays, p, mxx, mzz, m11, m22, b, unsymmetric))]
#[allow(clippy::too_many_arguments)]
fn stress(
    arrays: Arrays,
    p: f64,
    mxx: f64,
    mzz: f64,
    m11: f64,
    m22: f64,
    b: f64,
    unsymmetric: bool,
) -> PyResult<Vec<f64>> {
    let mut m = model_of(&arrays)?;
    let g = grosprop(&m);
    stresgen(
        &mut m,
        &Actions {
            p,
            mxx,
            mzz,
            m11,
            m22,
        },
        &g,
        unsymmetric,
    );
    if b != 0.0 {
        let c = cutwp_prop2(&m);
        add_bimoment_stress(&mut m, b, c.cw, &c.wn);
    }
    Ok(m.nodes.iter().map(|n| n.stress).collect())
}

/// First-yield actions: yieldMP (centreline) or yieldMP_extfiber (element faces), plus yieldB.
#[pyfunction]
fn first_yield<'py>(
    py: Python<'py>,
    arrays: Arrays,
    fy: f64,
    unsymmetric: bool,
    extreme_fibre: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let m = model_of(&arrays)?;
    let g = grosprop(&m);
    let y = if extreme_fibre {
        yield_mp_extfiber(&m, fy, &g, unsymmetric)
    } else {
        yield_mp(&m, fy, &g, unsymmetric)
    };
    let c = cutwp_prop2(&m);
    let d = PyDict::new(py);
    for (k, v) in [
        ("fy", fy),
        ("Py", y.py),
        ("Mxx", y.mxx),
        ("Mzz", y.mzz),
        ("M11", y.m11),
        ("M22", y.m22),
        ("B", yield_b(fy, c.cw, &c.wn)),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// Least-squares P, M11, M22, B fitted to the model's nodal stresses (stress_to_action).
#[pyfunction(name = "stress_to_action")]
fn py_stress_to_action<'py>(py: Python<'py>, arrays: Arrays) -> PyResult<Bound<'py, PyDict>> {
    let m = model_of(&arrays)?;
    let c = cutwp_prop2(&m);
    let s = stress_to_action(&m, &grosprop(&m), c.cw, &c.wn);
    let d = PyDict::new(py);
    for (k, v) in [
        ("P", s.p),
        ("M11", s.m11),
        ("M22", s.m22),
        ("B", s.b),
        ("err", s.err),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// One length: (length, m_terms, load factors, modes).
type Row = (f64, Vec<f64>, Vec<f64>, Table);

/// A signature curve's rows and its minima as (half-wavelength, load factor).
type Signature = (Vec<Row>, Vec<(f64, f64)>);

fn rows(r: Vec<LengthResult>) -> Vec<Row> {
    r.into_iter()
        .map(|l| (l.length, l.m_terms, l.load_factors, l.modes))
        .collect()
}

fn parse_spaces(s: &str) -> PyResult<Spaces> {
    let mut sp = Spaces::default();
    for ch in s.chars() {
        match ch.to_ascii_uppercase() {
            'G' => sp.global = true,
            'D' => sp.distortional = true,
            'L' => sp.local = true,
            'O' => sp.other = true,
            c => return Err(bad(format!("spaces: {c:?} is not one of G, D, L, O"))),
        }
    }
    if sp == Spaces::default() {
        return Err(bad("spaces is empty: give some of G, D, L, O"));
    }
    Ok(sp)
}

/// stripmain: every length, its longitudinal terms, load factors and modes. The GIL is released.
#[pyfunction]
#[pyo3(signature = (arrays, lengths, m_all, bc, neigs, spaces=None))]
fn strip(
    py: Python<'_>,
    arrays: Arrays,
    lengths: Vec<f64>,
    m_all: Vec<Vec<f64>>,
    bc: &str,
    neigs: usize,
    spaces: Option<&str>,
) -> PyResult<Vec<Row>> {
    let m = model_of(&arrays)?;
    let bc = parse_bc(bc)?;
    if neigs == 0 {
        return Err(bad("neigs must be at least 1"));
    }
    for (i, a) in lengths.iter().enumerate() {
        if !(a.is_finite() && *a > 0.0) {
            return Err(bad(format!("lengths[{i}] = {a} is not a positive length")));
        }
    }
    let spaces = spaces.map(parse_spaces).transpose()?;
    let res = py.detach(|| match spaces {
        None => stripmain(&m, &lengths, &m_all, bc, neigs),
        Some(sp) => stripmain_constrained(&m, &lengths, &m_all, bc, neigs, sp),
    });
    res.map(rows).map_err(to_py)
}

/// signature_ss (CUFSM's own 100 lengths) or S-S at given lengths with one term; plus minima.
#[pyfunction]
#[pyo3(signature = (arrays, lengths, neigs))]
fn signature(
    py: Python<'_>,
    arrays: Arrays,
    lengths: Option<Vec<f64>>,
    neigs: usize,
) -> PyResult<Signature> {
    let m = model_of(&arrays)?;
    if neigs == 0 {
        return Err(bad("neigs must be at least 1"));
    }
    if let Some(ls) = &lengths {
        for (i, a) in ls.iter().enumerate() {
            if !(a.is_finite() && *a > 0.0) {
                return Err(bad(format!("lengths[{i}] = {a} is not a positive length")));
            }
        }
    }
    let res = py
        .detach(|| match &lengths {
            None => signature_ss(&m, neigs),
            Some(ls) => {
                let m_all = vec![vec![1.0]; ls.len()];
                stripmain(&m, ls, &m_all, BoundaryCondition::SS, neigs)
            }
        })
        .map_err(to_py)?;
    let minima = signature_minima(&res)
        .into_iter()
        .map(|x| (x.length, x.load_factor))
        .collect();
    Ok((rows(res), minima))
}

/// cFSM classification [G, D, L, O] percent of every mode of a strip result.
#[pyfunction]
#[pyo3(signature = (arrays, results, bc, orth, norm, ospace))]
fn classify(
    py: Python<'_>,
    arrays: Arrays,
    results: Vec<Row>,
    bc: &str,
    orth: &str,
    norm: &str,
    ospace: &str,
) -> PyResult<Vec<Vec<[f64; 4]>>> {
    let m = model_of(&arrays)?;
    let bc = parse_bc(bc)?;
    let orth = match orth.to_ascii_lowercase().as_str() {
        "natural" => Orth::Natural,
        "axial" => Orth::Axial,
        "load" => Orth::Load,
        o => return Err(bad(format!("orth {o:?} is not natural, axial or load"))),
    };
    let norm = match norm.to_ascii_lowercase().as_str() {
        "none" => Norm::None,
        "vector" => Norm::Vector,
        "strain_energy" => Norm::StrainEnergy,
        "work" => Norm::Work,
        n => {
            return Err(bad(format!(
                "norm {n:?} is not none, vector, strain_energy or work"
            )))
        }
    };
    let ospace = match ospace.to_ascii_lowercase().as_str() {
        "st" => OSpace::St,
        "k" => OSpace::K,
        "kg" => OSpace::Kg,
        "vector" => OSpace::Vector,
        o => return Err(bad(format!("ospace {o:?} is not st, k, kg or vector"))),
    };
    let ndof = 4 * m.nodes.len();
    let lr: Vec<LengthResult> = results
        .into_iter()
        .map(|(length, m_terms, load_factors, modes)| {
            let want = ndof * m_terms.len();
            if let Some(bad_mode) = modes.iter().find(|md| md.len() != want) {
                return Err(bad(format!(
                    "a mode has {} entries; this model with {} terms needs {want}",
                    bad_mode.len(),
                    m_terms.len()
                )));
            }
            Ok(LengthResult {
                length,
                m_terms,
                load_factors,
                modes,
            })
        })
        .collect::<PyResult<_>>()?;
    py.detach(|| classify_with(&m, &lr, bc, orth, norm, ospace))
        .map_err(to_py)
}

/// CUFSM's C/Z template (templatecalc). Returns (node, elem) CUFSM arrays, 1-based.
#[pyfunction]
#[pyo3(signature = (shape, h, b1, b2, d1, d2, r1, r2, r3, r4, q1, q2, t, nh, nb1, nb2, nd1, nd2, nr1, nr2, nr3, nr4, centerline))]
#[allow(clippy::too_many_arguments)]
fn template(
    shape: &str,
    h: f64,
    b1: f64,
    b2: f64,
    d1: f64,
    d2: f64,
    r1: f64,
    r2: f64,
    r3: f64,
    r4: f64,
    q1: f64,
    q2: f64,
    t: f64,
    nh: usize,
    nb1: usize,
    nb2: usize,
    nd1: usize,
    nd2: usize,
    nr1: usize,
    nr2: usize,
    nr3: usize,
    nr4: usize,
    centerline: bool,
) -> PyResult<(Table, Table)> {
    let shape = match shape.to_ascii_uppercase().as_str() {
        "C" => Shape::C,
        "Z" => Shape::Z,
        s => return Err(bad(format!("shape {s:?} is not C or Z"))),
    };
    for (name, v) in [("h", h), ("b1", b1), ("b2", b2), ("t", t)] {
        if !(v.is_finite() && v > 0.0) {
            return Err(bad(format!("{name} = {v} must be positive")));
        }
    }
    for (name, v) in [
        ("d1", d1),
        ("d2", d2),
        ("r1", r1),
        ("r2", r2),
        ("r3", r3),
        ("r4", r4),
    ] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(bad(format!("{name} = {v} must be zero or positive")));
        }
    }
    if nh == 0 || nb1 == 0 || nb2 == 0 {
        return Err(bad("nh, nb1 and nb2 must be at least 1"));
    }
    let tp = Template {
        shape,
        h,
        b1,
        b2,
        d1,
        d2,
        r1,
        r2,
        r3,
        r4,
        q1,
        q2,
        t,
        nh,
        nb1,
        nb2,
        nd1,
        nd2,
        nr1,
        nr2,
        nr3,
        nr4,
        centerline,
    };
    // Material is a placeholder: the Python side supplies prop.
    let m = templatecalc(&tp, Material::isotropic(1.0, 0.3));
    let node = m
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let f = |b: bool| if b { 1.0 } else { 0.0 };
            vec![
                (i + 1) as f64,
                n.x,
                n.z,
                f(n.free[0]),
                f(n.free[1]),
                f(n.free[2]),
                f(n.free[3]),
                n.stress,
            ]
        })
        .collect();
    let elem = m
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| vec![(i + 1) as f64, (e.ni + 1) as f64, (e.nj + 1) as f64, e.t])
        .collect();
    Ok((node, elem))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MechanismError", m.py().get_type::<MechanismError>())?;
    m.add_function(wrap_pyfunction!(section_properties, m)?)?;
    m.add_function(wrap_pyfunction!(stress, m)?)?;
    m.add_function(wrap_pyfunction!(first_yield, m)?)?;
    m.add_function(wrap_pyfunction!(py_stress_to_action, m)?)?;
    m.add_function(wrap_pyfunction!(strip, m)?)?;
    m.add_function(wrap_pyfunction!(signature, m)?)?;
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    m.add_function(wrap_pyfunction!(template, m)?)?;
    Ok(())
}
