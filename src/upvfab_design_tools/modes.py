import numpy as np
from tqdm.auto import tqdm
from skfem import ElementDG, ElementTriP1, ElementVector

import gplugins.tidy3d.materials as mat


def get_TETM(modes):
    """Split modes by dominant TE/TM fraction."""
    modes_TE = []
    modes_TM = []
    for mode in modes:
        if mode.te_fraction > 0.5:
            modes_TE.append(mode)
        if mode.tm_fraction > 0.5:
            modes_TM.append(mode)
    return modes_TE, modes_TM


def guided_modes(modes, mat_cladd=mat.sio2(1.55), TOL=1e-2):
    """Return modes with effective index above cladding index by at least TOL."""
    guided = []
    for mode in modes:
        if (np.real(mode.n_eff) - mat_cladd) >= TOL:
            guided.append(mode)
    return guided


def slice_mode(mode, num_points=1024, x0=-4, x1=4, polarization="TE"):
    """Sample a 1D field cut at y=0 for a given polarization."""
    (et, et_basis), _ = mode.basis.split(mode.E)
    plot_basis = et_basis.with_element(ElementVector(ElementDG(ElementTriP1())))
    et_xy = plot_basis.project(et_basis.interpolate(et))
    (et_x, et_x_basis), (et_y, et_y_basis) = plot_basis.split(et_xy)

    query_pts = np.vstack(
        [
            np.linspace(x0, x1, num_points),
            np.zeros(num_points),
        ]
    )

    if polarization == "TE":
        p0_probes = et_x_basis.probes(query_pts)
        e = p0_probes @ et_x
    else:
        p0_probes = et_y_basis.probes(query_pts)
        e = p0_probes @ et_y

    return query_pts[0], e


def mode_overlap(mode_single, modes_multiple):
    """Calculate overlap between one mode and a mode list."""
    num_modes = np.size(modes_multiple)
    ovl = np.zeros((1, num_modes), dtype=np.complex128)
    for j, mode in enumerate(modes_multiple):
        ovl[0][j] = mode_single.calculate_overlap(mode)
    return ovl


def propagate_modes(
    wvl=1.55,
    dz=0.05,
    L=100.0,
    ovl_z_0=None,
    modes=None,
    slices=None,
):
    """Propagate modal superposition along z."""
    if ovl_z_0 is None:
        ovl_z_0 = []
    if modes is None:
        modes = []
    if slices is None:
        slices = []

    num_modes = np.size(modes)
    np_z = int(np.round(L / dz))
    L_v = np.linspace(0, L, np_z)

    beta_v = np.zeros((num_modes))
    for i, mode in enumerate(tqdm(modes)):
        beta_v[i] = 2 * np.pi / wvl * np.real(mode.n_eff)

    num_points = np.size(slices, 1)
    intensity_L = np.zeros((np_z, num_points))
    propag = np.zeros((num_modes), dtype=np.complex128)

    for i, L_i in enumerate(tqdm(L_v)):
        ph = np.exp(-1.0j * beta_v * L_i)
        propag = np.multiply(ovl_z_0, ph.T)
        e_z = np.zeros(num_points, dtype=np.complex128)
        for j, _ in enumerate(modes):
            e_z += slices[j] * propag[j]
        intensity_L[i] = np.abs(e_z) ** 2

    return L_v, intensity_L, propag
