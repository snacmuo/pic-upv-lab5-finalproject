from collections import OrderedDict

import numpy as np
import shapely as shp
from skfem.io.meshio import from_meshio
from femwell.mesh import mesh_from_OrderedDict
from femwell.visualization import plot_domains
import femwell.maxwell.waveguide as fmwg

import gplugins.tidy3d.materials as mat

from .modes import guided_modes


def waveguide(
    wg_width=1.0,
    wg_thickness=0.3,
    x_c=0.0,
    wvl=1.55,
    mat_core=mat.sin(1.55),
    mat_shallow=mat.sio2(1.55),
    mat_cladd=mat.sio2(1.55),
    num_modes=2,
    sh_width=10.0,
    sh_thickness=0.150,
    XY=(-1, -1, 1, 1),
    ENABLE_PLOTS=False,
):
    """Build and solve a single rectangular waveguide cross-section."""
    x0, y0, x1, y1 = XY
    env = shp.box(x0, y0, x1, y1)
    core = shp.box(
        -wg_width / 2 + x_c,
        -wg_thickness / 2.0,
        +wg_width / 2 + x_c,
        +wg_thickness / 2.0,
    )

    polygons = OrderedDict(
        core=core,
        oxide=shp.clip_by_rect(env, -np.inf, -np.inf, np.inf, np.inf),
    )

    resol_core = wg_thickness / 5
    resol_oxide = 0.5
    resolutions = dict(
        core={"resolution": resol_core, "distance": 0.5},
        oxide={"resolution": resol_oxide, "distance": 2.0},
    )

    mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=10))

    if ENABLE_PLOTS:
        mesh.draw().show()
        plot_domains(mesh)

    basis0 = fmwg.Basis(mesh, fmwg.ElementTriP0())
    epsilon = basis0.zeros()

    for subdomain, n in {"core": mat_core, "oxide": mat_cladd}.items():
        epsilon[basis0.get_dofs(elements=subdomain)] = n**2

    if ENABLE_PLOTS:
        basis0.plot(epsilon, colorbar=True).show()

    modes = fmwg.compute_modes(basis0, epsilon, wavelength=wvl, num_modes=num_modes, order=2)
    guided_m = guided_modes(modes, mat_cladd=mat_cladd)
    return guided_m, basis0


def waveguide_array(
    wg_width=1.0,
    wg_thickness=0.3,
    wg_gap=0.8,
    wg_N=2,
    x_c=0.0,
    wvl=1.55,
    mat_core=mat.sin(1.55),
    mat_shallow=mat.sio2(1.55),
    mat_cladd=mat.sio2(1.55),
    num_modes=2,
    sh_width=10.0,
    sh_thickness=0.150,
    XY=(-1, -1, 1, 1),
    ENABLE_PLOTS=False,
):
    """Build and solve a coupled waveguide-array cross-section."""
    wg_d = wg_width + wg_gap
    if wg_N % 2 == 0:
        centers = np.arange(-wg_N // 2, wg_N // 2) + 1 / 2
    else:
        centers = np.arange(-wg_N // 2 + 1, wg_N // 2 + 1)
    centers = centers * wg_d

    padding = 2 * wg_width
    x0 = np.min(centers) - (wg_width / 2.0 + padding)
    y0 = XY[1]
    x1 = np.max(centers) + (wg_width / 2.0 + padding)
    y1 = XY[3]
    XY = [x0, y0, x1, y1]

    env = shp.box(x0, y0, x1, y1)
    cores = []
    for i in range(wg_N):
        cores.append(
            shp.box(
                -wg_width / 2 + x_c + centers[i],
                -wg_thickness / 2.0,
                +wg_width / 2 + x_c + centers[i],
                +wg_thickness / 2.0,
            )
        )

    core = shp.ops.unary_union(cores)
    polygons = OrderedDict(
        core=core,
        oxide=shp.clip_by_rect(env, -np.inf, -np.inf, np.inf, np.inf),
    )

    resol_core = wg_thickness / 5
    resol_oxide = 0.5
    resolutions = dict(
        core={"resolution": resol_core, "distance": 0.5},
        oxide={"resolution": resol_oxide, "distance": 2.0},
    )

    mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=10))

    if ENABLE_PLOTS:
        mesh.draw().show()
        plot_domains(mesh)

    basis0 = fmwg.Basis(mesh, fmwg.ElementTriP0())
    epsilon = basis0.zeros()
    for subdomain, n in {"core": mat_core, "oxide": mat_cladd}.items():
        epsilon[basis0.get_dofs(elements=subdomain)] = n**2

    if ENABLE_PLOTS:
        basis0.plot(epsilon, colorbar=True).show()

    modes = fmwg.compute_modes(basis0, epsilon, wavelength=wvl, num_modes=num_modes, order=2)
    guided_m = guided_modes(modes, mat_cladd=mat_cladd)
    return guided_m, basis0, XY, centers


# Backward-compatible alias
waveguide_Array = waveguide_array
