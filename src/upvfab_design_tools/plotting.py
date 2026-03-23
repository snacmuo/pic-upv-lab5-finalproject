import matplotlib.pyplot as plt
import numpy as np
from skfem import ElementDG, ElementTriP1, ElementVector


def my_plot_mode(mode, i, xlim=(-2, 2), ylim=(-1, 1)):
    """Plot Ex/Ey transverse fields for a FEMWELL mode."""
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")

    (et, et_basis), _ = mode.basis.split(mode.E)
    plot_basis = et_basis.with_element(ElementVector(ElementDG(ElementTriP1())))
    et_xy = plot_basis.project(et_basis.interpolate(et))
    (et_x, et_x_basis), (et_y, et_y_basis) = plot_basis.split(et_xy)

    for ax in axs:
        mode.basis.mesh.draw(ax=ax, boundaries_only=True)
        for subdomain in mode.basis.mesh.subdomains.keys() - {"gmsh:bounding_entities"}:
            mode.basis.mesh.restrict(subdomain).draw(ax=ax, boundaries_only=True)

    et_x_basis.plot(et_x, shading="gouraud", ax=axs[0], vmin=np.min(et_x), vmax=np.max(et_x), cmap="bwr")
    et_y_basis.plot(et_y, shading="gouraud", ax=axs[1], vmin=np.min(et_y), vmax=np.max(et_y), cmap="bwr")

    ctxt = "black"
    fs = 8
    axs[0].annotate(f"Mode # {i}, n_eff = {np.real(mode.n_eff):.4f}", (-1.95, 1.5), color=ctxt, fontsize=fs)
    axs[0].annotate(f"TE {mode.te_fraction:.4f}, TM {mode.tm_fraction:.4f}", (-1.95, 1.0), color=ctxt, fontsize=fs)

    plt.xlim(xlim)
    plt.ylim(ylim)
    return fig, axs
