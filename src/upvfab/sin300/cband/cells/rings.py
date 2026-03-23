"""Ring Resonators."""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from upvfab.sin300.cband.tech import TECH


@gf.cell
def ring_single(
    gap: float = TECH.gap_strip,
    radius: float = TECH.radius_strip,
    length_x: float = 4.0, # TBD: Ring resonator parameters 
    length_y: float = TECH.gap_strip,
    cross_section: CrossSectionSpec = "strip",
    length_extension: float = 100.0,
) -> gf.Component:
    """Returns a single ring.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        cross_section: cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.


    .. code::

                    xxxxxxxxxxxxx
                xxxxx           xxxx
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2
    """
    return gf.c.ring_single(
        gap=gap,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend="bend_euler",
        straight="straight",
        coupler_ring="coupler_ring",
        cross_section=cross_section,
        length_extension=length_extension,
    )


@gf.cell
def ring_double(
    gap: float = TECH.gap_strip,
    gap_top: float | None = None,
    gap_bot: float | None = None,
    radius: float = TECH.radius_strip,
    length_x: float = 0.01,
    length_y: float = 0.01,
    cross_section: CrossSectionSpec = "strip",
    length_extension: float = 100.0,
) -> gf.Component:
    """Returns a double bus ring.

    two couplers (ct: top, cb: bottom)
    connected with two vertical straights (sl: left, sr: right)

    Args:
        gap: gap between for coupler.
        gap_top: gap for the top coupler. Defaults to gap.
        gap_bot: gap for the bottom coupler. Defaults to gap.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        bend: 90 degrees bend spec.
        straight: straight spec.
        coupler_ring: ring coupler spec.
        coupler_ring_top: top ring coupler spec. Defaults to coupler_ring.
        cross_section: cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.
    """
    return gf.c.ring_double(
        gap=gap,
        gap_top=gap_top,
        gap_bot=gap_bot,
        radius=radius,
        length_x=length_x,
        length_y=length_y,
        bend="bend_euler",
        straight="straight",
        coupler_ring="coupler_ring",
        cross_section=cross_section,
        length_extension=length_extension,
    )


if __name__ == "__main__":
    c = ring_single()
    c.show()
