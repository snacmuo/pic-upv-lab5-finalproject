"""Si220 pdk."""

from functools import lru_cache

from gdsfactory.config import CONF
from gdsfactory.cross_section import get_cross_sections
from gdsfactory.get_factories import get_cells
from gdsfactory.pdk import Pdk

from upvfab.sin300.cband import cells, config, tech
from upvfab.sin300.cband.config import PATH
from upvfab.sin300.cband.models import get_models
from upvfab.sin300.cband.tech import LAYER, LAYER_STACK, LAYER_VIEWS, routing_strategies

_models = get_models()
_cells = get_cells(cells)
_cross_sections = get_cross_sections(tech)

CONF.pdk = "upvfab.sin300.cband"


@lru_cache
def get_pdk() -> Pdk:
    """Return Cornerstone PDK."""
    return Pdk(
        name="upvfab.sin300.cband",
        cells=_cells,
        cross_sections=_cross_sections,  # type: ignore
        layers=LAYER,
        layer_stack=LAYER_STACK,
        layer_views=LAYER_VIEWS,
        models=_models,
        routing_strategies=routing_strategies,
    )


PDK = get_pdk()

__all__ = [
    "LAYER",
    "LAYER_STACK",
    "LAYER_VIEWS",
    "PATH",
    "cells",
    "config",
    "tech",
]
