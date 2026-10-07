"""The shared route-planning engine.

Ingest, graph, elevation, search, export, API and interface, independent of
what is being planned for. A sport package — `mtb_planner`, `trail` — supplies
the parts that do differ through `planner_core.sport.Sport` and activates it.
"""

__version__ = "0.2.0"

#: Bumped whenever the on-disk graph artifact layout changes. The server
#: refuses to load an artifact whose manifest disagrees.
#:
#: v2 added the per-edge feature arrays and the tech-confidence coverage map.
#: v3 added elevation: ascent, descent, grade extremes and the grade histogram.
#: The sport is recorded in the manifest with a default, so naming it was not
#: a layout change and v3 artifacts built for bikes still load.
ARTIFACT_SCHEMA_VERSION = 3

__all__ = ["ARTIFACT_SCHEMA_VERSION", "__version__"]
