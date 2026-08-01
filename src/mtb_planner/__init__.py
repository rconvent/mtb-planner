"""Mountain bike route planner.

Time-budgeted loop and point-to-point routing over OpenStreetMap data, scored
on inferred trail complexity rather than generic cycling suitability.
"""

__version__ = "0.1.0"

#: Bumped whenever the on-disk graph artifact layout changes. The server
#: refuses to load an artifact whose manifest disagrees.
#:
#: v2 added the per-edge feature arrays and the tech-confidence coverage map.
#: v3 added elevation: ascent, descent, grade extremes and the grade histogram.
ARTIFACT_SCHEMA_VERSION = 3

#: Bumped whenever edge features or the cost/time model change in a way that
#: alters routing results. Guards against a new server silently producing
#: plausible nonsense from an artifact built by an older feature pipeline.
#:
#: v2 introduced the difficulty inference chain and trail-group propagation.
#: v3 made climb available to the cost model.
COST_MODEL_VERSION = 3

__all__ = ["ARTIFACT_SCHEMA_VERSION", "COST_MODEL_VERSION", "__version__"]
