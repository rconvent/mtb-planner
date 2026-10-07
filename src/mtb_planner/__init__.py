"""Mountain bike route planner.

Time-budgeted loop and point-to-point routing over OpenStreetMap data, scored
on inferred trail complexity rather than generic cycling suitability. The
engine is `planner_core`; this package is what makes it a bike planner — who
may ride where, how hard a trail is on `mtb:scale`, and how fast a rider goes.
"""

#: Bumped whenever edge features or the cost/time model change in a way that
#: alters routing results. Guards against a new server silently producing
#: plausible nonsense from an artifact built by an older feature pipeline.
#:
#: v2 introduced the difficulty inference chain and trail-group propagation.
#: v3 made climb available to the cost model.
COST_MODEL_VERSION = 3

__all__ = ["COST_MODEL_VERSION"]
