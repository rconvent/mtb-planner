"""Trail running route planner.

The same engine as the mountain-bike planner — `planner_core` — with what makes
it a runner's: who may go where on foot, how hard a path is on the hiking
scale, and how fast a runner covers ground on a slope.
"""

#: Bumped whenever edge features or the cost/time model change in a way that
#: alters routing results for runners. Independent of the bike's: the two
#: models move separately.
COST_MODEL_VERSION = 1

__all__ = ["COST_MODEL_VERSION"]
