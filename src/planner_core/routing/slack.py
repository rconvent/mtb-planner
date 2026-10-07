"""Spending the slack in a point-to-point time budget.

A rider who says "I have two hours to get from A to B" has not asked for a
two-hour route. They have said the fastest way there is not the point, and that
anything up to two hours is affordable. The difference between the quickest
route and the budget is slack, and slack is what buys trail.

The lever already exists. `cost = t · (1 + λ · (1 − p_norm))` trades time for
preference, and λ is the dial: at the profile's own λ the planner takes the
route that profile describes, and every increase makes it willing to ride
further for better ground. Raising λ raises the route's time, so the question
"what is the best route inside this budget" becomes "what is the largest λ whose
route still fits" — one scalar, found by bisection, rather than a
resource-constrained shortest path, which is NP-hard and would need its own
search.

Two properties make this affordable enough to do inside one request:

1. λ touches nothing but the final multiplication. Time is physics and
   preference is the weighted features; neither moves when λ does. So each
   probe re-scales an already-materialised cost vector — about 15 ms — instead
   of solving the physics again, which is 500 ms on a country-sized graph.
2. Every probe is a normal A* under a normal cost vector, so nothing here
   weakens the invariants: cost never dips below time, and the time reported to
   the rider is still summed from the arcs rather than read off the cost.

Time is not guaranteed to increase monotonically with λ — a much higher λ can
find a different, shorter way through better ground. The bisection therefore
keeps the best *feasible* route it has actually seen rather than trusting the
bracket, so a non-monotonic step costs a little search quality and never
returns a route over budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from planner_core.config.logging import get_logger
from planner_core.cost.costvector import ArcCosts
from planner_core.graph.store import GraphStore
from planner_core.routing.budget import Deadline
from planner_core.routing.router import Router, SearchResult

logger = get_logger(__name__)

#: How many times the bisection halves the bracket. Each step is one search, so
#: this is a direct cost: six probes at roughly 25 ms is a fifth of a second on
#: the Belgian graph. Five steps locate λ to within about 3% of the range,
#: which is finer than the difference it makes to a route.
SLACK_STEPS = 5


@dataclass(frozen=True)
class Slack:
    """The best route found inside a time budget, and what it cost to find."""

    result: SearchResult
    costs: ArcCosts
    detour: float
    """The λ the returned route was planned under."""

    time_s: float
    within_budget: bool
    """Whether the route fits. False only when even the profile's own route is
    slower than the budget, which is a fact about the ride, not a failure."""

    probes: int
    """Searches spent, the baseline included."""


def spend_slack(
    store: GraphStore,
    costs: ArcCosts,
    source: int,
    target: int,
    budget_s: float,
    floor: float,
    ceiling: float,
    baseline: SearchResult,
    deadline: Deadline | None = None,
) -> Slack:
    """Find the largest λ in `[floor, ceiling]` whose route fits the budget.

    `baseline` is the route already planned at `floor`, passed in rather than
    re-searched: the caller needs it anyway to answer at all, and a route that
    does not fit the budget is still the answer.
    """
    # Timed through the router rather than summed here, so every figure this
    # module compares against the budget is the one the rider is shown. Adding
    # the same arcs a different way differs in the last bits, which is enough
    # to report a route that fits as over.
    baseline_s = Router(store, costs).time_of(baseline.arcs)
    best = Slack(
        result=baseline,
        costs=costs,
        detour=floor,
        time_s=baseline_s,
        within_budget=baseline_s <= budget_s,
        probes=1,
    )

    # No slack to spend, or none worth spending. The profile's own route
    # already runs past the budget, so a greedier λ can only make it worse.
    if not best.within_budget or ceiling <= floor:
        return best

    probes = 1

    def probe(detour: float) -> Slack | None:
        """Plan at one λ, returning it only if the route fits the budget."""
        nonlocal probes
        scaled = costs.with_detour(detour)
        router = Router(store, scaled)
        result = router.shortest_within(source, target, deadline=deadline)
        probes += 1
        if not result.reached:
            return None
        elapsed = router.time_of(result.arcs)
        if elapsed > budget_s:
            return None
        return Slack(
            result=result,
            costs=scaled,
            detour=detour,
            time_s=elapsed,
            within_budget=True,
            probes=probes,
        )

    # The greediest route first: when the whole budget covers it there is
    # nothing to bisect, and on a generous budget that is the common case.
    greediest = probe(ceiling)
    if greediest is not None:
        return replace(greediest, probes=probes)

    low, high = floor, ceiling
    for _ in range(SLACK_STEPS):
        if deadline is not None and deadline.expired:
            logger.info("Slack search stopped after %d probes: budget spent", probes)
            break
        middle = (low + high) / 2.0
        fits = probe(middle)
        if fits is None:
            high = middle
        else:
            low = middle
            best = fits

    return replace(best, probes=probes)
