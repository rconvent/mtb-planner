"""Generating loops that fit a time budget.

The construction is a penalised two-tree cycle. **One** bounded Dijkstra from the
start yields every reachable turning point at once, along with the path to each;
picking a turning point and searching back on penalised costs closes a loop. That
is one search plus one search per candidate, rather than a search per candidate
pair, and it is what makes a dozen genuinely different loops affordable.

Three details do the real work:

**Pivots are spread across heading sectors.** Ranking turning points purely by
quality returns twelve variations on the same trail, because the best trail is
in one direction. Taking the best turning point in each of twelve compass
sectors is the single biggest reason the returned loops *look* different, which
is what a rider perceives as being offered a choice.

**Pivots are ranked without reconstructing their paths.** `cost / time` is a
preference density for free: cost is `time · (1 + λ·(1 − p_norm))`, so the ratio
is the length-weighted preference along the whole path to that node, already
computed by the Dijkstra. Reconstructing thousands of paths to rank them would
cost more than the search that produced them.

**The tree is grown once, generously, and `f_pivot` calibrated inside it.** The
fraction of the budget spent reaching the turning point is a guess — 0.45 for a
loop that wanders — and the achieved time says how wrong it was. Growing the
tree to a larger fraction than any calibration will use makes re-selection free,
because the nodes are already settled and their paths already recorded.

Everything here is deterministic: sector winners are chosen by argmin, ties fall
to the lower node index, and the calibration is arithmetic. The same request
returns the same loops, without a seed to pin it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from mtb_planner.config.logging import get_logger
from mtb_planner.core.errors import NoRouteFoundError
from mtb_planner.core.geo import BBox, LatLon, bearing_deg_array
from mtb_planner.cost.costvector import ArcCosts, materialise, max_speed_mps
from mtb_planner.cost.profiles import RideProfile
from mtb_planner.cost.speed import Rider
from mtb_planner.graph.spatial_index import SpatialIndex
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.ball import ball_around, loop_radius_m
from mtb_planner.routing.budget import Deadline
from mtb_planner.routing.candidates import LoopMetrics, diversify, measure, score
from mtb_planner.routing.penalties import penalise
from mtb_planner.routing.repair import is_connected, repair_to_budget
from mtb_planner.routing.result import Route, build_route
from mtb_planner.routing.router import Frontier, Router
from mtb_planner.routing.snap import Snap, snap_point

logger = get_logger(__name__)

#: Compass sectors pivots are spread across.
HEADING_SECTORS = 12

#: Fraction of the budget spent reaching the turning point, before calibration.
INITIAL_PIVOT_FRACTION = 0.45

#: How far the one Dijkstra is grown, as a fraction of the budget. Larger than
#: any calibrated `f_pivot`, so re-selection never needs a second search.
TREE_FRACTION = 0.65

#: Bounds on the calibrated pivot fraction. The upper bound stays under
#: `TREE_FRACTION` so calibration cannot ask for nodes the tree never settled.
MIN_PIVOT_FRACTION = 0.15
MAX_PIVOT_FRACTION = 0.62

#: Calibration rounds. Two lands within a few percent; a third earns nothing.
CALIBRATION_ROUNDS = 2

#: How many surviving candidates are repaired. Repair costs several searches
#: each, and the ones furthest from budget are not worth the attempt.
REPAIR_LIMIT = 6


@dataclass(frozen=True)
class LoopRequest:
    """What a rider asked for."""

    start: LatLon
    budget_s: float
    profile: RideProfile
    rider: Rider
    variants: int = 3
    tolerance: float = 0.10
    max_overlap: float = 0.25
    max_snap_distance_m: float = 250.0


@dataclass(frozen=True)
class LoopCandidate:
    """One generated loop, with everything ranking and reporting need."""

    route: Route
    metrics: LoopMetrics
    score: float
    within_budget: bool


@dataclass(frozen=True)
class LoopPlan:
    """The result of a loop request."""

    candidates: tuple[LoopCandidate, ...]
    start_snap: Snap
    settled: int
    truncated: bool


def _pivot_sectors(
    store: GraphStore,
    frontier: Frontier,
    start: int,
    low_s: float,
    high_s: float,
    sectors: int = HEADING_SECTORS,
) -> tuple[int, ...]:
    """The best turning point in each compass sector.

    "Best" is the lowest `cost / time`, which is the preference density of the
    path the Dijkstra already found to that node — no reconstruction needed.
    """
    band = frontier.within_time(low_s, high_s)
    band = band[band != start]
    if band.size == 0:
        return ()

    time_s = frontier.time_s[band]
    usable = time_s > 0.0
    band, time_s = band[usable], time_s[usable]
    if band.size == 0:
        return ()

    density = frontier.cost[band] / time_s
    bearings = bearing_deg_array(
        float(store.node_lat[start]),
        float(store.node_lon[start]),
        store.node_lat[band],
        store.node_lon[band],
    )
    sector_of = (bearings / (360.0 / sectors)).astype(np.int64) % sectors

    chosen: list[int] = []
    for sector in range(sectors):
        members = np.flatnonzero(sector_of == sector)
        if members.size == 0:
            continue
        # argmin takes the first minimum, so ties fall to the lower node index
        # and the result cannot wobble between identical requests.
        chosen.append(int(band[members[np.argmin(density[members])]]))
    return tuple(chosen)


def _close_loop(
    store: GraphStore,
    costs: ArcCosts,
    frontier: Frontier,
    start: int,
    pivot: int,
) -> tuple[int, ...]:
    """Ride out to `pivot` and back by a different way."""
    outbound = frontier.path_to(pivot)
    if not outbound:
        return ()

    # The return leg pays four times over for ground already covered, and the
    # reverse of the last outbound arc is removed outright — the kernels forbid
    # a U-turn within one search, but the seam between two searches is ours.
    returning = Router(
        store,
        penalise(costs, store, outbound, forbid=[int(store.arc_rev[outbound[-1]])]),
    ).shortest(pivot, start)

    if not returning.reached:
        return ()

    loop = (*outbound, *returning.arcs)
    return loop if is_connected(store, loop) else ()


def _build_candidates(
    store: GraphStore,
    costs: ArcCosts,
    frontier: Frontier,
    start: int,
    request: LoopRequest,
    deadline: Deadline | None,
) -> tuple[dict[tuple[int, ...], LoopMetrics], bool]:
    """Generate loops, calibrating the pivot fraction as it learns.

    Keyed by arc sequence, so the same loop found from two pivot fractions is
    stored once.

    **The first round always runs to completion, deadline or not.** A dozen A*
    searches is the smallest unit of work that can produce a loop at all, and a
    server that has already spent its budget queueing should answer with a
    slightly worse route rather than an error. Later rounds are what the deadline
    cuts, which is the anytime behaviour the design asks for: candidates have
    accumulated, so stopping costs quality rather than the answer.
    """
    found: dict[tuple[int, ...], LoopMetrics] = {}
    fraction = INITIAL_PIVOT_FRACTION
    truncated = False

    for round_number in range(CALIBRATION_ROUNDS):
        spent = round_number > 0 and deadline is not None and deadline.expired
        if spent:
            truncated = True
            break

        target = fraction * request.budget_s
        pivots = _pivot_sectors(store, frontier, start, target * 0.85, target * 1.10)

        achieved: list[float] = []
        for pivot in pivots:
            if round_number > 0 and deadline is not None and deadline.expired:
                truncated = True
                break
            loop = _close_loop(store, costs, frontier, start, pivot)
            if not loop:
                continue
            metrics = measure(store, costs, loop)
            achieved.append(metrics.time_s)
            found.setdefault(loop, metrics)

        if not achieved:
            break

        # Calibrate: the median loop tells us how much budget a unit of pivot
        # fraction actually buys. Clamped inside the tree that has been grown.
        median = statistics.median(achieved)
        if median <= 0.0 or round_number == CALIBRATION_ROUNDS - 1:
            break
        # A spent budget stops here, keeping the round just completed.
        if deadline is not None and deadline.expired:
            truncated = True
            break
        fraction = float(
            np.clip(
                fraction * request.budget_s / median,
                MIN_PIVOT_FRACTION,
                MAX_PIVOT_FRACTION,
            )
        )

    return found, truncated


def _repair_candidates(
    router: Router,
    store: GraphStore,
    costs: ArcCosts,
    loops: dict[tuple[int, ...], LoopMetrics],
    request: LoopRequest,
    deadline: Deadline | None,
) -> dict[tuple[int, ...], LoopMetrics]:
    """Nudge the most promising candidates onto the budget."""
    ordered = sorted(loops.items(), key=lambda item: abs(item[1].budget_error(request.budget_s)))
    repaired = dict(loops)

    for loop, metrics in ordered[:REPAIR_LIMIT]:
        if deadline is not None and deadline.expired:
            break
        if abs(metrics.budget_error(request.budget_s)) <= request.tolerance:
            continue
        fixed = repair_to_budget(router, store, loop, request.budget_s, request.tolerance)
        if fixed != loop and fixed:
            repaired.setdefault(fixed, measure(store, costs, fixed))

    return repaired


def _acceptable(
    metrics: LoopMetrics, loop: tuple[int, ...], store: GraphStore, request: LoopRequest
) -> bool:
    """Whether a candidate is a loop a rider could be offered at all."""
    if metrics.distance_m <= 0.0:
        return False
    if metrics.overlap_ratio > request.max_overlap:
        return False
    nodes = (store.arc_tail(loop[0]), int(store.arc_head[loop[-1]]))
    return nodes[0] == nodes[1]


def plan_loops(
    store: GraphStore,
    index: SpatialIndex,
    request: LoopRequest,
    coverage: BBox | None = None,
    deadline: Deadline | None = None,
) -> LoopPlan:
    """Generate loops from a point that fit a time budget.

    Candidates that miss the budget are returned rather than discarded when too
    few land inside it: every loop reports its own measured time, so a rider can
    judge a near miss for themselves. Returning nothing when a 90-minute request
    could be answered with a 78-minute loop is the worse failure.
    """
    start_snap = snap_point(store, index, request.start, request.max_snap_distance_m, coverage)
    start = start_snap.node

    radius = loop_radius_m(request.budget_s, max_speed_mps(request.rider))
    ball = ball_around(store, start, radius)
    costs = materialise(store, request.profile, request.rider, arc_mask=ball.arc_inside)
    router = Router(store, costs)

    # One bounded Dijkstra. Time bounds the ride; cost bounds the search, and
    # `cost <= (1 + λ)·time` is what makes that bound provably complete rather
    # than merely usually adequate.
    tree_time = TREE_FRACTION * request.budget_s
    frontier = router.reachable_within(
        start,
        cost_bound=(1.0 + request.profile.detour) * tree_time,
        time_bound=tree_time,
    )

    loops, truncated = _build_candidates(store, costs, frontier, start, request, deadline)
    if not loops:
        if truncated:
            raise NoRouteFoundError(
                reason="search_timeout",
                detail=(
                    "Building a loop from there took longer than this server "
                    "allows. Try a shorter time budget."
                ),
            )
        raise NoRouteFoundError(
            reason="no_loop_found",
            detail=(
                "No loop could be built from there within that time. The area may "
                "have too few connected trails, or the budget may be too short to "
                "leave and return."
            ),
        )

    loops = _repair_candidates(router, store, costs, loops, request, deadline)

    accepted = {
        loop: metrics
        for loop, metrics in loops.items()
        if _acceptable(metrics, loop, store, request)
    }
    if not accepted:
        raise NoRouteFoundError(
            reason="loops_overlap",
            detail=(
                "Every loop found doubles back on itself too much to be worth "
                "riding. There may be only one trail through this area."
            ),
        )

    # Prefer loops inside the budget, but fall back to near misses rather than
    # returning nothing.
    inside = {
        loop: m
        for loop, m in accepted.items()
        if abs(m.budget_error(request.budget_s)) <= request.tolerance
    }
    pool = inside if len(inside) >= request.variants else accepted

    routes = list(pool)
    scores = [score(pool[loop], request.profile, request.budget_s) for loop in routes]
    picked = diversify(store, routes, scores, count=request.variants)

    candidates = tuple(
        LoopCandidate(
            route=build_route(store, start, routes[i], costs.time_s),
            metrics=pool[routes[i]],
            score=scores[i],
            within_budget=abs(pool[routes[i]].budget_error(request.budget_s)) <= request.tolerance,
        )
        for i in picked
    )

    logger.info(
        "Loop request: %d candidates, %d offered, %d settled%s",
        len(accepted),
        len(candidates),
        frontier.settled,
        ", truncated" if truncated else "",
    )

    return LoopPlan(
        candidates=candidates,
        start_snap=start_snap,
        settled=frontier.settled,
        truncated=truncated,
    )
