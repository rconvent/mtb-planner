"""Bringing a route onto its time budget.

Written over "a path with fixed endpoints", not over "a loop". A loop is the
case where the two endpoints coincide, so point-to-point gets the same
behaviour from the same code: when the fastest A→B route comes in well under
budget, spending the slack well is exactly the `lengthen` problem.

Two moves, and both are what a rider does:

**Too long** — replace the least enjoyable stretch with the quickest way across
it. Nobody shortens a ride by cutting the good bit.

**Too short** — hang a detour off the route, chosen for what it adds. *"I have
twenty-five minutes spare, I'll take the loop at the back of the woods."*

Everything here is deterministic. No sampling, no shuffling, no seeded choice:
ties resolve by position along the route, so the same request returns the same
repair every time without needing a seed to pin it.

Nothing is ever accepted that fails to improve the budget error, and nothing is
ever accepted that is not a genuinely connected path — so the worst case is that
the route comes back exactly as it went in.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from mtb_planner.cost.costvector import ArcCosts
from mtb_planner.graph.store import GraphStore
from mtb_planner.routing.penalties import penalise
from mtb_planner.routing.router import Router

#: Window sizes tried when shortening, as fractions of total riding time. Four
#: attempts is enough to find a stretch worth replacing without turning repair
#: into a search of its own.
_SHORTEN_WINDOWS = (0.50, 0.35, 0.25, 0.15)

#: Where along the route a detour is attempted, as fractions of the way round.
_DETOUR_POINTS = (0.5, 0.25, 0.75)

#: How many shorten/lengthen passes to make. Each one moves the route
#: substantially closer; more than a few means the budget is unreachable here.
MAX_ROUNDS = 3


def route_nodes(store: GraphStore, arcs: Sequence[int]) -> tuple[int, ...]:
    """The nodes an arc sequence visits, start included."""
    if not arcs:
        return ()
    return (store.arc_tail(arcs[0]), *(int(store.arc_head[a]) for a in arcs))


def is_connected(store: GraphStore, arcs: Sequence[int]) -> bool:
    """Whether consecutive arcs actually meet.

    Every repair is checked against this before being accepted. A spliced route
    that does not connect is not a worse route — it is not a route.
    """
    return all(
        int(store.arc_head[first]) == store.arc_tail(second)
        for first, second in itertools.pairwise(arcs)
    )


def fastest(costs: ArcCosts) -> ArcCosts:
    """The same graph costed by time alone.

    Used to cross a stretch as quickly as possible, which is what shortening
    needs: preference is what got the route into trouble, so it does not get a
    say in the way out.
    """
    return replace(costs, cost=costs.time_s)


def _time_of(costs: ArcCosts, arcs: Sequence[int]) -> float:
    """Riding time for an arc sequence."""
    if not arcs:
        return 0.0
    return float(costs.time_s[np.asarray(arcs, dtype=np.intp)].sum())


def _worst_windows(costs: ArcCosts, arcs: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Stretches of the route the rider would miss least, worst first.

    One window per size in `_SHORTEN_WINDOWS`: the contiguous run of that
    duration with the lowest time-weighted preference.
    """
    index = np.asarray(arcs, dtype=np.intp)
    times = costs.time_s[index]
    weighted = costs.preference[index] * times
    total = float(times.sum())
    if total <= 0.0:
        return ()

    cumulative_time = np.concatenate(([0.0], np.cumsum(times)))
    cumulative_pref = np.concatenate(([0.0], np.cumsum(weighted)))

    windows: list[tuple[float, tuple[int, int]]] = []
    for fraction in _SHORTEN_WINDOWS:
        target = total * fraction
        best: tuple[float, tuple[int, int]] | None = None
        for start in range(len(index)):
            # First end whose window covers at least the target duration.
            end = int(np.searchsorted(cumulative_time, cumulative_time[start] + target)) + 1
            if end > len(index):
                break
            span = cumulative_time[end] - cumulative_time[start]
            if span <= 0.0:
                continue
            mean_pref = (cumulative_pref[end] - cumulative_pref[start]) / span
            if best is None or mean_pref < best[0]:
                best = (float(mean_pref), (start, end))
        if best is not None and best[1] not in [w for _, w in windows]:
            windows.append(best)

    windows.sort(key=lambda item: item[0])
    return tuple(window for _, window in windows)


def shorten(
    router: Router,
    store: GraphStore,
    arcs: Sequence[int],
    budget_s: float,
) -> tuple[int, ...]:
    """Replace the least enjoyable stretch with the quickest way across it."""
    if not arcs:
        return tuple(arcs)

    costs = router.costs
    current = _time_of(costs, arcs)
    excess = current - budget_s
    if excess <= 0.0:
        return tuple(arcs)

    nodes = route_nodes(store, arcs)
    quick = Router(store, fastest(costs))

    for start, end in _worst_windows(costs, arcs):
        entry, exit_ = nodes[start], nodes[end]
        if entry == exit_:
            continue

        crossing = quick.shortest(entry, exit_)
        if not crossing.reached:
            continue

        candidate = (*arcs[:start], *crossing.arcs, *arcs[end:])
        if not is_connected(store, candidate):
            continue

        improved = _time_of(costs, candidate)
        # Only accept a genuine improvement: closer to the budget than before,
        # and not overshooting into being too short.
        if abs(improved - budget_s) < abs(current - budget_s):
            return candidate

    return tuple(arcs)


def _best_detour(
    router: Router,
    store: GraphStore,
    node: int,
    spare_s: float,
) -> tuple[int, ...]:
    """A small loop off `node` that spends about `spare_s` enjoyably.

    Built the same way as the main generator builds a loop, because that is what
    it is: reach out along a preference-weighted search, then come back on a
    penalised return so the detour is a loop rather than a spur.
    """
    if spare_s <= 0.0:
        return ()

    frontier = router.reachable_within(node, time_bound=spare_s / 2.0)
    band = frontier.within_time(spare_s * 0.25, spare_s * 0.5)
    if band.size == 0:
        return ()

    # The turn-around worth riding to: most preference gained per second spent.
    best_arcs: tuple[int, ...] = ()
    best_gain = -np.inf
    for target in band:
        outbound = frontier.path_to(int(target))
        if not outbound:
            continue
        spent = _time_of(router.costs, outbound)
        if spent <= 0.0:
            continue
        gain = float(router.costs.preference[np.asarray(outbound, dtype=np.intp)].mean())
        if gain > best_gain:
            best_gain = gain
            best_arcs = outbound

    if not best_arcs:
        return ()

    # Come back a different way, and never straight back down the last arc.
    returning = Router(
        store,
        penalise(
            router.costs,
            store,
            best_arcs,
            forbid=[int(store.arc_rev[best_arcs[-1]])],
        ),
    ).shortest(int(store.arc_head[best_arcs[-1]]), node)

    if not returning.reached:
        return ()

    detour = (*best_arcs, *returning.arcs)
    return detour if is_connected(store, detour) else ()


def lengthen(
    router: Router,
    store: GraphStore,
    arcs: Sequence[int],
    budget_s: float,
) -> tuple[int, ...]:
    """Hang a detour off the route to spend time left over."""
    if not arcs:
        return tuple(arcs)

    costs = router.costs
    current = _time_of(costs, arcs)
    spare = budget_s - current
    if spare <= 0.0:
        return tuple(arcs)

    nodes = route_nodes(store, arcs)

    for fraction in _DETOUR_POINTS:
        position = min(len(arcs), max(1, round(len(arcs) * fraction)))
        detour = _best_detour(router, store, nodes[position], spare)
        if not detour:
            continue

        candidate = (*arcs[:position], *detour, *arcs[position:])
        if not is_connected(store, candidate):
            continue

        improved = _time_of(costs, candidate)
        if abs(improved - budget_s) < abs(current - budget_s):
            return candidate

    return tuple(arcs)


def repair_to_budget(
    router: Router,
    store: GraphStore,
    arcs: Sequence[int],
    budget_s: float,
    tolerance: float,
    max_rounds: int = MAX_ROUNDS,
) -> tuple[int, ...]:
    """Bring a path with fixed endpoints onto its time budget.

    Returns the input unchanged when it is already within tolerance, or when no
    move improves it — repair never makes a route worse.
    """
    current = tuple(int(a) for a in arcs)
    if not current or budget_s <= 0.0:
        return current

    for _ in range(max_rounds):
        elapsed = _time_of(router.costs, current)
        error = (elapsed - budget_s) / budget_s
        if abs(error) <= tolerance:
            return current

        repaired = (
            shorten(router, store, current, budget_s)
            if error > 0.0
            else lengthen(router, store, current, budget_s)
        )
        if repaired == current:
            return current
        current = repaired

    return current


def budget_error(costs: ArcCosts, arcs: Sequence[int], budget_s: float) -> float:
    """Signed fractional deviation of a route from its budget."""
    if budget_s <= 0.0:
        return 0.0
    return (_time_of(costs, arcs) - budget_s) / budget_s


def arc_times(costs: ArcCosts, arcs: Sequence[int]) -> npt.NDArray[np.float64]:
    """Per-arc riding time along a route, for callers that need the profile."""
    if not arcs:
        return np.zeros(0, dtype=np.float64)
    return costs.time_s[np.asarray(arcs, dtype=np.intp)]
