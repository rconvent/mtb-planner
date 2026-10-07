"""The Phase 8 gate, measured on a real extract.

The plan sets four numbers for the loop generator: at least 90% of requests
return three or more loops, within ±10% of the requested time, with under 25% of
each loop ridden twice, at a p95 under two seconds.

None of that can be measured on a synthetic fixture — the whole question is
whether the construction copes with the shape of real trail networks, where the
good riding clusters and the connections between clusters are roads. So this
reads the built Luxembourg artifact and is skipped when it is absent, which
means it does not run in a clean checkout. `test_api_loop.py` covers behaviour;
this covers whether the behaviour is good enough.

Marked slow: it runs a few dozen full loop requests.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from mtb_planner.speed import Rider
from planner_core.core.errors import PlanningError
from planner_core.core.geo import LatLon
from planner_core.cost.profiles import get_profile
from planner_core.graph.serialize import load_artifact
from planner_core.graph.spatial_index import SpatialIndex
from planner_core.graph.store import GraphStore
from planner_core.routing.kernels import warm_up
from planner_core.routing.loop import LoopRequest, plan_loops

ARTIFACT = Path("data/graph/lux-v1")

#: How many start points to sample. Enough for a 90% pass rate to mean
#: something, few enough to finish in about a minute.
SAMPLES = 30

#: Budgets to sweep, in minutes. Spread across what a rider actually asks for.
BUDGETS = (45, 90, 150)

REQUIRED_LOOPS = 3
TOLERANCE = 0.10
MAX_OVERLAP = 0.25
PASS_RATE = 0.90
P95_SECONDS = 2.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (ARTIFACT / "manifest.json").exists(),
        reason=f"no built artifact at {ARTIFACT}; run `mtb build` first",
    ),
]


@pytest.fixture(scope="module")
def artifact() -> tuple[GraphStore, SpatialIndex]:
    """The real Luxembourg graph, loaded once."""
    warm_up()
    store, _ = load_artifact(ARTIFACT)
    return store, SpatialIndex(store)


@pytest.fixture(scope="module")
def starts(artifact: tuple[GraphStore, SpatialIndex]) -> list[LatLon]:
    """Sampled start points in the routable component.

    Seeded, so a failure is reproducible: the same thirty places every run.
    """
    store, _ = artifact
    rng = np.random.default_rng(20260812)
    routable = np.flatnonzero(store.node_scc == store.giant_scc)
    picked = rng.choice(routable, size=SAMPLES, replace=False)
    return [store.node_point(int(node)) for node in picked]


@pytest.fixture(scope="module")
def outcomes(
    artifact: tuple[GraphStore, SpatialIndex], starts: list[LatLon]
) -> list[dict[str, object]]:
    """One loop request per start, at a rotating budget."""
    store, index = artifact
    results: list[dict[str, object]] = []

    for position, start in enumerate(starts):
        budget = BUDGETS[position % len(BUDGETS)]
        request = LoopRequest(
            start=start,
            budget_s=budget * 60.0,
            profile=get_profile("tech"),
            athlete=Rider(),
            variants=REQUIRED_LOOPS,
        )
        began = time.perf_counter()
        try:
            plan = plan_loops(store, index, request, coverage=store.bbox())
            candidates = plan.candidates
            failure: str | None = None
        except PlanningError as exc:
            candidates = ()
            failure = type(exc).__name__
        elapsed = time.perf_counter() - began

        results.append(
            {
                "budget_minutes": budget,
                "seconds": elapsed,
                "failure": failure,
                "inside": [
                    c for c in candidates if abs(c.metrics.budget_error(budget * 60.0)) <= TOLERANCE
                ],
                "candidates": candidates,
            }
        )
    return results


def test_most_requests_return_three_loops_on_budget(outcomes):
    # The headline gate. Counted per request, not per loop: two good loops is a
    # worse answer than three, because a rider is choosing between them.
    satisfied = [o for o in outcomes if len(o["inside"]) >= REQUIRED_LOOPS]
    rate = len(satisfied) / len(outcomes)

    detail = "\n".join(
        f"  {o['budget_minutes']:>3} min: {len(o['inside'])}/{len(o['candidates'])} on budget"
        f"{' — ' + str(o['failure']) if o['failure'] else ''}"
        for o in outcomes
        if len(o["inside"]) < REQUIRED_LOOPS
    )
    assert rate >= PASS_RATE, (
        f"only {rate:.0%} of requests returned {REQUIRED_LOOPS} loops within "
        f"±{TOLERANCE:.0%} (need {PASS_RATE:.0%}).\nShortfalls:\n{detail}"
    )


def test_no_loop_doubles_back_beyond_the_limit(outcomes):
    for outcome in outcomes:
        for candidate in outcome["candidates"]:
            assert candidate.metrics.overlap_ratio <= MAX_OVERLAP


def test_the_slowest_requests_stay_under_two_seconds(outcomes):
    times = sorted(o["seconds"] for o in outcomes)
    p95 = times[max(0, int(len(times) * 0.95) - 1)]

    assert p95 < P95_SECONDS, (
        f"p95 {p95:.2f}s exceeds {P95_SECONDS}s "
        f"(median {statistics.median(times):.2f}s, worst {times[-1]:.2f}s)"
    )


def test_every_returned_loop_actually_closes(outcomes):
    for outcome in outcomes:
        for candidate in outcome["candidates"]:
            assert candidate.route.is_loop


def test_every_returned_loop_is_distinct(outcomes):
    # Three near-identical loops is one loop offered three times.
    for outcome in outcomes:
        arc_sets = [frozenset(c.route.arcs) for c in outcome["candidates"]]
        assert len(set(arc_sets)) == len(arc_sets)


def test_reported_time_is_measured_not_inferred(outcomes):
    # The separation invariant, on real data: what the rider is told must be the
    # sum of the arcs ridden, never the preference-weighted cost that chose them.
    for outcome in outcomes:
        for candidate in outcome["candidates"]:
            assert candidate.metrics.time_s == pytest.approx(
                candidate.route.stats.moving_time_s, rel=1e-9
            )


def test_requests_that_fail_are_rare(outcomes):
    failures = [o for o in outcomes if o["failure"]]

    assert len(failures) <= len(outcomes) * (1.0 - PASS_RATE), (
        f"{len(failures)} of {len(outcomes)} requests raised: "
        f"{ {o['failure'] for o in failures} }"
    )
