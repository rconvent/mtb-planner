"""Measuring, scoring and diversifying candidate loops.

Overlap is the load-bearing measurement: it is what separates a loop from an
out-and-back, and the generator rejects on it. Scoring is asserted by its
monotonic relationships rather than by absolute values — the weights are tunable
data, so a test pinning the number they currently produce would break on every
retune while proving nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from planner_core.cost.costvector import materialise
from planner_core.cost.profiles import RideProfile
from planner_core.graph.store import GraphStore
from planner_core.routing.candidates import (
    DEFAULT_WEIGHTS,
    LoopMetrics,
    diversify,
    measure,
    overlap_ratio,
    score,
    similarity,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def costs(ardennes_graph: GraphStore, profile: RideProfile):
    """A cost vector over the fixture network."""
    return materialise(ardennes_graph, profile)


def metrics(**overrides: float) -> LoopMetrics:
    """Neutral metrics, overridable one field at a time."""
    base = {
        "time_s": 3_600.0,
        "distance_m": 20_000.0,
        "overlap_ratio": 0.0,
        "preference": 0.0,
        "singletrack": 0.0,
        "signed": 0.0,
        "paved": 0.0,
        "traffic": 0.0,
        "unknown": 0.0,
        "tech": 2.0,
    }
    base.update(overrides)
    return LoopMetrics(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------- overlap


def test_a_route_riding_every_edge_once_has_no_overlap(ardennes_graph: GraphStore):
    # Three distinct edges, assuming the fixture grid has at least that many.
    arcs = [0, 2, 4]
    edges = {int(ardennes_graph.arc_edge[a]) for a in arcs}
    assert len(edges) == 3

    assert overlap_ratio(ardennes_graph, arcs) == 0.0


def test_riding_an_edge_there_and_back_is_half_overlap(ardennes_graph: GraphStore):
    # Half the distance covered is a repeat of the other half.
    arc = 0
    twin = int(ardennes_graph.arc_rev[arc])

    assert overlap_ratio(ardennes_graph, [arc, twin]) == pytest.approx(0.5)


def test_an_empty_route_has_no_overlap(ardennes_graph: GraphStore):
    assert overlap_ratio(ardennes_graph, []) == 0.0


def test_overlap_counts_both_directions_as_the_same_ground(ardennes_graph: GraphStore):
    # A rider does not experience riding back down a trail as new trail.
    arc = 0
    twin = int(ardennes_graph.arc_rev[arc])

    assert overlap_ratio(ardennes_graph, [arc, twin]) == overlap_ratio(ardennes_graph, [arc, arc])


# --------------------------------------------------------------- measurement


def test_measurements_are_summed_from_the_arcs_ridden(ardennes_graph: GraphStore, costs):
    arcs = [0, 2, 4]

    result = measure(ardennes_graph, costs, arcs)

    assert result.time_s == pytest.approx(sum(costs.time_s[a] for a in arcs))
    assert result.distance_m == pytest.approx(sum(costs.distance_m[a] for a in arcs))


def test_an_empty_route_measures_as_nothing(ardennes_graph: GraphStore, costs):
    result = measure(ardennes_graph, costs, [])

    assert result.time_s == 0.0
    assert result.distance_m == 0.0


@pytest.mark.parametrize("field", ["singletrack", "signed", "paved", "traffic", "unknown"])
def test_every_fraction_is_a_fraction(ardennes_graph: GraphStore, costs, field):
    result = measure(ardennes_graph, costs, [0, 2, 4])

    assert 0.0 <= getattr(result, field) <= 1.0


def test_preference_stays_within_its_declared_range(ardennes_graph: GraphStore, costs):
    result = measure(ardennes_graph, costs, [0, 2, 4])

    assert -1.0 <= result.preference <= 1.0


# --------------------------------------------------------------- budget


def test_a_loop_landing_on_the_budget_fits_perfectly():
    assert metrics(time_s=3_600.0).budget_fit(3_600.0) == pytest.approx(1.0)


def test_a_loop_missing_the_budget_fits_worse():
    assert metrics(time_s=4_320.0).budget_fit(3_600.0) < 1.0


def test_fit_is_symmetric_about_the_budget():
    # Twenty minutes short is as wrong as twenty minutes long.
    assert metrics(time_s=2_400.0).budget_fit(3_600.0) == pytest.approx(
        metrics(time_s=4_800.0).budget_fit(3_600.0)
    )


def test_fit_never_goes_negative():
    assert metrics(time_s=100_000.0).budget_fit(3_600.0) == 0.0


def test_budget_error_says_which_way_the_miss_went():
    assert metrics(time_s=3_960.0).budget_error(3_600.0) == pytest.approx(0.10)
    assert metrics(time_s=3_240.0).budget_error(3_600.0) == pytest.approx(-0.10)


# --------------------------------------------------------------- scoring


def test_a_loop_the_rider_prefers_scores_higher(profile: RideProfile):
    better = score(metrics(preference=0.8), profile, 3_600.0)
    worse = score(metrics(preference=-0.8), profile, 3_600.0)

    assert better > worse


def test_overlap_costs_a_candidate_score(profile: RideProfile):
    assert score(metrics(overlap_ratio=0.0), profile, 3_600.0) > score(
        metrics(overlap_ratio=0.4), profile, 3_600.0
    )


def test_missing_the_budget_costs_a_candidate_score(profile: RideProfile):
    assert score(metrics(time_s=3_600.0), profile, 3_600.0) > score(
        metrics(time_s=5_400.0), profile, 3_600.0
    )


def test_singletrack_and_signed_route_help(profile: RideProfile):
    plain = score(metrics(), profile, 3_600.0)

    assert score(metrics(singletrack=1.0), profile, 3_600.0) > plain
    assert score(metrics(signed=1.0), profile, 3_600.0) > plain


@pytest.mark.parametrize("field", ["paved", "traffic", "unknown"])
def test_undesirable_qualities_cost_score(profile: RideProfile, field):
    assert score(metrics(), profile, 3_600.0) > score(metrics(**{field: 1.0}), profile, 3_600.0)


def test_technicality_is_a_band_match_not_more_is_better(profile: RideProfile):
    # A rider asking for flow trails is no happier on an S5 than on a towpath.
    on_target = score(metrics(tech=profile.tech_target), profile, 3_600.0)
    far_above = score(metrics(tech=profile.tech_target + 3.0), profile, 3_600.0)
    far_below = score(metrics(tech=max(0.0, profile.tech_target - 3.0)), profile, 3_600.0)

    assert on_target > far_above
    assert on_target > far_below


def test_the_weights_are_data_not_scattered_constants():
    # Retuning has to be a single edit, or half the places get missed.
    assert DEFAULT_WEIGHTS.overlap < 0.0
    assert DEFAULT_WEIGHTS.preference > 0.0


# --------------------------------------------------------------- similarity


def test_a_route_is_identical_to_itself(ardennes_graph: GraphStore):
    assert similarity(ardennes_graph, [0, 2, 4], [0, 2, 4]) == pytest.approx(1.0)


def test_routes_sharing_no_ground_are_dissimilar(ardennes_graph: GraphStore):
    left, right = [0], [4]
    assert int(ardennes_graph.arc_edge[0]) != int(ardennes_graph.arc_edge[4])

    assert similarity(ardennes_graph, left, right) == 0.0


def test_the_same_trail_ridden_backwards_is_the_same_trail(ardennes_graph: GraphStore):
    twin = int(ardennes_graph.arc_rev[0])

    assert similarity(ardennes_graph, [0], [twin]) == pytest.approx(1.0)


def test_partial_sharing_lands_between(ardennes_graph: GraphStore):
    value = similarity(ardennes_graph, [0, 2], [0, 4])

    assert 0.0 < value < 1.0


def test_an_empty_route_is_similar_to_nothing(ardennes_graph: GraphStore):
    assert similarity(ardennes_graph, [], [0]) == 0.0


# --------------------------------------------------------------- diversify


def test_the_best_candidate_is_always_offered(ardennes_graph: GraphStore):
    routes = [[0], [2], [4]]
    scores = [0.1, 0.9, 0.5]

    chosen = diversify(ardennes_graph, routes, scores, count=1)

    assert chosen == (1,)


def test_diversifying_returns_as_many_as_asked_for(ardennes_graph: GraphStore):
    routes = [[0], [2], [4]]

    assert len(diversify(ardennes_graph, routes, [0.9, 0.5, 0.1], count=2)) == 2


def test_it_cannot_return_more_than_it_was_given(ardennes_graph: GraphStore):
    routes = [[0], [2]]

    assert len(diversify(ardennes_graph, routes, [0.9, 0.5], count=5)) == 2


def test_no_candidates_yields_nothing(ardennes_graph: GraphStore):
    assert diversify(ardennes_graph, [], [], count=3) == ()


def test_a_near_duplicate_loses_to_a_worse_but_different_route(ardennes_graph: GraphStore):
    # The whole point: three routes where the second is a copy of the best and
    # the third is distinct but scores lower. The distinct one must be offered.
    routes = [[0, 2], [0, 2], [4]]
    scores = [0.90, 0.89, 0.60]

    chosen = diversify(ardennes_graph, routes, scores, count=2, gamma=0.35)

    assert chosen[0] == 0
    assert chosen[1] == 2


def test_zero_diversity_just_takes_the_top_scores(ardennes_graph: GraphStore):
    routes = [[0, 2], [0, 2], [4]]
    scores = [0.90, 0.89, 0.60]

    chosen = diversify(ardennes_graph, routes, scores, count=2, gamma=0.0)

    assert chosen == (0, 1)


def test_diversifying_is_deterministic(ardennes_graph: GraphStore):
    routes = [[0, 2], [0, 2], [4], [0, 4]]
    scores = [0.9, 0.9, 0.6, 0.7]

    first = diversify(ardennes_graph, routes, scores, count=3)
    second = diversify(ardennes_graph, routes, scores, count=3)

    assert first == second


def test_identical_scores_do_not_wobble(ardennes_graph: GraphStore):
    # A user reporting a route must be able to have it reproduced, which means
    # ties cannot be broken by whatever order the heap happened to produce.
    routes = [[0], [2], [4]]
    scores = [0.5, 0.5, 0.5]

    assert diversify(ardennes_graph, routes, scores, count=3) == diversify(
        ardennes_graph, routes, scores, count=3
    )


def test_every_returned_index_is_distinct(ardennes_graph: GraphStore):
    routes = [[0], [2], [4], [0, 2]]
    scores = [0.9, 0.8, 0.7, 0.6]

    chosen = diversify(ardennes_graph, routes, scores, count=4)

    assert len(set(chosen)) == len(chosen)
    assert set(chosen) <= set(range(len(routes)))


def test_scores_all_equal_and_routes_all_identical_still_terminates(
    ardennes_graph: GraphStore,
):
    routes = [[0], [0], [0]]

    chosen = diversify(ardennes_graph, routes, [0.5, 0.5, 0.5], count=3)

    assert len(chosen) == 3
    assert np.all(np.diff(sorted(chosen)) > 0)
