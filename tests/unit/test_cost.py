"""Speed, profiles, and the cost the search actually minimises.

The invariant tests are the important ones. Three properties of
`cost = t·(1 + λ(1 − p_norm))` hold the design together: cost never dips below
time, so the A* heuristic stays admissible; cost never exceeds `(1+λ)·time`,
which bounds a budget-limited search; and time is measured separately from
cost, so the figures shown to a rider are not inferred from the thing that
drove the routing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mtb_planner.speed import (
    DESCENT_CAP_M_S,
    ROLLING_RESISTANCE,
    Rider,
    solve_speed,
    speed_table,
)
from mtb_planner.sport import MTB
from planner_core.core.errors import BuildError
from planner_core.core.units import MAX_SPEED_M_S, MIN_SPEED_M_S
from planner_core.cost.costvector import TECH_CEILING_MARGIN, materialise, preference
from planner_core.cost.profiles import FEATURE_NAMES, get_profile, load_profiles
from planner_core.cost.speed import GRADE_BAND_CENTRES, TECH_BUCKETS
from planner_core.elevation.profile import GRADE_BANDS
from planner_core.features.enums import PAVED_SURFACES, SurfaceClass
from planner_core.graph.store import GraphStore

pytestmark = pytest.mark.unit


def speed_on(grade: float, surface: SurfaceClass, rider: Rider | None = None) -> float:
    """Speed in m/s on one grade and surface, ignoring technicality."""
    return float(
        solve_speed(
            np.array([grade]),
            np.array([ROLLING_RESISTANCE[surface]]),
            rider or Rider(),
        )[0]
    )


# --------------------------------------------------------------- speed model


def test_flat_tarmac_speed_is_plausible():
    # A fit rider at 200 W on the flat, not a track sprinter.
    km_h = speed_on(0.0, SurfaceClass.PAVED) * 3.6

    assert 20.0 < km_h < 32.0


def test_flat_hardpack_is_slower_than_tarmac():
    assert speed_on(0.0, SurfaceClass.GROUND) < speed_on(0.0, SurfaceClass.PAVED)


def test_mud_is_far_slower_than_tarmac():
    # Rolling resistance spans more than a factor of ten across surfaces,
    # which is why classifying them earns its place.
    assert speed_on(0.0, SurfaceClass.SAND_MUD) < 0.6 * speed_on(0.0, SurfaceClass.PAVED)


@pytest.mark.parametrize("grade", [0.04, 0.08, 0.15])
def test_climbing_is_slower_than_flat(grade):
    assert speed_on(grade, SurfaceClass.GROUND) < speed_on(0.0, SurfaceClass.GROUND)


def test_climbing_slows_monotonically_with_grade():
    speeds = [speed_on(g, SurfaceClass.GROUND) for g in (0.0, 0.04, 0.08, 0.12, 0.18)]

    assert speeds == sorted(speeds, reverse=True)


def test_a_steep_climb_is_walking_pace():
    km_h = speed_on(0.15, SurfaceClass.GROUND) * 3.6

    assert 2.0 < km_h < 8.0


def test_speed_never_leaves_its_physical_bounds():
    grades = np.linspace(-0.5, 0.5, 41)
    rolling = np.full(grades.shape, ROLLING_RESISTANCE[SurfaceClass.GROUND])

    speeds = solve_speed(grades, rolling, Rider())

    assert np.all(speeds >= MIN_SPEED_M_S)
    assert np.all(speeds <= MAX_SPEED_M_S)


def test_more_power_means_more_speed():
    strong = speed_on(0.06, SurfaceClass.GROUND, Rider(power_w=280.0))
    casual = speed_on(0.06, SurfaceClass.GROUND, Rider(power_w=130.0))

    assert strong > casual


# --------------------------------------------------------------- speed table


def test_the_table_covers_every_band_surface_and_grade():
    table = speed_table(Rider())

    assert table.shape == (GRADE_BANDS, len(SurfaceClass), TECH_BUCKETS)
    assert np.all(table > 0.0)


def test_technical_descents_are_capped_below_easy_ones():
    # Gravity does not care how rough the trail is; the rider does.
    table = speed_table(Rider())

    easy = table[0, int(SurfaceClass.GROUND), 0]
    hard = table[0, int(SurfaceClass.GROUND), 5]
    assert hard < easy


def test_a_descent_is_never_faster_than_its_cap():
    table = speed_table(Rider())
    descending = GRADE_BAND_CENTRES < 0

    for tech in range(TECH_BUCKETS):
        limit = DESCENT_CAP_M_S[tech] * 1.25  # the most generous surface factor
        assert np.all(table[descending, :, tech] <= limit + 1e-9)


def test_the_cap_applies_on_the_flat_too():
    # Nobody rides flat S3 singletrack at 25 km/h.
    table = speed_table(Rider())
    flat = GRADE_BANDS // 2

    assert table[flat, int(SurfaceClass.GROUND), 4] < table[flat, int(SurfaceClass.GROUND), 0]


def test_a_cautious_rider_descends_slower():
    bold = speed_table(Rider(descent_skill=1.25))
    timid = speed_table(Rider(descent_skill=0.75))

    assert timid[0, int(SurfaceClass.GROUND), 2] < bold[0, int(SurfaceClass.GROUND), 2]


# --------------------------------------------------------------- profiles


def test_the_shipped_profiles_load():
    profiles = load_profiles(MTB.profiles_path)

    assert {"xc", "flow", "tech", "gravity"} <= set(profiles)


def test_every_shipped_profile_weights_only_known_features():
    for profile in load_profiles(MTB.profiles_path).values():
        assert len(profile.weights) == len(FEATURE_NAMES)


def test_profiles_differ_in_what_they_want():
    profiles = load_profiles(MTB.profiles_path)

    # The whole point: a tech rider is looking for rougher ground than an
    # XC rider, and will detour further to find it.
    assert profiles["tech"].tech_target > profiles["xc"].tech_target
    assert profiles["tech"].detour > profiles["xc"].detour


def test_the_gravity_profile_rewards_descent_and_resents_climbing():
    gravity = get_profile("gravity")

    assert gravity.weight_of("descent") > 0.0
    assert gravity.weight_of("climb") < 0.0


def test_every_profile_penalises_traffic():
    for profile in load_profiles(MTB.profiles_path).values():
        assert profile.weight_of("traffic") < 0.0


def test_an_unknown_profile_names_the_alternatives():
    with pytest.raises(BuildError, match="Available:"):
        get_profile("downhill-only")


def test_a_missing_profile_file_is_reported_clearly(tmp_path: Path):
    with pytest.raises(BuildError, match="No profiles"):
        load_profiles(tmp_path / "absent.yaml")


def test_malformed_yaml_is_reported_clearly(tmp_path: Path):
    path = tmp_path / "profiles.yaml"
    path.write_text("profiles: [\n", encoding="utf-8")

    with pytest.raises(BuildError, match="not valid YAML"):
        load_profiles(path)


def test_a_profile_weighting_an_unknown_feature_is_rejected(tmp_path: Path):
    # Silently ignoring it would leave a rider wondering why a slider did
    # nothing at all.
    path = tmp_path / "profiles.yaml"
    path.write_text(
        "profiles:\n  - id: x\n    name: X\n    tech_target: 1\n    tech_sigma: 1\n"
        "    tech_max: 3\n    detour: 1\n    weights: {rockiness: 1.0}\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildError, match="unknown feature"):
        load_profiles(path)


def test_overrides_leave_the_original_untouched():
    original = get_profile("flow")

    adjusted = original.with_overrides(tech_target=4.0, detour=0.0)

    assert adjusted.tech_target == pytest.approx(4.0)
    assert original.tech_target != pytest.approx(4.0)


def test_a_weight_override_replaces_only_that_weight():
    original = get_profile("flow")

    adjusted = original.with_overrides(weights={"singletrack": 0.0})

    assert adjusted.weight_of("singletrack") == pytest.approx(0.0)
    assert adjusted.weight_of("paved") == pytest.approx(original.weight_of("paved"))


def test_overriding_an_unknown_feature_is_rejected():
    with pytest.raises(BuildError, match="Unknown feature"):
        get_profile("flow").with_overrides(weights={"vibes": 1.0})


# --------------------------------------------------------------- preference


def test_preference_stays_within_its_bounds(ardennes_graph: GraphStore):
    grade = np.zeros(ardennes_graph.n_edges, dtype=np.float64)

    scores = preference(ardennes_graph.features, grade, get_profile("flow"))

    assert np.all(scores >= -1.0)
    assert np.all(scores <= 1.0)


def test_difficulty_is_a_band_match_not_more_is_better(ardennes_graph: GraphStore):
    # An XC rider is no happier on an S5 than on a towpath, so a profile
    # targeting S1 must score an S1 above an S5.
    features = ardennes_graph.features
    grade = np.zeros(ardennes_graph.n_edges, dtype=np.float64)
    easy = get_profile("xc")

    on_target = np.abs(features.tech - easy.tech_target) < 0.5
    far_above = features.tech > easy.tech_target + 2.0
    if not (on_target.any() and far_above.any()):
        pytest.skip("fixture lacks the spread of technicality this compares")

    scores = preference(features, grade, easy)
    assert scores[on_target].mean() > scores[far_above].mean()


# --------------------------------------------------------------- the invariant


def test_a_sealed_road_is_never_a_technicality_match(lattice_graph: GraphStore):
    # The band scores how close an edge is to the difficulty the rider wants,
    # and a road scores tech 0. Ungated, a low target paid tarmac a bonus large
    # enough to cancel the profile's own dislike of it, and the planner sent
    # mountain bikers down the road. Moving the target must therefore change
    # what unpaved edges are worth and leave paved ones exactly where they were.
    features = lattice_graph.features
    grade = np.zeros(lattice_graph.n_edges, dtype=np.float64)
    paved = np.isin(features.surface_class, [int(surface) for surface in PAVED_SURFACES])
    assert paved.any() and (~paved).any(), "fixture needs both surfaces to compare"

    base = get_profile("xc")
    near = preference(features, grade, base.with_overrides(tech_target=0.0))
    far = preference(features, grade, base.with_overrides(tech_target=5.0))

    np.testing.assert_allclose(near[paved], far[paved])
    assert not np.allclose(near[~paved], far[~paved])


def test_rescaling_the_detour_matches_materialising_it(ardennes_graph: GraphStore):
    # The point-to-point slack search re-costs an existing vector instead of
    # solving the physics again. It is only allowed to do that while the two
    # routes agree exactly.
    profile = get_profile("flow")
    greedier = 3.0

    direct = materialise(ardennes_graph, profile.with_overrides(detour=greedier))
    rescaled = materialise(ardennes_graph, profile).with_detour(greedier)

    np.testing.assert_allclose(rescaled.cost, direct.cost)
    np.testing.assert_allclose(rescaled.time_s, direct.time_s)
    np.testing.assert_allclose(rescaled.preference, direct.preference)
    np.testing.assert_array_equal(rescaled.allowed, direct.allowed)


def test_rescaling_leaves_the_measurements_alone(ardennes_graph: GraphStore):
    # Only the cost may move. Time, distance and preference are what the rider
    # is shown and what the budget is tested against.
    costs = materialise(ardennes_graph, get_profile("flow"))

    rescaled = costs.with_detour(5.0)

    np.testing.assert_allclose(rescaled.time_s, costs.time_s)
    np.testing.assert_allclose(rescaled.distance_m, costs.distance_m)
    np.testing.assert_allclose(rescaled.preference, costs.preference)
    assert np.all(rescaled.cost >= costs.cost - 1e-9)


def test_cost_never_dips_below_time(ardennes_graph: GraphStore):
    # This is what makes straight-line distance an admissible heuristic.
    costs = materialise(ardennes_graph, get_profile("flow"))

    assert np.all(costs.cost >= costs.time_s - 1e-9)


def test_cost_is_bounded_by_the_detour_willingness(ardennes_graph: GraphStore):
    # The bound a budget-limited search relies on.
    profile = get_profile("tech")

    costs = materialise(ardennes_graph, profile)

    assert np.all(costs.cost <= costs.time_s * (1.0 + profile.detour) + 1e-9)


def test_no_detour_willingness_means_cost_is_exactly_time(ardennes_graph: GraphStore):
    fastest = get_profile("flow").with_overrides(detour=0.0)

    costs = materialise(ardennes_graph, fastest)

    np.testing.assert_allclose(costs.cost, costs.time_s)


def test_time_is_positive_wherever_there_is_distance(ardennes_graph: GraphStore):
    costs = materialise(ardennes_graph, get_profile("flow"))

    assert np.all(costs.time_s[costs.distance_m > 0] > 0.0)


def test_the_heuristic_scale_never_overestimates(ardennes_graph: GraphStore):
    # It must be a lower bound on seconds per metre, or A* stops being optimal.
    costs = materialise(ardennes_graph, get_profile("flow"))

    moving = costs.distance_m > 0
    assert costs.min_seconds_per_metre <= (costs.time_s[moving] / costs.distance_m[moving]).min()
    assert costs.min_seconds_per_metre > 0.0


def test_a_profile_ceiling_removes_trails_beyond_it(ardennes_graph: GraphStore):
    # A safety boundary, not a preference: never route a beginner down an S4.
    timid = get_profile("xc")
    beyond = ardennes_graph.features.tech > timid.tech_max + TECH_CEILING_MARGIN
    if not beyond.any():
        pytest.skip("fixture has nothing above the ceiling")

    costs = materialise(ardennes_graph, timid)

    blocked = costs.allowed[np.isin(ardennes_graph.arc_edge, np.flatnonzero(beyond))]
    assert not blocked.any()


def test_a_generous_ceiling_blocks_nothing_extra(ardennes_graph: GraphStore):
    bold = get_profile("tech")

    costs = materialise(ardennes_graph, bold)

    assert int(costs.allowed.sum()) == ardennes_graph.n_allowed_arcs


def test_every_arc_is_costed(ardennes_graph: GraphStore):
    costs = materialise(ardennes_graph, get_profile("gravity"))

    assert len(costs) == ardennes_graph.n_arcs
    # An arc and its twin cover the same ground, whatever they cost to ride.
    np.testing.assert_allclose(costs.distance_m, costs.distance_m[ardennes_graph.arc_rev])
