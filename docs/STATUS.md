# Project status and handoff

Written at the end of the session that built Phases 0–7, then updated by the
session that closed the Phase 7 gaps and built Phase 8. Read this first.

The implementation plan lives outside the repo at
`C:\Users\rconven\.claude\plans\help-me-plan-a-twinkly-hamming.md`. It holds the
full design rationale and the phase table; this file records what is actually
built, what is measured, and what is not done.

## Where things stand

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton, tooling, quality gate | Done, verified |
| 1 | Ingest → graph → A* → GPX | Done, verified on real data |
| 2 | Contraction, `.npy` artifact, manifest gate | Done, verified |
| 3 | Difficulty inference, propagation | Done, verified |
| 4 | Elevation: DEM, smoothing, grade histograms | Done, verified |
| 5 | Speed model, ride profiles, cost vector | Done, verified |
| 6 | Numba kernels, Router facade | Done, verified |
| 7 | FastAPI endpoints | Done, verified |
| 8 | Loop generator | Done, gate met on real data |
| 9 | Frontend: MapLibre, control rail, elevation profile | Done |
| 9b | Trail overlay, A-to-B UI, keyboard chart, WCAG AA | Done |
| 10 | Polish: hillshade from DEM, vector tiles, Lighthouse | Not started |
| 11 | Trail running: shared engine, `trail` sport | Done, verified on fixtures; never built on real data |

**Nothing is committed.** `git init` has run; the tree is untracked. Whether
`.claude/` belongs in git is undecided.

Quality gate at handoff: ruff, black, isort, mypy (86 source files), bandit all
clean. **942 tests pass, 2 skipped** — 883 Python, 59 JavaScript. Coverage ≥ 80%
enforced, currently 88%.

The suite now spans three runners. `uv run pytest` covers everything including a
wrapper that shells out to Node; `node --test tests/js/*.test.mjs` runs the
front-end logic alone; `-m e2e` is the browser pass.

### Phase 8 gate, measured

The plan's criteria and what 30 sampled starts across Luxembourg actually gave
(`tests/integration/test_loop_gate.py`, marked `slow`, skipped without a built
artifact):

| Criterion | Required | Measured |
|---|---|---|
| Requests returning ≥3 loops within ±10% | ≥90% | **100%** (30/30) |
| Overlap per loop | <25% | **max 24.3%** |
| Latency p95 | <2 s | **399 ms** (median 215 ms, worst 582 ms) |
| Failed requests | — | **none** |

Per budget: 45 min median 114 ms, 90 min 215 ms, 150 min 332 ms. All ten
requests at each budget returned three loops on target.

## Start here

**Phase 10, polish.** Four of the five concerns raised at the end of Phase 9 are
closed — see "What the follow-up work closed" below. What remains, in order of
how much it changes the product:

1. **Hillshade and contours from the project DEM.** The four GLO-30 tiles are on
   disk and unused by the interface. An OSM raster basemap now gives orientation
   — towns, water, road numbers — but not relief, and a route that climbs 600 m
   still looks the same as one that does not. `api/routes/style.py` is where a
   `raster-dem` source goes; the basemap layer shows the shape to follow.
   **This is the biggest remaining gap.**
2. **The trail overlay is GeoJSON, not vector tiles.** It works and it is cached,
   but a dense view is about 1.1 MB and refetches on pan. Read the measurements
   below before deciding it matters: served locally it costs tens of
   milliseconds.
3. **Loops are 49–71% paved on real data**, even on the `tech` profile. The UI
   shows the fraction, so the honesty is covered, but the profile weights still
   deserve one tuning pass.
4. **No Lighthouse run and no screen-reader pass.** Text contrast is measured and
   enforced and the chart is keyboard-operable, but nothing has been profiled for
   performance, and the map canvas is unavoidably visual.

### What the earlier "Start here" items became

1. **Endpoint tests — done.** `tests/unit/test_route_cache.py`,
   `tests/unit/test_planning_schemas.py`, `tests/unit/test_budget.py` and
   `tests/integration/test_api_plan.py`. Writing them found two real defects,
   both fixed: an unknown profile id returned **500** (`BuildError` was
   unmapped — now `UnknownProfileError` → 422 `unknown_profile`), and any
   unmatched path under `/api/routes/` was answered as an **expired route**,
   so `GET /api/routes/loop` would have told a rider to re-plan a route that
   never existed.
2. **Anytime budget — wired.** See the decisions section; the approach is not
   the obvious one and the reason matters.
3. **`plan.py` split — done.** Now `api/deps.py` (process-wide cache, slots,
   profile/rider resolution), `api/concurrency.py` (`PlanSlots`), and
   `api/routes/{profiles,point_to_point,planned,loop}.py`. The tests were
   written against the HTTP boundary *before* the split, so they prove it
   preserved behaviour. The concurrency guard is now a FastAPI dependency, which
   is what made it testable — `dependency_overrides[plan_slots]` instead of
   reaching into another module's private semaphore.

## Running it

```powershell
uv sync --all-groups          # Python 3.13 is pinned
uv run mtb --help

# Data already present under data/ (gitignored):
#   data/osm/luxembourg-latest.osm.pbf   47 MB
#   data/dem/*.tif                       4 GLO-30 tiles, ~150 MB
#   data/graph/lux-v1/                   built artifact, 40 MB

uv run mtb build --pbf data\osm\luxembourg-latest.osm.pbf --artifact lux-v1
uv run mtb inspect data\graph\lux-v1
uv run mtb route --artifact data\graph\lux-v1 --profile tech `
  --from "49.6116,6.1319" --to "49.7920,6.2870" --out out\ride.gpx

$env:MTB_ARTIFACT = "lux-v1"; uv run mtb serve      # then /docs
```

Planning a loop over HTTP:

```powershell
$body = @{ start=@{lat=49.6116; lon=6.1319}; profile='tech'
           budget_minutes=120; variants=3; seed=42 } | ConvertTo-Json
$r = Invoke-RestMethod -Method Post -ContentType 'application/json' `
       -Uri http://127.0.0.1:8000/api/routes/loop -Body $body
$r.data.loops | Select-Object -Property @{n='km';e={$_.stats.distance_km}},
    @{n='min';e={$_.stats.moving_time_min}}, overlap_ratio, within_budget, score
Invoke-WebRequest "http://127.0.0.1:8000$($r.data.loops[0].gpx_url)" -OutFile .\out\loop.gpx
```

The interface, and the front-end test suites:

```powershell
$env:MTB_ARTIFACT = "lux-v1"; uv run mtb serve
Start-Process "http://127.0.0.1:8000/?start=49.6116,6.1319&profile=tech&min=120"

node --test tests\js\pure.test.mjs tests\js\route.test.mjs   # 59, no npm needed
uv run pytest tests\e2e -m e2e                               # real Chromium, 34
uv run playwright install chromium                            # once, if missing
```

Full quality gate:

```powershell
uv run ruff check src tests; uv run black --check src tests
uv run isort --check-only src tests; uv run mypy src; uv run bandit -r src -q
uv run pytest --cov=src/mtb_planner --cov-report=term-missing --cov-fail-under=80
```

The full Benelux + Ardennes AOI (`config/aoi.toml`, 7 extracts, ~2.5 GB) has
**never been built**. Luxembourg is the only real data exercised. That is the
outstanding scale risk.

### Phase 9 gate, assessed

The gate is *"looks like the design direction, not a template"* — a judgement,
not a measurement, so here is the evidence rather than a verdict. Screenshots
were taken at 1440, 1024, 768, 375 and 320 with a real Luxembourg loop on screen.

What is demonstrably there: full-bleed map with a floating rail and dock at
deliberately different widths, insets, radii and rhythm; warm paper with ~3%
turbulence grain; hairline rules; the readout as the loudest thing on the page at
~2.5rem in tabular mono with no headline competing; one oklch difficulty ramp
used identically on the route line, the chips and the profile; dashed line and
hatched coverage bar for inferred difficulty; a segmented profile picker rather
than a select; and the linked hover working in both directions.

What a designer would still object to at the time: the map was empty apart from
the route, which left a large dead area between the panels. **That is now
fixed** — the trail overlay fills it with the real network, and the screen reads
as the survey plate it was meant to be. What is still missing is terrain: no
hillshade and no contours, so the map is flat in a tool whose whole subject is
climbing and descending.

## Two sports, one engine

The bike-only `mtb_planner` package was split in two: `planner_core` holds
everything that does not care what is being planned for, and each sport is a
small package that hands the engine one `planner_core.sport.Sport`. `mtb_planner`
is the bike, unchanged in behaviour; `trail` is new.

**What a sport supplies.** An access function (`Access.NO / WALK / YES`) and a
direction function; a difficulty function onto the shared 0–6 ordinal; which
route relations it reads and which count as signed; an athlete whose
`speed_table()` is the whole time model; an effort control; a profiles file; a
settings class with its own env prefix; and the words the UI and CLI use.
Everything else — ingest passes, graph build, propagation, elevation, cost
vector, kernels, loops, slack, API, interface — is shared.

**One sport per process.** The entry point (`mtb`, `trail`, or
`create_app(sport)`) calls `activate()`, and code that needs the sport reads
`active_sport()`. Engine functions a test may want under a different sport —
`read_extracts`, `build_graph`, `compute_features`, the manifest checks — also
take it as an optional argument. The test suite activates the bike at import and
puts it back after every test.

**Verified that the bike did not move.** `materialise` was run on `be-v1` for all
four bike profiles with the pre-split code (`main` at `3593903`) and with the
split code: cost, time, preference and allowed arrays hash identically across
2,065,825 edges. `be-v1` loads without a rebuild — the manifest's new `sport`
field defaults to `mtb`, and the renamed `n_signed_routes` column is still read
from `feat_n_mtb_routes.npy`.

**The runner's model**, in `src/trail/`:

- *Access* follows `foot=*`. `bicycle=no` is irrelevant; `foot=no`,
  `foot=use_sidepath` and private-without-`foot` close a way; `smoothness=
  impassable` does not. Steps are `YES`. `demanding_alpine_hiking` and above
  are `WALK`, timed at a 0.8 m/s scramble. Only `oneway:foot` sets a direction.
- *Difficulty* reads `sac_scale` as the survey (hiking 1 … difficult_alpine 6),
  `mtb:scale` as a proxy compressed at the top (S4 → 3.2), and otherwise a
  composite where trail visibility weighs more and width and smoothness less
  than on the bike. Uphill is the downhill figure minus 0.3. Flow is zero.
- *Speed* is Minetti et al. (2002), `C(i)` in J/kg/m, divided into a metabolic
  power calibrated from the runner's flat pace. Surface multiplies the cost
  (sand ×1.35), each technicality grade adds 12%, and descents are capped by
  technicality and surface as on the bike (5.0 m/s at T0 down to 0.8 m/s at T6).
  At the default 6:00/km: 10.0 km/h on flat tarmac, 8.3 on a T1 dirt path,
  5.0 at +10%, 2.4 at +30%.
- *Signed routes* are `hiking`, `foot`, `running`, `fitness_trail`.
- *Profiles* `easy · trail · technical · vertical`, λ 0.9 / 1.4 / 2.0 / 2.4 —
  lower than the bike's because firm trail is only about 1.2× slower than
  tarmac on foot, against 1.5× on a bike.

**Separate graphs, shared downloads.** `TRAIL_DATA_DIR` defaults to the same
`data/`, so extracts and DEM tiles are fetched once. Built graphs go to
`data/graph-trail/` (`TRAIL_GRAPH_SUBDIR`), because the AOI config names one
artifact and a trail build would otherwise overwrite the bike's. The trail
server defaults to port 8001.

**Verified live.** `mtb serve` on `be-v1` and `trail serve` on a fixture
artifact built with `trail build` ran side by side: the bike planned La Roche →
Houffalize under `flow` with a three-hour budget (26.3 km, 168 min), the runner
planned 30-minute loops, and each page rendered its own wording and grade prefix.

## Why A-to-B came back on the road

Reported from the Belgian build: point-to-point routes looked like a road
router's, and neither the profile nor the time slider seemed to change them.
Three separate causes, measured on `be-v1` (1.6M nodes / 2.07M edges), all now
closed.

**The time slider was not wired to anything.** `PlanRequest` had no time field
at all, and `api.js` sent `{start, end, profile}`. `renderStart()` hid the
variants field in A-to-B and left Time visible next to a comment saying the
budget "becomes slack to spend rather than a target to hit" — an intent nothing
implemented. It is implemented now; see `routing/slack.py`.

**The penalty was spending itself on the speed difference.** `cost = t·(1 + λ(1
− p_norm))` multiplies *time*, and tarmac is about 1.5x faster per metre than
singletrack in the speed model (0.263 s/m against 0.399). So the first stretch
of every λ goes on cancelling that out before it buys any trail. Effective cost
per metre over the whole graph, before:

| profile | λ | paved | singletrack | trail wins by |
|---|---|---|---|---|
| xc | 0.8 | 0.388 | 0.405 | *nothing — the road was cheaper* |
| flow | 1.2 | 0.528 | 0.419 | 26% |
| tech | 1.5 | 0.615 | 0.429 | 43% |
| gravity | 2.0 | 0.711 | 0.516 | 38% |

Under `xc` the planner preferred the road unless the trail was outright
shorter, and `flow` would only accept a trail up to a quarter longer — less than
linking forest trails usually costs. λ is now 1.2 / 1.9 / 2.6 / 3.4, chosen
from those measurements to give roughly 30% / 64% / 97% / 75% tolerance.

**The technicality band was paying tarmac a bonus.** The band scores how close
an edge sits to the difficulty the rider wants, and a sealed road scores tech 0.
Under `xc` — target 1.0, sigma 0.8 — that put a road at 0.48 of a perfect match,
worth **+0.71** against a paved weight of only −0.60. Decomposed, a paved edge
came to −0.23 where singletrack came to +2.11. The band is now gated on unpaved,
which is the change that flipped `xc`: paved cost per metre 0.388 → 0.441 with
λ untouched. A smooth road is not a mild version of a trail.

Measured effect, La Roche → Houffalize, 23 km through the Ardennes:

| profile | paved before | paved after | singletrack before | after |
|---|---|---|---|---|
| xc | 69.2% | 34.2% | 18.8% | 49.1% |
| tech | 44.6% | 26.5% | 46.7% | 63.9% |

**Slack is spent by bisecting λ, not by a second search.** A budget on a
point-to-point route is slack, not a target: the destination is fixed, so the
only question is what to do with the time left over after the quickest way
there. Raising λ raises both preference and elapsed time, so "the best route
inside this budget" becomes "the largest λ whose route still fits" — one scalar,
found by bisection, instead of a resource-constrained shortest path, which is
NP-hard. `ArcCosts.with_detour` re-costs an already-materialised vector (~15 ms)
rather than solving the physics again (~500 ms), so six probes cost about 20 ms
on top of a 9 ms search. Same pair under `flow`: 131.7 min and 44.6% paved with
no budget; 167.0 min and 24.6% paved given three hours.

A budget the ride cannot meet is **reported, not enforced**: the rider gets the
route to where they asked to go, `within_budget` is false, and a warning names
the real duration. Trimming the route to fit would mean not arriving.

Loop generation was checked for a regression, since `plan_loops` bounds its
Dijkstra by `(1 + λ)·tree_time`. There is none: 90-minute `flow` went 7,389 →
6,769 settled nodes, and wall clock is unchanged at roughly 300–350 ms.

## What the follow-up work closed

Four of the five Phase 9 concerns, each with tests.

**The map is no longer empty.** `graph/viewport.py` and `GET /api/network` serve
the trail network out of the same graph the router uses, so what a rider sees is
exactly what can be routed over, with no second extraction path to drift. Paths
and tracks are drawn properly, roads drop to hairlines, signed routes get their
own weight. Three things keep it affordable: a vectorised bounding-box test on
node coordinates, per-class zoom thresholds, and dropping any edge shorter than
four pixels of ground at the current zoom, which cut candidates from 27,185 to
8,310 on one dense Luxembourg view. It is emphatically **not** a basemap — no
towns, no labels, no land cover — but it turns a blank plate into a legible one.

**Truncation is visible.** A dense view exceeds the 6,000-feature cap, and the
interface now says so: "Showing the 6,000 longest of 8,310 trails in view. Zoom
in for the rest." A silently thinned map reads as an area with very few trails,
which is the wrong thing to make a rider believe.

**Point-to-point has a UI.** A Loop / A-to-B control in the rail. In A-to-B the
first click sets the start and the second the finish, then it alternates, so
placing a pin never needs a mode switch as well. The variants slider hides,
because a point-to-point route has one answer and there is nothing to rank. Time
stays, and its hint changes to say what it means there — slack to spend on
better ground rather than a target to hit.

**A loop survives a re-read.** `CachedRoute` now carries the metrics, score and
budget verdict, and `RouteModel` gained those fields as optionals — null for a
point-to-point route, which genuinely has none. A shared link no longer comes
back stripped of the character and overlap that were shown when it was planned.

**Text contrast is measured, not asserted.** `--ink-faint` carries every label
and hint at 11 to 13 px and measured **3.53:1** light and **3.41:1** dark against
the panel, below AA. Now 47% and 76% lightness, and a test computes the WCAG
ratio from the oklch tokens for all three ink tones in both themes. That test is
how the failure was found in the first place.

**The elevation chart is keyboard-operable.** It was pointer-only, and it is the
only way to inspect a point on a route. Now `role="slider"` with arrows stepping,
page keys jumping, home and end at the ends, and `aria-valuetext` announcing the
readout. The map marker follows the keyboard exactly as it follows the pointer.

## What Phase 9 built

`GET /` serves one Jinja page; everything after that is the same JSON API any
other client would use. No bundler, no npm, no framework — seven ES modules
totalling about 1,100 lines, plus vendored MapLibre.

```
src/planner_core/web/
  templates/app.html        semantic shell; the script fills values, not markup
  static/css/tokens.css     the ONLY file that names a colour
  static/css/app.css        layout; asymmetric rail and dock
  static/css/fonts.css      self-hosted Inter Tight + IBM Plex Mono
  static/js/ramp.js         difficulty ramp, oklch interpolation -> sRGB
  static/js/route.js        payload -> map features, distance <-> position
  static/js/urlstate.js     the URL is the state
  static/js/format.js       every figure a rider reads
  static/js/api.js          one envelope unwrapper
  static/js/profile.js      the elevation chart and half the linked hover
  static/js/map.js          MapLibre, route layers, the other half
  static/js/app.js          wiring
  static/style/paper.json   the paper base; Phase 10 adds sources to it
  static/vendor/            MapLibre 5.6.0 + 4 woff2, SRI-hashed, committed
```

The API grew two fields to make the design possible, both derived from data
already in the artifact (`routing/segments.py`):

- **`segments[]`** — per edge: `first`/`last` index into the geometry, `at_m`,
  `tech` in the direction ridden, and `surveyed`. Indices rather than
  coordinates, so difficulty costs a few numbers per edge instead of a second
  copy of the polyline.
- **`elevation[]`** — `at_m` and `height_m` at every segment boundary.

## How a loop request flows

Worth reading once before touching `routing/loop.py`; each step is a module and
each is separately tested.

```
snap start ─► ball.py            circle of 0.5·budget·v_max, as index masks
           ─► costvector.py      materialise restricted to that ball
           ─► router.py          ONE bounded Dijkstra to 0.65·budget  ─► Frontier
           ─► loop.py            best pivot in each of 12 heading sectors,
                                 ranked by cost/time (= preference density)
           ─► penalties.py       return leg: outbound arcs ×4, U-turn arc removed
           ─► loop.py            recalibrate f_pivot from the median achieved
                                 time, re-select from the SAME tree (free)
           ─► repair.py          shorten (replace worst stretch) / lengthen
                                 (hang a lasso detour), ≤3 rounds
           ─► candidates.py      measure, reject on overlap and closure,
                                 score, then MMR for diversity
```

Two searches per candidate at most, one Dijkstra total, and the calibration
round is free because the tree is grown wider than any `f_pivot` will ask for.

## The basemap

`GET /api/style` assembles the map style per request: `paper.json` holds the
design, and the endpoint injects whatever the deployment is configured to put
under it. Requested directly, because without it a rider could see the shape of a
route but not which valley it was in.

Three things are deliberate.

**It is a setting, not a dependency.** `MTB_BASEMAP_URL` empty gives the paper
plate the rest of the design assumes, and the content security policy tightens
back to `self` for everything. Everything except the tiles stays local.

**The origin reaches the policy by derivation, not by being written twice.**
`Settings.basemap_host` parses the configured URL, and `content_security_policy`
takes it as an argument. A policy naming a different host from the one actually
fetched blocks every tile and leaves a blank map with only a console error.

**It appears in `connect-src` as well as `img-src`.** MapLibre pulls raster tiles
through the Fetch API, not as `<img>` elements. `img-src` alone was my first
attempt and it silently blocked all 45 tiles: no Python error, no failed request,
just an empty map. Found by watching the browser console, not by reasoning.

The basemap is drawn at 55% opacity and −0.6 saturation so the difficulty ramp
stays the only saturated thing on screen, and the trail overlay steps back when
one is present: roads hide entirely, because OSM already draws them better, and
the trails fade to emphasis rather than a second set of lines over the first.

## Bugs the first real end-to-end run exposed

Somebody ran the documented commands as a script against a clean machine. Five
things came out of it, in rough order of severity.

**`europe/benelux` is not a Geofabrik region.** The area of interest is named
after Benelux and its primary extract never existed. Geofabrik answers an unknown
path with its download index page and a **200**, so nothing raised: a 9.6 KB HTML
document was written as `benelux-latest.osm.pbf`, reported as "already present
and verified" on every later run, and a full AOI build would have contained no
Belgium and no Netherlands with nothing to say so. Now three separate extracts,
merged at ingest like every other border. **Two tests asserted the fictional
slug** and so agreed with the bug for the life of the project; they encoded the
intention in the plan and nothing ever checked that the path resolved.

**A download was called "verified" when nothing was verified.** Geofabrik does
not publish a checksum for every region, and the check read
`expected is None or md5 matches` — so with no sidecar any existing file passed,
including an empty one. Now: a size floor, a check that the bytes really are a
PBF (`OSMHeader` in the first blob header), and a log line that says which of
those actually happened.

**A missing DEM tile aborted the whole fetch.** GLO-30 publishes land only, so a
rectangular area of interest asks for tiles over the Channel and the North Sea
that were never made. One 404 raised `typer.Exit(1)` and left the remaining tiles
undownloaded. Absent tiles are now counted and reported as the expected thing
they are.

**Nothing resolved the artifact sensibly.** `mtb serve` had no `--artifact` at
all, so the only way to serve a built graph was to remember `MTB_ARTIFACT` —
four separate sessions ended with a server that started, served the interface,
listed the profiles and answered every plan with 503. Adding the flag was not
enough: the configured name defaults to `aoi-v1`, which nobody builds, and
refusing to serve the one graph on the machine because it is called something
else is pedantry rather than safety.

`graph.manifest.resolve_artifact_dir` now states the rule once, and server
startup, `mtb route` and the `serve` pre-flight all use it: **use the configured
artifact if it is built; otherwise fall back to the only one present; never
choose between several.** The last clause is the one that matters — silently
picking one of two is how somebody plans against last month's graph without
noticing — and it is tested directly.

That change also broke three tests, correctly: "no graph is loaded" had quietly
come to mean "unless the developer happens to have built one". The autouse
`_isolated_data_dir` fixture now points every test at an empty data directory, so
the degraded-server paths are deterministic instead of depending on what is on
disk.

**The README CLI block was a runnable script containing alternatives.** Listing
`fetch osm europe/luxembourg` beside a bare `fetch osm` reads as "either" to a
person and "both" to a script: one extract followed by all seven, four DEM tiles
followed by forty-two. Alternatives now live in prose, and a test asserts no
single block offers two forms of the same command.

### What this says about the tests

Every one of these passed the suite. The pattern was the same each time: the
tests asserted the *shape* of something rather than exercising it. There are now
tests for fictional flags, placeholder values, hidden configuration
dependencies, competing commands in one block, non-PBF downloads, and whether
the AOI actually names the countries it claims to cover — but the durable lesson
is that documentation and configuration need running, not reading.

## Decisions that must not be quietly undone

**An artifact belongs to one sport.** The manifest records `sport`, and loading
under another refuses with "built for trail, not mtb". A runner's graph holds
footpaths a rider may not legally use, and a rider's holds bike-only cycleways;
serving either to the other is wrong in a way no route looks wrong.

**Sport-specific logic stays in the sport package.** If `planner_core` ever
needs an `if sport.id == ...`, the `Sport` interface is missing a field. Add the
field; do not branch.

**The separation invariant.** `cost = time × (1 + λ·(1 − preference))`, with
time, distance and preference kept independent. Three properties depend on it:
the A* heuristic stays admissible, a budget-limited search has a provable
bound, and reported figures are measured rather than inferred from the thing
that drove the routing. Never switch to `cost = time + penalty`; additive
penalties break all three. Tests in `tests/unit/test_cost.py` pin this.

**Difficulty is a property of trail, not of every surface.** The technicality
band match is gated on unpaved. Ungated, a sealed road scores tech 0, and any
profile with a low target pays it a bonus for being close — which is how the XC
profile came to prefer the road. Re-introducing the ungated band would put it
straight back.

**λ is the only thing the slack search may move.** `ArcCosts.with_detour`
re-costs an existing vector; time, distance and preference must come back
untouched, or the figures shown to the rider stop matching the search that chose
the route. `tests/unit/test_slack.py` pins that, and pins that no budget between
the quickest route and the greediest one ever returns a route over budget.

**Legal exclusions are hard.** `bicycle=no` and `access=private|no` remove ways
from the graph. They are never soft penalties. `access=forestry` is *not* a
bicycle ban — excluding it would delete most of the Wallonian forest network.

**Elevation corrections are not optional.** Smoothing, the 0.8 m dead-band and
bridge/tunnel interpolation together remove ~20% of raw ascent (measured: 661 m
raw → 531 m corrected on one 26 km route). Without them the planner overstates
climbing on every route.

**Difficulty carries confidence.** `tech_conf` distinguishes surveyed from
estimated. At 3.8% surveyed coverage, presenting inference as survey would be
dishonest. Keep it visible in any UI.

**One router.** All searching goes through `routing/router.py`. The loop
generator must use `Router.reachable_within` rather than growing its own
search with its own subtly different U-turn rule.

**Version gates.** `ARTIFACT_SCHEMA_VERSION` (3) lives in
`planner_core/__init__.py`; each sport has its own `COST_MODEL_VERSION`, in
`mtb_planner/__init__.py` (3) and `trail/__init__.py` (1). Bump the schema when
the artifact layout changes and a sport's cost model when its interpretation
changes; a stale artifact then refuses to load
instead of silently producing plausible-looking wrong routes.

**The timeout is enforced outside the kernel, by deepening.** Numba's nopython
mode cannot read a clock — `time.perf_counter()` inside `@njit` is a
`TypingError`, verified, not assumed. So `MTB_ROUTE_TIMEOUT_S` cannot be checked
mid-search. Instead `Router.shortest_within` runs the search under a settle
ceiling and multiplies the ceiling by 8 between rounds for as long as the
`Deadline` allows. The wasted work is bounded: every round before the last costs
about a seventh of the last, and a search that fits the first ceiling
(`INITIAL_SETTLE_CEILING = 250_000`) pays nothing at all — a real 29 km
Luxembourg route settles 79,853, so the common case is one round.

The rejected alternative was converting seconds into a settle count with a
measured throughput constant. That constant is machine-specific and wrong in
the dangerous direction on hardware slower than the one it was measured on.
Do not reintroduce it.

**MapLibre cannot parse `oklch()`, and fails silently when given it.** The whole
palette is oklch, for good reason — the difficulty ramp has to be perceptually
even. But a paint value in oklch makes MapLibre reject the layer, and the only
symptom is a map with no route on it: no Python error, no failed request, no
exception in the page. It cost most of an afternoon. So `ramp.js` interpolates in
oklch and converts to sRGB as the last step before handing anything to the
canvas, and `readToken()` converts on the way out of CSS. Two tests guard it —
one asserting `paper.json` contains no oklch, one asserting the browser console
stays clean. **Do not "simplify" by passing a token straight to a paint
property.**

**The elevation profile is relative height, not altitude.** The artifact stores
per-edge ascent and descent but no absolute height, so there is no datum. The
chart is labelled "height gained from the start" for that reason. It reconciles
with the reported ascent by construction, because it is cumulated from the same
per-edge figures, and a test asserts they agree. Adding real altitude means
storing per-edge endpoint heights: a schema bump.

**Colour is declared once, in `tokens.css`.** A test walks the other stylesheets
and fails on any hex or `oklch(` literal. The duplication that does exist — the
ramp stops in `ramp.js` — is there because a canvas cannot read a custom
property, and a test asserts the two copies agree stop by stop.

**The route is one feature per edge, not one line with a gradient.** MapLibre
cannot vary a dash pattern along a single feature, and the dash is what marks
difficulty that was inferred rather than surveyed. Two layers over one source,
filtered on `surveyed`. The draw-in animation is a filter on `at_m` rather than
an opacity fade, so the line grows the way a ride is ridden.

**The loop generator has no RNG, and needs none.** The plan called for an
injected `np.random.Generator` to make a stochastic generator reproducible. It
turned out the construction is deterministic by design: sector winners come from
`argmin` (first minimum, so ties fall to the lower node index), calibration is
arithmetic, repair picks insertion points at fixed fractions along the route, and
MMR sorts stably on raw scores. Reproducibility is therefore structural rather
than seeded. `seed` is still accepted and echoed on the loop endpoint — the API
contract promised it, and a client may send it — and the schema says plainly that
it changes nothing. **If randomness is ever introduced, inject the generator;
never reach for module-level `random`.**

**MMR compares raw scores, not rescaled ones.** Min-max normalising relevance
across the candidate set is the textbook form and it is wrong at this scale: over
a handful of candidates it stretches whatever spread exists onto the full range,
so a hundredth of a point becomes decisive and near-duplicates always win.
`score()` already returns a roughly [0, 1] quantity — the same scale as the
similarity it trades against. `tests/unit/test_candidates.py` pins this with a
near-duplicate that must lose to a worse but distinct route.

**Pivots are ranked by `cost / time`, not by reconstructing their paths.** Since
`cost = time · (1 + λ·(1 − p_norm))`, that ratio *is* the length-weighted
preference along the path the Dijkstra already found. Reconstructing thousands of
paths to rank them would cost more than the search that produced them. This is
also the reason the pivot search must never be given a cost vector whose cost is
anything other than the standard one.

**The search ball is a mask, never a renumbered subgraph.** `routing/ball.py`
returns boolean masks over the graph's own node and arc indices. Renumbering
would mean translating every index in and out, and one missed translation puts a
route subtly in the wrong place rather than raising. An arc is inside only when
both endpoints are, which is what makes a confined search unable to leave.

**A penalty moves `cost` and nothing else.** `routing/penalties.py` multiplies
cost on already-ridden arcs *and their reverse twins*; `time_s`, `distance_m` and
`preference` are passed through untouched. If a penalty ever reached `time_s`,
every duration reported for a loop would be inflated by however much backtracking
the generator happened to discourage. Tests assert this directly.

**The first calibration round always completes, deadline or not.** A dozen A*
searches is the smallest unit of work that can produce a loop at all, so a server
that has already spent its budget queueing answers with a slightly worse route
rather than an error. Later rounds are what the deadline cuts — which is the
anytime behaviour the plan asked for, and the place where "best-so-far" genuinely
exists.

**A truncated point-to-point search is a failure, not a partial route.**
`NoRouteFoundError(reason="search_timeout")`, distinct from `"disconnected"`,
because they need different actions: retry versus move the pin. A path that
stops short of the destination is not a slightly worse route — it leaves the
rider somewhere they did not ask to be. `SearchResult.truncated` is what
separates "ran out of budget" from "no such route"; the kernel settles exactly
one node past its ceiling before breaking, which is how that is detected.

## Measured facts, so they need not be rediscovered

From real OSM data (Luxembourg, 47 MB extract → 164,650 nodes / 204,175 edges):

- **`mtb:scale` covers 5.08% of trail ways** (Belgium taginfo: 3.4%). Inference
  is the primary engine, not a fallback.
- **`mtb:type=downhill` appears on 11 ways in all of Belgium.** Bike-park
  detection from OSM is impossible here; the `gravity` profile is
  terrain-derived instead, and cannot tell a real park from a steep forest
  trail.
- **Contraction removes only 3.2%** — because 15,991 of 22,785 mergeable
  degree-2 nodes carry a barrier (gates on forest tracks), and a gate must stay
  a node. Not a bug. More is available once the merge key can loosen from
  identical-tags to feature-equality.
- **Trail-group propagation by name/route is nearly worthless here**: only 111
  of 4,590 surveyed segments also carry a name. Corridor adjacency is what
  works, lifting propagation from 31 to 1,180 edges.
- **Difficulty coverage by length**: 3.8% surveyed, 1.0% proxy, 84.5%
  estimated, 10.7% default.
- **Reachability**: 98.0% of edges lie in the giant strongly connected
  component.
- **Search latency**: compiled kernels give median 14 ms / p95 32 ms on random
  long-range pairs, 12–14× the pure-Python reference. Twelve searches — one
  loop request — cost 0.39 s.
- **A full HTTP plan request** (Luxembourg, 29.2 km / 612 m / 137 min on the
  `tech` profile) takes **182 ms end to end** and settles 79,853 nodes.
- **A full HTTP loop request** (same start, 120 min, `tech`, 3 variants) takes
  **384 ms** and returns 22.8 / 24.0 / 26.5 km loops at 120.7 / 124.5 / 122.8
  minutes, 0–2% overlap.
- **`materialise` costs 119 ms over the whole Luxembourg graph** (204k edges).
  Restricted to a 60-minute ball it is 61 ms; a 120-minute ball on Luxembourg is
  83% of the graph, so the restriction barely helps *there*. It matters for the
  full AOI, which is ~27× larger — 3 s unrestricted, which alone would blow the
  2 s gate.
- **Real loops are 49–71% paved**, even on the `tech` profile, with technicality
  0.31–0.52 and 31–37% singletrack. Luxembourg's trail pockets are joined by
  roads, so a two-hour loop has to use them. The scoring does penalise paved;
  this is what the data supports, not a scoring bug. Worth one tuning pass before
  the UI shows a surface breakdown.
- **The trail overlay**, on one dense Luxembourg view of 0.15 by 0.10 degrees:
  at zoom 13, 8,310 edges qualify and the 6,000 cap yields **1,147 KB**; zoom 11
  gives 10,902 qualifying and 1,338 KB; zoom 15 gives 7,791 and 1,136 KB. The
  four-pixel length floor brought the zoom-13 candidate count down from 27,185.
  Payload is dominated by feature count, not coordinate precision: going from six
  decimals to five saved under 7%.
- **Text contrast against `--paper-raised`**: `--ink` 16.8:1, `--ink-muted`
  6.9:1, `--ink-faint` 4.5:1 after correction and 3.5:1 before. Dark theme:
  12.2:1, 6.3:1, and 4.5:1 after correction and 3.4:1 before.
- **The search ball is generous.** `max_speed_mps(Rider())` is 11.2 m/s (40 km/h)
  from the power model on a favourable downgrade, so a 120-minute budget gives a
  40.5 km radius. Generous is the safe direction — it can never exclude a
  feasible loop — but it is not a tight bound.

## Known gaps and risks

- **The trail model has never met real data.** No trail artifact has been built
  from a real extract, and none of its constants — Minetti calibration aside —
  is measured. The λ values were reasoned from the speed model, not measured on
  a graph the way the bike's were. First job: `trail build` Belgium and repeat
  the cost-per-metre table in "Why A-to-B came back on the road" for runners.
- **Belgian foot access on cycleways is a guess.** `highway=cycleway` is open on
  foot unless tagged `foot=no`, because RAVeL greenways are walked as much as
  ridden. Dutch and Flemish fietspaden are often closed to walkers where a
  footway exists, and OSM rarely says so.
- **The browser tests have not run against the trail UI.** The e2e suite is
  bike-only and needs `playwright install chromium`; the trail page is checked
  over HTTP only.
- **Effort is not in the interface.** The API takes `effort` (watts or min/km)
  and lists it as a slider, but the rail does not render it for either sport.

- **Signed-route seeding is not built.** The plan's Phase 8 step 1 — seed
  candidates from closed `route=mtb` relations near the start — is the one part
  of the pipeline that is missing, and the plan rates it the best
  quality-per-line in the project. **It is blocked on the artifact:** edges carry
  only `n_signed_routes` (a count) and `route_flags` (a bit), so relation *identity*
  is gone by the time the graph is built and a specific signed loop cannot be
  reconstructed. Doing it needs a per-edge relation id plus a relation table,
  which is a `ARTIFACT_SCHEMA_VERSION` bump and a rebuild. Partial consolation:
  the `signed_mtb` feature already biases the cost function towards signed edges,
  so signed trails are *preferred* — they are just not used as ready-made loops.
- **The slack search saturates at the λ ceiling.** Above the budget that
  affords `MAX_DETOUR = 6.0`, more time buys nothing: the planner has already
  taken all the trail it wants, and every larger budget returns the same route.
  Defensible — unspent slack is slack — but it means the slider goes quiet at
  the top of its range. Below that, five bisection steps locate λ to about 3% of
  the range, so a little of the budget is routinely left unspent.
- **Nothing calibrates λ against what a rider would actually accept.** The four
  values are set from measured cost per metre so each profile tolerates roughly
  the detour its description promises. That the promise is the right one is a
  judgement, not a measurement, and the GPX calibration harness would inform it.
- **A budget the area cannot fill returns an error, not the longest loop
  available.** `_pivot_sectors` looks for turning points in a band around
  `f_pivot · budget`; if the whole reachable network is smaller than that band,
  no pivots exist and the request fails with `no_loop_found`. A rider in a small
  isolated trail pocket asking for two hours gets an error rather than "here is
  the best 40-minute loop, flagged". The fix is a fallback to the outermost band
  the tree actually reached, guarded so a wildly impossible request still fails
  honestly. Deliberately left out to avoid shipping an untested path; the error
  message does at least name the cause.
- **No hillshade or contours.** The four GLO-30 tiles are on disk and unused by
  the interface. Terrain is not decoration in a mountain-bike tool: a route that
  climbs 600 m looks the same as one that does not without it. This is the
  largest gap now, and `static/style/paper.json` is written to receive a
  `raster-dem` source.
- **The trail overlay is not a basemap.** No towns, no labels, no water, no land
  cover. A rider can see the trail network and the road skeleton, which is enough
  to judge where a route goes, but not to recognise a village.
- **A dense trail view is about 1.1 MB.** Measured below. Cached for an hour and
  served locally, so it costs tens of milliseconds in practice, but it is far
  outside the 300 KB page budget in the plan, and vector tiles are the real
  answer.
- **The elevation profile is one point per edge, not per vertex.** The wiggle
  inside an edge is not recorded, only its net rise. On a 25 km route that is
  ~340 points, more than a chart a few hundred pixels wide can show, so it does
  not matter visually — but a short route over one long edge will look like a
  straight ramp.
- **`nearestDistance` is vertex resolution**, not a true perpendicular
  projection onto the line. Route vertices are metres apart where it matters, so
  the marker lands under the cursor; on a long straight edge the marker can sit
  noticeably off the pointer.
- **Only text contrast has been audited.** The ink tones are measured and
  enforced; nothing else has been. No screen-reader pass, no Lighthouse run, and
  the map canvas itself is unavoidably visual.
- **No calibration against real rides.** Every time estimate rests on
  literature constants: 200 W, 90 kg, CdA 0.45, a rolling-resistance table and
  descent caps. The model's *shapes* are right and monotonic, and tests pin the
  relationships — but no absolute figure is validated. The plan's GPX
  calibration harness (map-match 20–50 of your rides, least-squares fit
  `Rider`) is the highest-value remaining work for trustworthiness. Until then
  treat route *rankings* as reliable and the *clock* as indicative.
- **Full AOI never built.** Memory reworks in Phase 2 were designed for it but
  only Luxembourg has been run.
- **`routing/astar.py` is now test-only** — the oracle the compiled kernel is
  checked against. Deliberate; not on any production path.
- **Plateau alternatives** (plan, Phase 6) not implemented and not stubbed. Note
  that loops now get their variety from sector spreading and MMR instead, so this
  only matters for point-to-point.
- **`scenic` feature absent** from the vector: it needs `landuse`, which ingest
  does not retain. Adding it means a new retained tag and a schema bump.

## Working notes

- A **GateGuard hook** blocks the first write to every new path and allows the
  retry, so each new file costs two calls. `ECC_GATEGUARD=off` disables it —
  worth doing for a phase that creates many files.
- Session cost reached roughly $395; the phases are large. Budget accordingly.
- Tests use synthetic OSM fixtures built in code (`tests/fixtures/`), never
  committed binaries. `ardennes_like` is the workhorse and deliberately
  contains a private driveway, a `bicycle=no` path, a corroborated one-way
  descent, steps, an isolated fragment and a signed route relation.
- **Three test clients, and picking the wrong one wastes an hour.**
  `routing_client` publishes `ardennes_like` (11 routable nodes, 1.5 km of trail)
  — right for error paths, but **it cannot generate a loop at any budget the API
  accepts**, since the schema minimum is 15 minutes. `lattice_client` publishes
  the `lattice` network (9×9, ~60 km of trail in a 3.5 km square) and is what
  loop tests need. Plain `client` has no graph, for the 503 paths.
- The `lattice` fixture was added *because* of that: it is the only synthetic
  network big enough to exercise loop generation succeeding. Its rows and columns
  are each one way spanning the whole lattice, so splitting is still exercised,
  and its tags cycle through four surface/technicality combinations so scoring and
  diversification have something to distinguish.
- The fixture network is **seventeen nodes**, which is too small to exhaust any
  realistic ceiling. Tests of budget behaviour inject `initial_ceiling`
  directly; the one end-to-end timeout test also has to monkeypatch
  `routing.router.INITIAL_SETTLE_CEILING`. If a budget test starts passing for
  no clear reason, check that it is still actually forcing truncation.
- `tests/assertions.py` holds `assert_valid_loop` beside `assert_valid_route`.
  Use it for anything producing loops — it checks closure, the overlap ceiling,
  and that reported figures match what was measured.
- **The front end is tested by three runners, and each catches what the others
  cannot.** `tests/js/*.test.mjs` under `node --test` covers the pure logic —
  ramp, formatting, URL state, route geometry — with no npm and no bundler.
  `tests/integration/test_web_shell.py` covers the page, the assets, the SRI
  hash, the CSP and the design rules, and shells out to Node so one pytest run
  covers everything. `tests/e2e/` drives real Chromium and is the **only** place
  a MapLibre style error, a bad ES import or a broken interaction shows up.
- **`node --test tests/js` (a directory) fails on Windows** — Node treats the
  path as a CommonJS module. Pass the files: `node --test tests/js/*.test.mjs`.
  The pytest wrapper globs them for this reason.
- **The e2e tests republish the graph per test.** The shared `_reset_registry`
  fixture clears the registry around every test, so a module-scoped publish
  leaves every test after the first waiting 20 s for a plan that cannot arrive.
  If the e2e suite suddenly takes ten minutes, that is what happened.
- **Re-vendoring MapLibre changes the SRI hash**, and a stale hash means the
  bundle silently does not execute — a blank map, no error. Rewrite
  `static/vendor/manifest.json` whenever the file changes; a test compares the
  hash in the page against the bytes on disk.
- **The gate test needs the built artifact.** `tests/integration/test_loop_gate.py`
  skips itself without `data/graph/lux-v1/manifest.json`, so a clean checkout
  reports it as skipped rather than failing. Its start points are drawn from a
  fixed seed, so a failure names the same thirty places every run.
