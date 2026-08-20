# MTB Route Planner

Plans mountain bike routes around what a rider actually cares about: **how much time they have**, and **what kind of riding they want**.

General-purpose planners optimise for generic cycling and treat trail character as an afterthought — they will happily route a "mountain bike" ride down a canal towpath. This one makes trail complexity and surface character first-class inputs to the routing cost function, so a request for 90 minutes of technical singletrack returns technical singletrack.

Covers **Benelux plus the Ardennes border strips** of France and Germany, from offline OpenStreetMap extracts. Produces time-budgeted **loops** and **point-to-point** routes, viewable on a map and exportable as GPX.

## Status

Working end to end: ingest, graph build, difficulty inference, elevation, routing, loop generation, HTTP API and map interface. See `docs/STATUS.md` for what is measured, what is missing, and what to do next.

Not done: hillshade and contours in the interface, and the full Benelux area of interest has never been built — Luxembourg is the only extract exercised so far.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is pinned via `.python-version`.

Start with **one country**, not the whole area of interest. A single Geofabrik extract builds in a few minutes; the full area is seven extracts and about 2.5 GB, and has never been run.

```powershell
uv sync --all-groups

# 1. One extract. Region slugs are positional arguments; omitting them
#    fetches the whole area of interest, seven extracts and about 2.5 GB.
uv run mtb fetch osm europe/luxembourg

# 2. Elevation for the same area, as min_lon,min_lat,max_lon,max_lat.
#    Without it every route reports zero climbing.
uv run mtb fetch dem --bbox 5.7,49.4,6.6,50.2

# 3. Build a routing artifact. The name you give here is the name you serve.
uv run mtb build --pbf data\osm\luxembourg-latest.osm.pbf --artifact lux-v1

# 4. Check it. Node and edge counts, connectivity, difficulty coverage, tag stats.
uv run mtb inspect data\graph\lux-v1

# 5. Smoke test without the server: plan a route straight to a GPX file.
uv run mtb route --artifact data\graph\lux-v1 --profile tech `
  --from "49.6116,6.1319" --to "49.7920,6.2870" --out out\ride.gpx

# 6. Serve it. With one artifact built you need no flag at all;
#    --artifact only matters once several are built.
uv run mtb serve
```

Then open **<http://127.0.0.1:8000/>** and click the map to place a start.

> **If `mtb serve` fails with `[WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`**, port 8000 is reserved or in use — Windows excludes ranges for Hyper-V and WSL. Try another port: `uv run mtb serve --port 8010`, or check `netsh interface ipv4 show excludedportrange protocol=tcp`.

> **If the map loads but nothing plans**, no artifact could be resolved. The server uses the configured name (`MTB_ARTIFACT`, default `aoi-v1`) if it is built, and otherwise falls back to the only artifact present — but it will never choose between several. Pass `--artifact NAME` in that case. `http://127.0.0.1:8000/api/health` says which one is loaded.

The DEM tiles are public Cloud-Optimized GeoTIFFs on AWS Open Data, with no account and no API key, at roughly 42 MB per one-degree tile. `mtb build` picks up whatever is in `data/dem/` automatically; pass `--no-elevation` to skip elevation deliberately and accept that every route then reports zero climbing.

## Using the interface

One screen. The rail on the left sets up a ride, the dock on the right reports it.

- **Click the map** to place the start. In **A to B** mode the second click places the finish.
- **Riding** picks a profile; **Time** is the budget; **Options to offer** is how many loops to return.
- The route is coloured by difficulty on a single green-to-red-violet scale. **Dashed sections are estimated rather than surveyed** — the coverage bar under the elevation profile says how much of the route each accounts for.
- **Hover or arrow-key the elevation profile** to slide a marker along the trail, and hover the trail to move the chart cursor.
- The URL is the state: `?start=49.6116,6.1319&profile=tech&min=120` is a shareable, reproducible link.
- Underneath is an OpenStreetMap basemap, desaturated so the difficulty ramp stays the only saturated thing on screen. It is there for orientation: towns, water, road numbers.
- Over it, the trail network from the same graph the router uses, so what you see is what can be routed. Dense views are capped and the rail says so when trails are left out.

## CLI

```powershell
uv run mtb --help

uv run mtb fetch osm europe/luxembourg
uv run mtb fetch dem --bbox 5.7,49.4,6.6,50.2
uv run mtb build --pbf data\osm\luxembourg-latest.osm.pbf --artifact lux-v1
uv run mtb inspect data\graph\lux-v1
uv run mtb route --artifact data\graph\lux-v1 --profile tech `
  --from 49.6116,6.1319 --to 49.7920,6.2870 --out out\ride.gpx
uv run mtb serve --artifact lux-v1 --reload
```

### Another area

Belgium is the region the difficulty inference was designed around, and the obvious step up from Luxembourg. Same five commands, different arguments — run this block *instead of* the one above, not after it:

```powershell
uv run mtb fetch osm europe/belgium
uv run mtb fetch dem --bbox 2.5,49.4,6.5,51.6
uv run mtb build --pbf data\osm\belgium-latest.osm.pbf --artifact be-v1
uv run mtb inspect data\graph\be-v1
uv run mtb serve --artifact be-v1
```

Budget for it. The extract is **692 MB** against 47 MB for Luxembourg, so expect roughly fifteen times the nodes, edges and build time: Luxembourg is 164,830 nodes in 50 s and a 40 MB artifact, which scales to something like ten minutes and 600 MB. Those are scaled figures, not measured ones — Luxembourg is still the only extract that has actually been built. The DEM box needs 15 tiles at up to ~42 MB each, and some of them are sea and simply do not exist, which is reported and skipped.

With two artifacts on disk the server can no longer pick one for you, so `--artifact` stops being optional.

Every command in this file runs exactly as written, in this order, with nothing to substitute — so the block above is safe to run as a script. Wider alternatives are described in prose below rather than listed as commands, because a block of shell lines gets run wholesale: listing `fetch osm europe/luxembourg` next to a bare `fetch osm` means downloading one extract and then all seven.

To cover the whole area of interest instead, omit the region from `mtb fetch osm` and use `mtb fetch dem --config config\aoi.toml`; that is about 2.5 GB of extracts and 42 DEM tiles, and has never been built end to end. `mtb fetch osm` takes region slugs as positional arguments, not `--config`; both fetch commands take `--dest` and `--force`. `mtb build` also takes `--out`, `--bbox`, `--no-contract` and `--no-elevation`. `mtb inspect` takes `--json` and `--top`.

## API

Everything answers in one envelope: `{success, data, error, meta}`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | the interface |
| `GET` | `/api/health` | liveness, and which artifact is loaded |
| `GET` | `/api/profiles` | profiles and the controls that adjust them |
| `POST` | `/api/routes/loop` | `{start, budget_minutes, profile, variants}` |
| `POST` | `/api/routes/point-to-point` | `{start, end, profile}` |
| `GET` | `/api/routes/{id}` | re-read a planned route |
| `GET` | `/api/routes/{id}/gpx` | download |
| `GET` | `/api/network` | trails in a bounding box, as bare GeoJSON for the map |

Interactive documentation at `/docs`.

## How it works

```
.pbf extracts ──┐
                ├─► offline build (CLI) ─► artifact (.npy CSR arrays) ─► FastAPI ─► MapLibre UI
GLO-30 DEM   ───┘
```

Nothing touches OpenStreetMap or the elevation model at request time. An offline build step produces a memory-resident graph; the server answers every request from RAM.

Three design decisions carry most of the weight:

- **Difficulty is inferred, not read.** `mtb:scale` exists on only ~3.4% of Belgian paths, so the technicality of a trail is derived from `surface`, `tracktype`, `smoothness`, `width` and friends, then refined by propagating surveyed values across connected trail groups. Every edge carries a confidence level, and the interface distinguishes surveyed from estimated.
- **Time, distance and preference stay separate.** Cost is `time × (1 + λ·(1 − preference))`. Preference drives every routing decision, but the time and distance reported back to you are measured independently — so the budget is honest.
- **Time is the constraint, distance is an outcome.** Two hours of technical singletrack is a much shorter ride than two hours of gravel, and the speed model knows it.

## Ride profiles

`xc` · `flow` · `tech` · `gravity` — named presets over a single weight vector, with slider overrides, defined in `config/profiles.yaml`.

`gravity` is derived from terrain (gradient, inferred technicality, descent-weighted cost) rather than bike-park tagging, which barely exists in Belgian OSM. It finds steep descent laps anywhere, but cannot tell a real bike park from a steep forest trail.

## Development

```powershell
uv run ruff check src tests
uv run black --check src tests
uv run isort --check-only src tests
uv run mypy src
uv run bandit -r src -q
uv run pytest --cov=src/mtb_planner --cov-report=term-missing --cov-fail-under=80
```

The front end has no bundler and no npm dependencies, so its tests run on the runtime built into Node. `uv run pytest` shells out to them, and they can also be run alone:

```powershell
node --test tests\js\pure.test.mjs tests\js\route.test.mjs
uv run pytest tests\e2e -m e2e          # real browser; needs `playwright install chromium`
uv run pytest -m "not slow and not e2e" # the fast pass
```

Test markers: `unit` (default, fast) · `integration` · `slow` (needs a built artifact or network) · `golden` · `e2e`.

## The basemap, and running offline

The map draws OpenStreetMap raster tiles for orientation. This is the one place the app talks to a third party, and it is a setting rather than a dependency:

```powershell
$env:MTB_BASEMAP_URL = ""        # paper only: route and trails, nothing external
$env:MTB_BASEMAP_URL = "https://your-tiles/{z}/{x}/{y}.png"
```

With it empty the map is the plain paper plate the rest of the design assumes, and the content security policy tightens back to `self` for everything. Attribution is required and always shown; `MTB_BASEMAP_ATTRIBUTION` sets the wording.

Default tiles come from the OpenStreetMap Foundation, whose [tile usage policy](https://operations.osmfoundation.org/policies/tiles/) permits modest personal use but not bulk or commercial traffic. For anything beyond one rider planning rides, point `MTB_BASEMAP_URL` at your own tiles or a paid provider.

Everything else stays local: routing, elevation and difficulty all come from the artifact on disk, and no plan request leaves the machine.

## Access and legality

Routes are generated from OpenStreetMap data, which does not fully model access law. Flanders restricts mountain biking to designated routes in many nature reserves; the Wallonian forest code restricts off-track riding and imposes hunting-season closures that OSM does not represent at all.

`bicycle=no` and `access=private|no` are treated as hard exclusions, and signed `route=mtb` ways are preferred. **This is not a guarantee that a generated route is legal to ride.** Check local signage and regulations.

Ride times rest on literature constants and have never been calibrated against real rides. Treat route *rankings* as reliable and the *clock* as indicative.

## Data sources and licensing

- Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, [ODbL](https://opendatacommons.org/licenses/odbl/). Extracts from [Geofabrik](https://download.geofabrik.de/).
- Elevation: [Copernicus DEM GLO-30](https://registry.opendata.aws/copernicus-dem/), © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under the Copernicus DEM open licence.
- Map rendering: [MapLibre GL JS](https://maplibre.org/) (BSD-3-Clause), vendored under `src/mtb_planner/web/static/vendor/`.
- Typefaces: Inter Tight and IBM Plex Mono, both under the SIL Open Font License.

## Licence

MIT.
