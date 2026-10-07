// Wiring. The only module that touches both the API and the DOM.
//
// State lives in the URL, so there is no store to keep in sync: a change writes
// the query string, and the query string is what a reload or a shared link reads
// back. Six values do not justify anything more.

import { ApiError, fetchHealth, fetchProfiles, planLoop, planPointToPoint } from "./api.js";
import { along, duration, kilometres, metres, percent } from "./format.js";
import { RouteMap } from "./map.js";
import { ElevationProfile } from "./profile.js";
import { techBucket } from "./ramp.js";
import { surveyedFraction } from "./route.js";
import { readState, writeState } from "./urlstate.js";

const el = (id) => document.getElementById(id);

// "S" on the mtb:scale, "T" on the hiking scale. The server says which, so one
// interface serves every sport.
const GRADE_PREFIX = document.body.dataset.gradePrefix ?? "S";

const dom = {
  dock: el("dock"),
  modes: el("modes"),
  endField: el("end-field"),
  endReadout: el("end-readout"),
  variantsField: el("variants-field"),
  profiles: el("profiles"),
  profileNote: el("profile-note"),
  budget: el("budget"),
  budgetOut: el("budget-out"),
  budgetHint: el("budget-hint"),
  variants: el("variants"),
  variantsOut: el("variants-out"),
  plan: el("plan"),
  status: el("status"),
  start: el("start-readout"),
  sheet: el("sheet"),
  time: el("stat-time"),
  distance: el("stat-distance"),
  ascent: el("stat-ascent"),
  descent: el("stat-descent"),
  speed: el("stat-speed"),
  tech: el("stat-tech"),
  chips: el("chips"),
  coverage: el("coverage"),
  coverageNote: el("coverage-note"),
  candidates: el("candidates"),
  warnings: el("warnings"),
  gpx: el("gpx"),
  profileChart: el("profile-chart"),
  axisStart: el("axis-start"),
  axisEnd: el("axis-end"),
  networkNote: el("network-note"),
};

let state = readState(location.search);
let profileSpecs = [];
let results = [];
let selected = 0;
let inFlight = null;

const chart = new ElevationProfile(dom.profileChart, {
  onProbe: (distance) => map.showProbeAt(distance),
});

const map = new RouteMap("map", {
  styleUrl: "/api/style",
  onPick: (lngLat) => {
    const point = { lat: lngLat.lat, lon: lngLat.lng };
    // In A-to-B the first click sets the start and the second the finish, then
    // it alternates: placing a pin should never need a mode switch as well.
    if (state.mode === "ab" && state.start && !state.end) {
      state = { ...state, end: point };
    } else if (state.mode === "ab") {
      state = { ...state, start: point, end: null };
    } else {
      state = { ...state, start: point };
    }
    syncUrl();
    renderStart();
    if (state.mode !== "ab" || state.end) plan();
  },
  onProbe: (distance) => {
    if (distance === null) chart.clearProbe(false);
    else chart.showAt(distance);
  },
  onNetwork: ({ drawn, total, truncated }) => {
    // Saying so is the point. A map showing a fifth of the trails with no
    // notice reads as an area with very few trails.
    dom.networkNote.textContent = truncated
      ? `Showing the ${drawn.toLocaleString()} longest of ${total.toLocaleString()} trails in view. Zoom in for the rest.`
      : "";
  },
});

// --- rendering ---------------------------------------------------------

function asText(point, fallback) {
  return point ? `${point.lat.toFixed(4)}, ${point.lon.toFixed(4)}` : fallback;
}

function renderStart() {
  const ab = state.mode === "ab";
  dom.start.textContent = asText(state.start, "click the map");
  dom.endReadout.textContent = asText(state.end, "click the map again");

  dom.endField.hidden = !ab;
  // A point-to-point route has one answer, so there is nothing to offer options
  // over; the time budget becomes slack to spend rather than a target to hit,
  // and the hint has to say so or the slider looks broken when the route comes
  // back the same length as before.
  dom.variantsField.hidden = ab;
  dom.budgetHint.textContent = ab
    ? "Time is slack. Whatever is left after the quickest way there is spent on better ground."
    : "Time is the budget. Distance is measured and reported, never enforced.";

  dom.plan.disabled = !state.start || (ab && !state.end);
  map.showFinish(ab ? state.end : null);
}

function renderModes() {
  for (const button of dom.modes.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
  }
}

function renderProfiles() {
  dom.profiles.innerHTML = "";
  for (const spec of profileSpecs) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = spec.name;
    button.dataset.profile = spec.id;
    button.setAttribute("aria-pressed", String(spec.id === state.profile));
    button.addEventListener("click", () => {
      state = { ...state, profile: spec.id };
      syncUrl();
      renderProfiles();
      if (state.start) plan();
    });
    dom.profiles.append(button);
  }
  const chosen = profileSpecs.find((spec) => spec.id === state.profile);
  dom.profileNote.textContent = chosen ? chosen.description : "";
}

function renderBudget() {
  dom.budget.value = String(state.minutes);
  dom.budgetOut.textContent = duration(state.minutes);
  dom.variants.value = String(state.variants);
  dom.variantsOut.textContent = String(state.variants);
}

function say(message) {
  dom.status.hidden = !message;
  dom.status.textContent = message ?? "";
}

function renderChips(loop) {
  dom.chips.innerHTML = "";
  const character = loop.character ?? {};
  const bucket = techBucket(character.technicality ?? 0);

  const chips = [
    { label: `${GRADE_PREFIX}${bucket} typical`, tech: bucket },
    { label: `${percent(character.singletrack_fraction ?? 0)} singletrack`, neutral: true },
    { label: `${percent(character.paved_fraction ?? 0)} surfaced`, neutral: true },
  ];
  if ((character.signed_fraction ?? 0) > 0.05) {
    chips.push({ label: `${percent(character.signed_fraction)} signed`, neutral: true });
  }

  for (const chip of chips) {
    const span = document.createElement("span");
    span.className = chip.neutral ? "chip neutral" : "chip";
    if (!chip.neutral) span.dataset.tech = String(chip.tech);
    span.textContent = chip.label;
    dom.chips.append(span);
  }
}

function renderCoverage(loop) {
  const surveyed = surveyedFraction(loop);
  dom.coverage.innerHTML = "";

  const known = document.createElement("span");
  known.className = "surveyed";
  known.style.width = `${(surveyed * 100).toFixed(1)}%`;

  const guessed = document.createElement("span");
  guessed.className = "estimated";
  guessed.style.width = `${((1 - surveyed) * 100).toFixed(1)}%`;

  dom.coverage.append(known, guessed);
  // Stated rather than implied. Inferring difficulty is the normal case here,
  // and a rider is entitled to know how much of it was measured.
  dom.coverageNote.textContent =
    `${percent(surveyed)} of this route has surveyed difficulty; the rest is estimated.`;
}

function renderWarnings(loop) {
  dom.warnings.innerHTML = "";
  for (const note of loop.warnings ?? []) {
    const item = document.createElement("li");
    item.textContent = note;
    dom.warnings.append(item);
  }
}

function renderStats(loop) {
  const stats = loop.stats ?? {};
  dom.time.textContent = duration(stats.moving_time_min ?? 0);
  dom.distance.textContent = kilometres(stats.distance_km ?? 0);
  dom.ascent.textContent = metres(stats.ascent_m ?? 0);
  dom.descent.textContent = metres(stats.descent_m ?? 0);
  dom.speed.textContent = (stats.average_speed_km_h ?? 0).toFixed(1);
  dom.tech.textContent = String(techBucket(loop.character?.technicality ?? 0));
  dom.gpx.href = loop.gpx_url ?? "#";
}

function renderCandidates() {
  dom.candidates.innerHTML = "";
  if (results.length < 2) return;

  results.forEach((loop, index) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate";
    button.setAttribute("aria-current", String(index === selected));

    const ordinal = document.createElement("span");
    ordinal.className = "ordinal";
    ordinal.textContent = String(index + 1);

    const summary = document.createElement("span");
    summary.className = "summary";
    summary.textContent = `${duration(loop.stats.moving_time_min)} · ${kilometres(
      loop.stats.distance_km
    )} km`;

    const aside = document.createElement("span");
    aside.className = "aside";
    aside.textContent = `${metres(loop.stats.ascent_m)} m up`;

    button.append(ordinal, summary, aside);
    button.addEventListener("click", () => select(index));
    item.append(button);
    dom.candidates.append(item);
  });
}

function select(index) {
  selected = index;
  const loop = results[index];
  if (!loop) return;

  renderStats(loop);
  renderChips(loop);
  renderCoverage(loop);
  renderWarnings(loop);
  renderCandidates();

  chart.render(loop.elevation ?? []);
  dom.axisStart.textContent = "0 km";
  dom.axisEnd.textContent = along(loop.stats?.distance_m ?? 0);

  map.show(loop);
  dom.dock.hidden = false;
}

// --- planning ----------------------------------------------------------

async function plan() {
  if (!state.start) {
    say("Click the map to choose where the ride starts.");
    return;
  }

  if (inFlight) inFlight.abort();
  inFlight = new AbortController();

  dom.plan.disabled = true;
  dom.plan.textContent = "Planning";
  say(null);

  try {
    const body =
      state.mode === "ab" && state.end
        ? await planPointToPoint(state, inFlight.signal)
        : await planLoop(state, inFlight.signal);

    results = body.data.loops ?? body.data.routes ?? [];
    selected = 0;

    if (results.length === 0) {
      say("Nothing came back for that request.");
      return;
    }

    select(0);

    if (body.meta.truncated) {
      say("The search ran out of time, so these are the best found so far.");
    } else if (state.mode !== "ab" && results.some((loop) => loop.within_budget === false)) {
      // Point-to-point says this in the route's own warnings, where it can name
      // how long the ride actually takes rather than only that it does not fit.
      say("No loop fitted the time exactly; the closest are shown.");
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    handleFailure(error);
  } finally {
    dom.plan.disabled = false;
    dom.plan.textContent = "Plan";
    inFlight = null;
  }
}

function handleFailure(error) {
  if (!(error instanceof ApiError)) {
    say("Something went wrong talking to the planner.");
    return;
  }

  // The server writes messages meant for a rider, so they are shown as sent.
  say(error.message);

  // Off-network arrives with the nearest routable point, so offer it.
  if (error.code === "point_off_network" && error.detail?.nearest) {
    const nearest = error.detail.nearest;
    state = { ...state, start: { lat: nearest.lat, lon: nearest.lon } };
    syncUrl();
    renderStart();
    say(`${error.message} Moved the start to the nearest trail.`);
    plan();
  }
}

// --- state -------------------------------------------------------------

function syncUrl() {
  history.replaceState(null, "", writeState(state) || location.pathname);
}

/** Whether the current state describes a request the planner can answer.
 *
 * The same condition that disables the Plan button. A slider must consult it
 * too: in A-to-B with no finish placed yet, replanning would quietly fall
 * through to generating a loop, which is not what the rider is looking at.
 */
function canPlan() {
  return Boolean(state.start) && (state.mode !== "ab" || Boolean(state.end));
}

// --- boot --------------------------------------------------------------

dom.budget.addEventListener("input", () => {
  state = { ...state, minutes: Number(dom.budget.value) };
  dom.budgetOut.textContent = duration(state.minutes);
});
dom.budget.addEventListener("change", () => {
  syncUrl();
  if (canPlan()) plan();
});

dom.variants.addEventListener("input", () => {
  state = { ...state, variants: Number(dom.variants.value) };
  dom.variantsOut.textContent = String(state.variants);
});
dom.variants.addEventListener("change", () => {
  syncUrl();
  if (canPlan()) plan();
});

for (const button of dom.modes.querySelectorAll("button")) {
  button.addEventListener("click", () => {
    const mode = button.dataset.mode;
    if (mode === state.mode) return;
    // Dropping the finish on the way back to a loop, so a later switch to A-to-B
    // does not silently reuse a pin the rider cannot see.
    state = { ...state, mode, end: mode === "ab" ? state.end : null };
    syncUrl();
    renderModes();
    renderStart();
    if (canPlan()) plan();
  });
}

dom.plan.addEventListener("click", () => plan());

window.addEventListener("popstate", () => {
  state = readState(location.search);
  renderModes();
  renderStart();
  renderBudget();
  renderProfiles();
});

renderModes();
renderStart();
renderBudget();

fetchProfiles()
  .then((body) => {
    profileSpecs = body.data.profiles ?? [];
    // A link made for one sport can name a profile the other does not have;
    // fall back to this sport's default rather than planning with none.
    if (!profileSpecs.some((spec) => spec.id === state.profile)) {
      const fallback = body.data.default_profile ?? profileSpecs[0]?.id;
      if (fallback) state = { ...state, profile: fallback };
    }
    renderProfiles();
  })
  .catch(() => say("Could not load the profiles."));

fetchHealth()
  .then((body) => {
    const graph = body.data.graph ?? {};
    dom.sheet.textContent = graph.loaded ? graph.artifact : "no artifact";
    if (!graph.loaded) {
      say(graph.error ?? "The routing graph is not loaded, so planning is unavailable.");
      dom.plan.disabled = true;
    } else if (canPlan()) {
      plan();
    }
  })
  .catch(() => say("Could not reach the planner."));
