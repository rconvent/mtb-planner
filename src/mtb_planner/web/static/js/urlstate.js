// The URL is the state.
//
// This screen has about six values in it. Putting them in the query string
// rather than a store means a planned ride is a link: shareable, bookmarkable,
// reproducible, and restored on reload without a line of persistence code. It
// is also why the server echoes the seed in every response.
//
// Everything is clamped and defaulted on the way in, because a URL is user
// input and the most likely edit to one is a typo.

export const MIN_MINUTES = 15;
export const MAX_MINUTES = 600;
export const MAX_VARIANTS = 6;

export const DEFAULTS = Object.freeze({
  mode: "loop",
  start: null,
  end: null,
  profile: "flow",
  minutes: 90,
  variants: 3,
  seed: null,
});

/**
 * Parse a `lat,lon` pair.
 *
 * Returns null rather than throwing or guessing: an unparseable pin should
 * leave the map where it was, not drop the rider at 0,0 off the coast of
 * Africa — which is exactly what a bare `Number()` of a bad string produces.
 */
export function parsePoint(text) {
  if (typeof text !== "string") return null;
  const parts = text.split(",");
  if (parts.length !== 2) return null;

  const lat = Number(parts[0]);
  const lon = Number(parts[1]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;

  return { lat, lon };
}

function clampInt(value, low, high, fallback) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(high, Math.max(low, parsed));
}

/** Read state out of a query string. */
export function readState(search = "") {
  const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);

  const mode = params.get("mode") === "ab" ? "ab" : DEFAULTS.mode;
  const seedRaw = params.get("seed");
  const seed = seedRaw === null ? null : clampInt(seedRaw, 0, 2 ** 31 - 1, DEFAULTS.seed);

  return {
    mode,
    start: parsePoint(params.get("start")),
    end: parsePoint(params.get("end")),
    profile: params.get("profile") || DEFAULTS.profile,
    minutes: clampInt(params.get("min"), MIN_MINUTES, MAX_MINUTES, DEFAULTS.minutes),
    variants: clampInt(params.get("variants"), 1, MAX_VARIANTS, DEFAULTS.variants),
    seed,
  };
}

/**
 * Serialise state back to a query string.
 *
 * Defaults are omitted, so a link carries only what was actually chosen and
 * stays short enough to paste into a message.
 */
export function writeState(state = {}) {
  const merged = { ...DEFAULTS, ...state };
  const params = new URLSearchParams();

  if (merged.mode !== DEFAULTS.mode) params.set("mode", merged.mode);
  if (merged.start) params.set("start", pointToText(merged.start));
  if (merged.end && merged.mode === "ab") params.set("end", pointToText(merged.end));
  if (merged.profile !== DEFAULTS.profile) params.set("profile", merged.profile);
  if (merged.minutes !== DEFAULTS.minutes) params.set("min", String(merged.minutes));
  if (merged.variants !== DEFAULTS.variants) params.set("variants", String(merged.variants));
  if (merged.seed !== null && merged.seed !== undefined) params.set("seed", String(merged.seed));

  const query = params.toString();
  return query ? `?${query}` : "";
}

function pointToText(point) {
  return `${point.lat.toFixed(5)},${point.lon.toFixed(5)}`;
}
