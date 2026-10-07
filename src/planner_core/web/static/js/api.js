// Talking to the planner.
//
// Every endpoint answers in the same envelope, on success and on failure alike,
// so there is one place that unwraps it and one shape of error to render. The
// server already writes messages meant for a rider — "No routable way near
// 50.13,5.79 (nearest is 118 m away)" — so they are surfaced as they arrive
// rather than replaced with something vaguer.

/** An API failure carrying the stable code and structured detail. */
export class ApiError extends Error {
  constructor(message, code, detail, status) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.detail = detail;
    this.status = status;
  }
}

async function unwrap(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      "The server sent something unreadable.",
      "unreadable",
      null,
      response.status
    );
  }

  if (response.ok && body?.success) return body;

  const error = body?.error ?? {};
  throw new ApiError(
    error.message || "The request failed.",
    error.code || "unknown",
    error.detail ?? null,
    response.status
  );
}

async function post(path, payload, signal) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  return unwrap(response);
}

/** The profiles and the controls that adjust them. */
export async function fetchProfiles() {
  return unwrap(await fetch("/api/profiles"));
}

/** Liveness and which artifact is loaded. */
export async function fetchHealth() {
  return unwrap(await fetch("/api/health"));
}

/** Generate loops from a point within a time budget. */
export async function planLoop(state, signal) {
  return post(
    "/api/routes/loop",
    {
      start: state.start,
      budget_minutes: state.minutes,
      profile: state.profile,
      variants: state.variants,
      ...(state.seed === null || state.seed === undefined ? {} : { seed: state.seed }),
    },
    signal
  );
}

/** Plan a route between two points.
 *
 * The budget travels with a point-to-point request too, where it means slack
 * rather than a target: the route goes where it was asked to go, and time left
 * over after the quickest way there is spent on better ground instead.
 */
export async function planPointToPoint(state, signal) {
  return post(
    "/api/routes/point-to-point",
    {
      start: state.start,
      end: state.end,
      profile: state.profile,
      budget_minutes: state.minutes,
    },
    signal
  );
}
