// Turning figures into what a rider reads.
//
// Pure and separate from the DOM because these are the numbers the whole
// product is judged on: a time budget shown as 1:59 when the server said 120
// minutes is a bug the eye does not catch, and it is much easier to pin here
// than through a rendered panel.
//
// Precision is deliberately coarse. The time model is uncalibrated against real
// rides, so a duration printed to the second would assert an accuracy the model
// does not have. Minutes for time, one decimal for kilometres, whole metres for
// climbing.

/** Hours and minutes, or bare minutes under an hour. */
export function duration(totalMinutes) {
  const value = Math.max(0, Math.round(Number(totalMinutes) || 0));
  if (value < 60) return `${value} min`;
  const hours = Math.floor(value / 60);
  const minutes = value % 60;
  return minutes === 0 ? `${hours} h` : `${hours} h ${String(minutes).padStart(2, "0")}`;
}

/** Distance in kilometres, to one decimal. */
export function kilometres(km) {
  return (Number(km) || 0).toFixed(1);
}

/** Whole metres, for climbing. */
export function metres(m) {
  return String(Math.round(Number(m) || 0));
}

/** A fraction as a whole per cent. */
export function percent(fraction) {
  return `${Math.round((Number(fraction) || 0) * 100)}%`;
}

/**
 * A distance along a route, scaled to how far along it is.
 *
 * Metres near the start of a ride, kilometres once past one: "420 m" is useful
 * where "0.4 km" is not, and "18.3 km" is useful where "18300 m" is not.
 */
export function along(distanceM) {
  const value = Math.max(0, Number(distanceM) || 0);
  return value < 1000 ? `${Math.round(value)} m` : `${(value / 1000).toFixed(1)} km`;
}

/** Signed height difference, for a profile readout. */
export function height(m) {
  const value = Math.round(Number(m) || 0);
  return value > 0 ? `+${value} m` : `${value} m`;
}

/** A coordinate pair at the precision the API accepts, for the URL. */
export function coordinate(lat, lon) {
  return `${(Number(lat) || 0).toFixed(5)},${(Number(lon) || 0).toFixed(5)}`;
}
