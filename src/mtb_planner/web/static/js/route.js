// Turning a route payload into things a map and a chart can use.
//
// Three jobs, all pure, all worth testing away from a browser:
//
//   - split the polyline into per-segment features, so difficulty can be
//     painted along it and inferred stretches dashed;
//   - find the point on the route at a given distance, which is what lets
//     hovering the elevation chart put a marker on the trail;
//   - find the distance at a given point, which is the same link the other way
//     round.
//
// Segment lengths come from the distances the server already reported, never
// from re-measuring the polyline: the server summed edge lengths, and a flat
// projection of a drawn line understates ground truth. Only the *within* a
// segment interpolation uses planar geometry, where a few metres of error is
// invisible under a cursor.

const EARTH_M_PER_DEG = 111320;

/** Total route length, as the server measured it. */
export function totalLength(route) {
  const reported = route?.stats?.distance_m;
  if (Number.isFinite(reported) && reported > 0) return reported;

  const segments = route?.segments ?? [];
  return segments.length === 0 ? 0 : segments[segments.length - 1].at_m;
}

/**
 * Length of every segment, derived from consecutive start distances.
 *
 * The last one is measured against the reported total, so the lengths sum to
 * exactly the distance shown to the rider.
 */
export function segmentLengths(route) {
  const segments = route?.segments ?? [];
  const total = totalLength(route);

  return segments.map((segment, index) =>
    index < segments.length - 1
      ? segments[index + 1].at_m - segment.at_m
      : Math.max(0, total - segment.at_m)
  );
}

/**
 * One GeoJSON feature per edge, carrying its difficulty.
 *
 * Per-segment rather than one line with a gradient, because MapLibre cannot
 * vary a dash pattern along a single feature — and the dash is what marks
 * difficulty that was inferred rather than surveyed.
 */
export function segmentFeatures(route) {
  const coordinates = route?.geometry?.coordinates ?? [];
  const segments = route?.segments ?? [];

  const features = segments
    .map((segment) => ({
      type: "Feature",
      properties: {
        tech: segment.tech,
        surveyed: segment.surveyed === true,
        at_m: segment.at_m,
      },
      geometry: {
        type: "LineString",
        coordinates: coordinates.slice(segment.first, segment.last + 1),
      },
    }))
    .filter((feature) => feature.geometry.coordinates.length >= 2);

  return { type: "FeatureCollection", features };
}

/** The whole route as one feature, for a halo underneath the segments. */
export function routeFeature(route) {
  return {
    type: "Feature",
    properties: {},
    geometry: {
      type: "LineString",
      coordinates: route?.geometry?.coordinates ?? [],
    },
  };
}

/** Bounding box as MapLibre wants it: [[west, south], [east, north]]. */
export function bounds(route) {
  const coordinates = route?.geometry?.coordinates ?? [];
  if (coordinates.length === 0) return null;

  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;

  for (const [lon, lat] of coordinates) {
    if (lon < west) west = lon;
    if (lon > east) east = lon;
    if (lat < south) south = lat;
    if (lat > north) north = lat;
  }

  return [
    [west, south],
    [east, north],
  ];
}

/**
 * The point on the route at `distanceM` along it.
 *
 * Walks to the containing segment, then interpolates between its vertices by
 * their share of its planar length. Clamped at both ends, so a cursor dragged
 * past the chart does not wrap the marker round to the start.
 */
export function positionAt(route, distanceM) {
  const coordinates = route?.geometry?.coordinates ?? [];
  const segments = route?.segments ?? [];
  if (coordinates.length === 0) return null;

  const first = { lon: coordinates[0][0], lat: coordinates[0][1] };
  if (segments.length === 0) return first;

  const lengths = segmentLengths(route);
  const total = totalLength(route);
  const target = Math.min(total, Math.max(0, Number(distanceM) || 0));

  let index = segments.length - 1;
  for (let i = 0; i < segments.length; i += 1) {
    if (target <= segments[i].at_m + lengths[i]) {
      index = i;
      break;
    }
  }

  const segment = segments[index];
  const points = coordinates.slice(segment.first, segment.last + 1);
  if (points.length === 0) return first;
  if (points.length === 1) return { lon: points[0][0], lat: points[0][1] };

  const spans = [];
  let walked = 0;
  for (let i = 1; i < points.length; i += 1) {
    const span = planarDistance(points[i - 1], points[i]);
    spans.push(span);
    walked += span;
  }
  if (walked <= 0) return { lon: points[0][0], lat: points[0][1] };

  // Scaled into the polyline metric, so the marker sits where the reported
  // distance says rather than drifting by the projection error.
  const within = target - segment.at_m;
  const share = lengths[index] > 0 ? within / lengths[index] : 0;
  const wanted = Math.min(walked, Math.max(0, share * walked));

  let travelled = 0;
  for (let i = 0; i < spans.length; i += 1) {
    if (travelled + spans[i] >= wanted) {
      const t = spans[i] === 0 ? 0 : (wanted - travelled) / spans[i];
      return {
        lon: points[i][0] + (points[i + 1][0] - points[i][0]) * t,
        lat: points[i][1] + (points[i + 1][1] - points[i][1]) * t,
      };
    }
    travelled += spans[i];
  }

  const end = points[points.length - 1];
  return { lon: end[0], lat: end[1] };
}

/**
 * The distance along the route nearest to a point.
 *
 * Vertex resolution rather than true perpendicular projection: the marker only
 * has to land on the trail under the cursor, and route vertices are metres
 * apart where that matters.
 */
export function nearestDistance(route, point) {
  const coordinates = route?.geometry?.coordinates ?? [];
  const segments = route?.segments ?? [];
  if (coordinates.length === 0 || segments.length === 0) return 0;

  const lengths = segmentLengths(route);
  const target = [point.lon, point.lat];
  let best = Infinity;
  let bestDistance = 0;

  segments.forEach((segment, index) => {
    const points = coordinates.slice(segment.first, segment.last + 1);
    if (points.length === 0) return;

    let spanTotal = 0;
    for (let i = 1; i < points.length; i += 1) {
      spanTotal += planarDistance(points[i - 1], points[i]);
    }

    let walked = 0;
    for (let i = 0; i < points.length; i += 1) {
      if (i > 0) walked += planarDistance(points[i - 1], points[i]);
      const gap = planarDistance(points[i], target);
      if (gap < best) {
        best = gap;
        const share = spanTotal > 0 ? walked / spanTotal : 0;
        bestDistance = segment.at_m + share * lengths[index];
      }
    }
  });

  return bestDistance;
}

/** How much of the route has difficulty that was actually surveyed. */
export function surveyedFraction(route) {
  const segments = route?.segments ?? [];
  const total = totalLength(route);
  if (total <= 0) return 0;

  const lengths = segmentLengths(route);
  let surveyed = 0;
  segments.forEach((segment, index) => {
    if (segment.surveyed) surveyed += lengths[index];
  });
  return surveyed / total;
}

function planarDistance(a, b) {
  const meanLat = ((a[1] + b[1]) / 2) * (Math.PI / 180);
  const dx = (b[0] - a[0]) * EARTH_M_PER_DEG * Math.cos(meanLat);
  const dy = (b[1] - a[1]) * EARTH_M_PER_DEG;
  return Math.sqrt(dx * dx + dy * dy);
}
