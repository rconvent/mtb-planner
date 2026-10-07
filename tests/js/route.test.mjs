// Turning a route payload into map features and positions.
//
// `positionAt` and `nearestDistance` are the two halves of the interaction the
// design rests on: hovering the elevation chart slides a marker along the trail,
// and hovering the trail slides the chart cursor. If either drifts, the two
// disagree in front of the user, so both are pinned against a route whose
// geometry is arithmetic rather than realistic.

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  bounds,
  nearestDistance,
  positionAt,
  routeFeature,
  segmentFeatures,
  segmentLengths,
  surveyedFraction,
  totalLength,
} from "../../src/planner_core/web/static/js/route.js";

// Four points due east, split into three segments of roughly equal length.
// A lon step of 0.001 at this latitude is about 71 m, but the segment lengths
// are whatever the payload says they are, which is the point.
const ROUTE = {
  stats: { distance_m: 300 },
  geometry: {
    type: "LineString",
    coordinates: [
      [5.0, 50.0],
      [5.001, 50.0],
      [5.002, 50.0],
      [5.003, 50.0],
    ],
  },
  segments: [
    { first: 0, last: 1, at_m: 0, tech: 1, surveyed: true },
    { first: 1, last: 2, at_m: 100, tech: 3, surveyed: false },
    { first: 2, last: 3, at_m: 200, tech: 5, surveyed: true },
  ],
};

const EMPTY = { stats: { distance_m: 0 }, geometry: { coordinates: [] }, segments: [] };

describe("route measurements", () => {
  it("takes the total length the server reported", () => {
    // Never re-measured from the polyline: the server summed edge lengths, and
    // a flat projection of a drawn line understates the ground.
    assert.equal(totalLength(ROUTE), 300);
  });

  it("derives each segment length from the next start distance", () => {
    assert.deepEqual(segmentLengths(ROUTE), [100, 100, 100]);
  });

  it("makes the lengths sum to the distance shown to the rider", () => {
    const sum = segmentLengths(ROUTE).reduce((a, b) => a + b, 0);
    assert.equal(sum, ROUTE.stats.distance_m);
  });

  it("copes with a route of nothing", () => {
    assert.equal(totalLength(EMPTY), 0);
    assert.deepEqual(segmentLengths(EMPTY), []);
    assert.equal(surveyedFraction(EMPTY), 0);
  });

  it("reports how much difficulty was actually surveyed", () => {
    // Two of three segments, by length.
    assert.ok(Math.abs(surveyedFraction(ROUTE) - 2 / 3) < 1e-9);
  });
});

describe("map features", () => {
  it("emits one feature per segment", () => {
    assert.equal(segmentFeatures(ROUTE).features.length, 3);
  });

  it("carries difficulty and survey state onto each feature", () => {
    const [first, second] = segmentFeatures(ROUTE).features;

    assert.equal(first.properties.tech, 1);
    assert.equal(first.properties.surveyed, true);
    assert.equal(second.properties.surveyed, false);
  });

  it("slices each segment to its own geometry", () => {
    const [first] = segmentFeatures(ROUTE).features;

    assert.deepEqual(first.geometry.coordinates, [
      [5.0, 50.0],
      [5.001, 50.0],
    ]);
  });

  it("drops a segment too short to draw", () => {
    const degenerate = {
      ...ROUTE,
      segments: [{ first: 0, last: 0, at_m: 0, tech: 1, surveyed: true }],
    };

    assert.equal(segmentFeatures(degenerate).features.length, 0);
  });

  it("also offers the whole route as one feature for a halo", () => {
    assert.equal(routeFeature(ROUTE).geometry.coordinates.length, 4);
  });

  it("bounds the route west-south then east-north", () => {
    assert.deepEqual(bounds(ROUTE), [
      [5.0, 50.0],
      [5.003, 50.0],
    ]);
  });

  it("has no bounds for an empty route", () => {
    assert.equal(bounds(EMPTY), null);
  });
});

describe("position along the route", () => {
  it("starts at the first point", () => {
    const point = positionAt(ROUTE, 0);

    assert.ok(Math.abs(point.lon - 5.0) < 1e-9);
  });

  it("ends at the last point", () => {
    const point = positionAt(ROUTE, 300);

    assert.ok(Math.abs(point.lon - 5.003) < 1e-9);
  });

  it("lands on a segment boundary exactly", () => {
    const point = positionAt(ROUTE, 100);

    assert.ok(Math.abs(point.lon - 5.001) < 1e-6, `got ${point.lon}`);
  });

  it("interpolates within a segment", () => {
    const point = positionAt(ROUTE, 150);

    assert.ok(Math.abs(point.lon - 5.0015) < 1e-6, `got ${point.lon}`);
  });

  it("clamps rather than wrapping past either end", () => {
    // A cursor dragged off the chart must not send the marker back to the start.
    assert.ok(Math.abs(positionAt(ROUTE, -50).lon - 5.0) < 1e-9);
    assert.ok(Math.abs(positionAt(ROUTE, 9999).lon - 5.003) < 1e-9);
  });

  it("treats a missing distance as the start", () => {
    assert.ok(Math.abs(positionAt(ROUTE, undefined).lon - 5.0) < 1e-9);
  });

  it("has no position on an empty route", () => {
    assert.equal(positionAt(EMPTY, 10), null);
  });
});

describe("distance at a position", () => {
  it("finds the start", () => {
    assert.ok(nearestDistance(ROUTE, { lon: 5.0, lat: 50.0 }) < 1e-6);
  });

  it("finds a mid-route vertex", () => {
    const found = nearestDistance(ROUTE, { lon: 5.002, lat: 50.0 });

    assert.ok(Math.abs(found - 200) < 1, `got ${found}`);
  });

  it("snaps a point beside the trail onto it", () => {
    // The rider hovers near the line, not exactly on it.
    const found = nearestDistance(ROUTE, { lon: 5.001, lat: 50.0004 });

    assert.ok(Math.abs(found - 100) < 1, `got ${found}`);
  });

  it("round-trips against positionAt", () => {
    // The two halves of the linked hover must agree, or the marker and the
    // chart cursor point at different places on the same route. Vertices are
    // 100 m apart here and the search is vertex-resolution, so agreement is
    // expected to within a little over half a span.
    for (const distance of [0, 50, 100, 175, 250, 300]) {
      const point = positionAt(ROUTE, distance);
      const back = nearestDistance(ROUTE, point);
      assert.ok(Math.abs(back - distance) <= 55, `${distance} -> ${back}`);
    }
  });

  it("returns the start for an empty route", () => {
    assert.equal(nearestDistance(EMPTY, { lon: 5, lat: 50 }), 0);
  });
});
