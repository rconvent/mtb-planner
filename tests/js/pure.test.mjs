// The difficulty ramp, the number formatting and the URL state.
//
// Run with `node --test tests/js`, which needs no npm and no bundler — the same
// reason the front end has no build step. A pytest wrapper shells out to it so
// one command still runs the whole suite.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import {
  MAX_TECH,
  TECH_STOPS,
  cssColour,
  oklchToRgb,
  parseOklch,
  techBucket,
  techColour,
  techColourExpression,
  techOklch,
} from "../../src/mtb_planner/web/static/js/ramp.js";
import {
  along,
  coordinate,
  duration,
  height,
  kilometres,
  metres,
  percent,
} from "../../src/mtb_planner/web/static/js/format.js";
import {
  DEFAULTS,
  MAX_MINUTES,
  MIN_MINUTES,
  parsePoint,
  readState,
  writeState,
} from "../../src/mtb_planner/web/static/js/urlstate.js";

describe("the difficulty ramp", () => {
  it("covers the whole mtb:scale ordinal", () => {
    assert.equal(TECH_STOPS.length, 7);
    assert.equal(MAX_TECH, 6);
  });

  it("gives every whole step its own colour", () => {
    const seen = new Set();
    for (let step = 0; step <= MAX_TECH; step += 1) seen.add(techColour(step));
    assert.equal(seen.size, 7);
  });

  it("interpolates between stops rather than banding", () => {
    // Inferred difficulty is a continuous float, so S2.5 must not snap to S2 or
    // S3 — banding would imply a precision the inference chain does not have.
    const middle = techColour(2.5);
    assert.notEqual(middle, techColour(2));
    assert.notEqual(middle, techColour(3));
  });

  it("takes the short way round the hue wheel", () => {
    // The ramp crosses zero between red and red-violet. Averaging hue naively
    // swings the long way through green, which looks like a bug on the map.
    const midpoint = techOklch(5.5).h;
    assert.ok(midpoint > 340 || midpoint < 15, `unexpected hue ${midpoint}`);
  });

  it("hands the map sRGB, because MapLibre cannot parse oklch", () => {
    // Not cosmetic: an oklch paint value makes MapLibre reject the layer and
    // the route silently does not draw at all.
    for (let step = 0; step <= MAX_TECH; step += 1) {
      assert.match(techColour(step), /^rgb\(\d+, \d+, \d+\)$/);
    }
  });

  it("still interpolates perceptually before converting", () => {
    // Interpolate in oklch, hand over sRGB. Interpolating in sRGB instead would
    // bunch the ramp through the reds.
    const mid = techOklch(2.5);
    assert.ok(mid.l > 0.78 && mid.l < 0.81, `lightness ${mid.l}`);
    assert.ok(mid.h > 70 && mid.h < 95, `hue ${mid.h}`);
  });

  it("clamps anything off the scale", () => {
    assert.equal(techColour(-3), techColour(0));
    assert.equal(techColour(99), techColour(MAX_TECH));
  });

  it("treats a missing difficulty as the easiest", () => {
    assert.equal(techColour(undefined), techColour(0));
    assert.equal(techColour(null), techColour(0));
  });

  it("buckets to the nearest whole step for a chip label", () => {
    assert.equal(techBucket(2.4), 2);
    assert.equal(techBucket(2.6), 3);
    assert.equal(techBucket(-1), 0);
    assert.equal(techBucket(9), 6);
  });

  it("builds a MapLibre expression over the same stops", () => {
    const expression = techColourExpression("tech");

    assert.equal(expression[0], "interpolate");
    assert.deepEqual(expression[2], ["get", "tech"]);
    // Seven stops, each a value and a colour.
    assert.equal(expression.length, 3 + 14);
    assert.equal(expression[4], techColour(0));
  });

  it("converts a known oklch value to the expected sRGB", () => {
    // oklch(62.8% 0.2577 29.23) is sRGB red; a wrong matrix shows up here
    // before it shows up as a wrongly coloured trail.
    const red = oklchToRgb({ l: 0.628, c: 0.2577, h: 29.23 });
    const [r, g, b] = red.match(/\d+/g).map(Number);
    assert.ok(r > 250, `red channel ${r}`);
    assert.ok(g < 12, `green channel ${g}`);
    assert.ok(b < 12, `blue channel ${b}`);
  });

  it("parses the oklch strings the tokens are written in", () => {
    assert.deepEqual(parseOklch("oklch(72% 0.14 145)"), { l: 0.72, c: 0.14, h: 145 });
    assert.equal(parseOklch("rgb(1, 2, 3)"), null);
    assert.equal(parseOklch(undefined), null);
  });

  it("passes through a colour the map already understands", () => {
    // So a token set in rgb or a keyword is not mangled on the way to the map.
    assert.equal(cssColour("rgb(1, 2, 3)"), "rgb(1, 2, 3)");
    assert.equal(cssColour("", "black"), "black");
    assert.match(cssColour("oklch(72% 0.14 145)"), /^rgb\(/);
  });

  it("matches the ramp the stylesheet paints chips with", () => {
    // Two copies exist because a canvas cannot read a custom property. This is
    // what stops them drifting apart.
    const css = readFileSync(
      new URL("../../src/mtb_planner/web/static/css/tokens.css", import.meta.url),
      "utf8"
    );

    TECH_STOPS.forEach((stop, step) => {
      const declared = css.match(new RegExp(`--tech-${step}:\\s*oklch\\(([^)]+)\\)`));
      assert.ok(declared, `tokens.css has no --tech-${step}`);
      const [l, c, h] = declared[1].trim().split(/\s+/);
      // Tolerance, not equality: 0.58 * 100 is 57.99999999999999 in binary
      // floating point, and the point of this test is drift in the ramp rather
      // than the last bit of a double.
      const close = (a, b) => Math.abs(a - b) < 1e-6;
      assert.ok(close(Number.parseFloat(l), stop.l * 100), `lightness of --tech-${step}`);
      assert.ok(close(Number.parseFloat(c), stop.c), `chroma of --tech-${step}`);
      assert.ok(close(Number.parseFloat(h), stop.h), `hue of --tech-${step}`);
    });
  });
});

describe("formatting figures", () => {
  it("shows bare minutes under an hour", () => {
    assert.equal(duration(45), "45 min");
    assert.equal(duration(59), "59 min");
  });

  it("shows hours and minutes above one", () => {
    assert.equal(duration(60), "1 h");
    assert.equal(duration(90), "1 h 30");
    assert.equal(duration(125), "2 h 05");
  });

  it("pads the minutes so a column of times lines up", () => {
    assert.ok(duration(125).endsWith("05"));
  });

  it("never shows a negative duration", () => {
    assert.equal(duration(-10), "0 min");
  });

  it("keeps distance to one decimal", () => {
    assert.equal(kilometres(26.432), "26.4");
    assert.equal(kilometres(0), "0.0");
  });

  it("rounds climbing to whole metres", () => {
    // The DEM corrections are worth about 20% of raw ascent; a decimal here
    // would assert precision the elevation model does not have.
    assert.equal(metres(531.44), "531");
  });

  it("renders a fraction as a whole per cent", () => {
    assert.equal(percent(0.036), "4%");
    assert.equal(percent(1), "100%");
  });

  it("scales a distance along the route to its size", () => {
    assert.equal(along(420), "420 m");
    assert.equal(along(18300), "18.3 km");
  });

  it("signs a height change", () => {
    assert.equal(height(42), "+42 m");
    assert.equal(height(-42), "-42 m");
  });

  it("writes coordinates at the precision the API takes", () => {
    assert.equal(coordinate(49.6116, 6.1319), "49.61160,6.13190");
  });

  it("treats rubbish as zero rather than NaN", () => {
    assert.equal(kilometres(undefined), "0.0");
    assert.equal(metres("nonsense"), "0");
    assert.equal(duration(null), "0 min");
  });
});

describe("URL state", () => {
  it("returns the defaults for an empty query", () => {
    const state = readState("");

    assert.equal(state.mode, DEFAULTS.mode);
    assert.equal(state.profile, DEFAULTS.profile);
    assert.equal(state.minutes, DEFAULTS.minutes);
    assert.equal(state.start, null);
  });

  it("reads the whole shape of a shared link", () => {
    const state = readState("?start=50.13,5.79&profile=flow&min=120&seed=42");

    assert.deepEqual(state.start, { lat: 50.13, lon: 5.79 });
    assert.equal(state.profile, "flow");
    assert.equal(state.minutes, 120);
    assert.equal(state.seed, 42);
  });

  it("tolerates a leading question mark or not", () => {
    assert.equal(readState("?min=120").minutes, readState("min=120").minutes);
  });

  it("clamps a budget outside what the API accepts", () => {
    assert.equal(readState("?min=1").minutes, MIN_MINUTES);
    assert.equal(readState("?min=99999").minutes, MAX_MINUTES);
  });

  it("ignores a pin it cannot parse rather than guessing", () => {
    // A bare Number() of a bad string yields 0,0 — the Gulf of Guinea — which
    // looks like a working app that has lost the rider.
    assert.equal(parsePoint("nonsense"), null);
    assert.equal(parsePoint("50.13"), null);
    assert.equal(parsePoint("50.13,5.79,3"), null);
    assert.equal(parsePoint(""), null);
    assert.equal(parsePoint(undefined), null);
  });

  it("rejects coordinates off the planet", () => {
    assert.equal(parsePoint("91,5"), null);
    assert.equal(parsePoint("50,181"), null);
  });

  it("round-trips a full state", () => {
    const original = {
      mode: "ab",
      start: { lat: 50.13, lon: 5.79 },
      end: { lat: 50.2, lon: 5.9 },
      profile: "tech",
      minutes: 150,
      variants: 4,
      seed: 7,
    };

    const restored = readState(writeState(original));

    assert.equal(restored.mode, "ab");
    assert.equal(restored.profile, "tech");
    assert.equal(restored.minutes, 150);
    assert.equal(restored.variants, 4);
    assert.equal(restored.seed, 7);
    assert.ok(Math.abs(restored.start.lat - 50.13) < 1e-5);
    assert.ok(Math.abs(restored.end.lon - 5.9) < 1e-5);
  });

  it("leaves defaults out of the link", () => {
    // So a shared URL carries what was chosen and stays short enough to paste.
    assert.equal(writeState({}), "");
    assert.ok(!writeState({ profile: DEFAULTS.profile }).includes("profile"));
  });

  it("omits a destination unless the mode uses one", () => {
    const query = writeState({ mode: "loop", end: { lat: 50, lon: 5 } });

    assert.ok(!query.includes("end="));
  });

  it("keeps a seed of zero, which is a real seed", () => {
    assert.equal(readState(writeState({ seed: 0 })).seed, 0);
  });
});
