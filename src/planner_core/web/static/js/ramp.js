// The difficulty ramp.
//
// One continuous scale from S0 to S6. The stops are defined in oklch and
// interpolated in oklch, because that is the only way the steps come out
// perceptually even: the same sweep in sRGB bunches darkly through the reds and
// washes out through yellow, which would make S4 look like a bigger jump than
// S5.
//
// MapLibre cannot parse an oklch() string — it rejects the layer outright and
// the route simply does not draw — so everything handed to the map is converted
// to sRGB here. Interpolate perceptually, hand over sRGB: the conversion is the
// last step, not the first.
//
// The same seven stops appear in tokens.css for the chips and the profile fill,
// because a canvas cannot read a CSS custom property. Keeping the two copies in
// step matters more than avoiding the duplication, so a test asserts they agree.

/** Stops of the ramp in oklch, indexed by mtb:scale value. */
export const TECH_STOPS = [
  { l: 0.72, c: 0.14, h: 145 },
  { l: 0.75, c: 0.15, h: 120 },
  { l: 0.8, c: 0.15, h: 95 },
  { l: 0.78, c: 0.16, h: 70 },
  { l: 0.68, c: 0.19, h: 40 },
  { l: 0.58, c: 0.21, h: 15 },
  { l: 0.5, c: 0.2, h: 340 },
];

/** Highest value on the mtb:scale ordinal. */
export const MAX_TECH = TECH_STOPS.length - 1;

/**
 * The oklch coordinates for a technicality, interpolated between stops.
 *
 * Difficulty is a continuous float once it has been inferred rather than
 * surveyed, so S2.4 must land between the S2 and S3 stops instead of snapping
 * to one of them: banding the ramp would imply a precision the inference chain
 * does not have.
 */
export function techOklch(tech) {
  const clamped = Math.min(MAX_TECH, Math.max(0, Number(tech) || 0));
  const lower = Math.floor(clamped);
  const upper = Math.min(MAX_TECH, lower + 1);
  const t = clamped - lower;

  const a = TECH_STOPS[lower];
  const b = TECH_STOPS[upper];

  // Hue wraps: the ramp crosses zero between red and red-violet, so a plain
  // average would swing the long way round through green.
  let dh = b.h - a.h;
  if (dh > 180) dh -= 360;
  if (dh < -180) dh += 360;

  return {
    l: a.l + (b.l - a.l) * t,
    c: a.c + (b.c - a.c) * t,
    h: (a.h + dh * t + 360) % 360,
  };
}

/** Colour for a technicality, as sRGB the map can paint with. */
export function techColour(tech) {
  return oklchToRgb(techOklch(tech));
}

/** The nearest whole scale step, for labelling a chip. */
export function techBucket(tech) {
  return Math.min(MAX_TECH, Math.max(0, Math.round(Number(tech) || 0)));
}

/**
 * A MapLibre `interpolate` expression over the ramp.
 *
 * Built rather than written out, so the ramp has one definition even though it
 * is consumed by both CSS and a canvas paint property.
 */
export function techColourExpression(property = "tech") {
  const stops = [];
  for (let step = 0; step <= MAX_TECH; step += 1) {
    stops.push(step, techColour(step));
  }
  return ["interpolate", ["linear"], ["get", property], ...stops];
}

/** Parse an `oklch(L% C H)` string, or return null if it is something else. */
export function parseOklch(text) {
  if (typeof text !== "string") return null;
  const match = text.trim().match(/^oklch\(\s*([\d.]+)%?\s+([\d.]+)\s+([\d.]+)/i);
  if (!match) return null;
  return {
    l: Number.parseFloat(match[1]) / 100,
    c: Number.parseFloat(match[2]),
    h: Number.parseFloat(match[3]),
  };
}

/**
 * Convert any colour the design tokens might hold into one the map can parse.
 *
 * Anything already in a form MapLibre understands is passed through untouched,
 * so this stays safe if a token is ever set in hex or rgb.
 */
export function cssColour(text, fallback = "black") {
  if (!text) return fallback;
  const parsed = parseOklch(text);
  return parsed ? oklchToRgb(parsed) : text;
}

/**
 * oklch to sRGB.
 *
 * OKLab through LMS to linear sRGB, then the sRGB transfer function. Values
 * outside the display gamut are clamped per channel, which desaturates rather
 * than shifting hue — acceptable here because the ramp was chosen to sit inside
 * sRGB in the first place.
 */
export function oklchToRgb({ l, c, h }) {
  const hRad = (h * Math.PI) / 180;
  const a = c * Math.cos(hRad);
  const b = c * Math.sin(hRad);

  const lms = [
    l + 0.3963377774 * a + 0.2158037573 * b,
    l - 0.1055613458 * a - 0.0638541728 * b,
    l - 0.0894841775 * a - 1.291485548 * b,
  ].map((value) => value ** 3);

  const [lp, mp, sp] = lms;
  const linear = [
    4.0767416621 * lp - 3.3077115913 * mp + 0.2309699292 * sp,
    -1.2684380046 * lp + 2.6097574011 * mp - 0.3413193965 * sp,
    -0.0041960863 * lp - 0.7034186147 * mp + 1.707614701 * sp,
  ];

  const channels = linear.map((value) => {
    const encoded = value <= 0.0031308 ? 12.92 * value : 1.055 * value ** (1 / 2.4) - 0.055;
    return Math.round(Math.min(1, Math.max(0, encoded)) * 255);
  });

  return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`;
}
