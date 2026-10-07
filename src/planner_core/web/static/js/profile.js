// The elevation profile, and half of the interaction that sells the product.
//
// Hovering the chart slides a marker along the route; hovering the route slides
// the cursor here. This module owns the chart and reports where the pointer is
// as a distance along the ride; whoever wired it up decides what that means on
// the map. Keeping the link in terms of distance rather than pixels or indices
// is what lets the two directions share one definition.
//
// Drawn as SVG rather than canvas: a few hundred points is nothing, the shape
// stays crisp at any zoom, and the fill can be revealed with a CSS clip-path
// instead of an animation loop.
//
// The height axis is relative. The artifact stores per-edge ascent and descent
// but no absolute altitude, so the chart shows height gained from the start and
// says so in its label rather than implying metres above sea level.

const VIEW_W = 1000;
const VIEW_H = 200;
const PAD_TOP = 12;
const PAD_BOTTOM = 8;

export class ElevationProfile {
  /**
   * @param {HTMLElement} root container carrying the .profile class
   * @param {{onProbe?: (distanceM: number|null) => void}} handlers
   */
  constructor(root, handlers = {}) {
    this.root = root;
    this.onProbe = handlers.onProbe ?? (() => {});
    this.points = [];
    this.total = 0;
    this.low = 0;
    this.span = 1;
    this.cursorAt = 0;

    root.innerHTML = `
      <svg viewBox="0 0 ${VIEW_W} ${VIEW_H}" preserveAspectRatio="none"
           tabindex="0" role="slider" aria-label="Elevation profile"
           aria-valuemin="0" aria-valuenow="0" aria-valuetext="start of the route">
        <path class="fill" d=""></path>
        <path class="trace" d=""></path>
        <line class="baseline" x1="0" y1="${VIEW_H}" x2="${VIEW_W}" y2="${VIEW_H}"></line>
        <line class="cursor" y1="0" y2="${VIEW_H}" x1="0" x2="0"></line>
      </svg>
      <div class="tip" hidden></div>
    `;

    this.svg = root.querySelector("svg");
    this.fill = root.querySelector(".fill");
    this.trace = root.querySelector(".trace");
    this.cursorLine = root.querySelector(".cursor");
    this.tip = root.querySelector(".tip");

    this.svg.addEventListener("pointermove", (event) => this.probeFromPointer(event));
    this.svg.addEventListener("pointerdown", (event) => this.probeFromPointer(event));
    this.svg.addEventListener("pointerleave", () => this.clearProbe());

    // The chart is the only way to inspect a point on the route, so it cannot
    // be pointer-only. Exposed as a slider along the ride: arrows step, page
    // keys jump, home and end go to the ends.
    this.svg.addEventListener("keydown", (event) => this.probeFromKey(event));
    this.svg.addEventListener("blur", () => this.clearProbe());
  }

  /** Draw a route profile, or clear the chart when there is nothing to show. */
  render(elevation) {
    this.points = Array.isArray(elevation) ? elevation : [];
    this.root.classList.remove("drawn");

    if (this.points.length < 2) {
      this.fill.setAttribute("d", "");
      this.trace.setAttribute("d", "");
      this.total = 0;
      return;
    }

    this.total = this.points[this.points.length - 1].at_m;
    this.cursorAt = 0;
    this.svg.setAttribute("aria-valuemax", String(Math.round(this.total)));
    const heights = this.points.map((point) => point.height_m);
    const low = Math.min(...heights);
    const high = Math.max(...heights);
    // A flat ride would divide by zero and a nearly flat one would be drawn as
    // an alpine stage; a floor on the span keeps the shape honest.
    this.low = low;
    this.span = Math.max(20, high - low);

    const coords = this.points.map((point) => [this.xOf(point.at_m), this.yOf(point.height_m)]);
    const line = coords.map(
      ([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`
    );

    this.trace.setAttribute("d", line.join(" "));
    this.fill.setAttribute("d", `${line.join(" ")} L${VIEW_W} ${VIEW_H} L0 ${VIEW_H} Z`);

    // Next frame, so the clip-path transition has a start state to run from.
    requestAnimationFrame(() => this.root.classList.add("drawn"));
  }

  /** Move the cursor to a distance along the route without emitting an event. */
  showAt(distanceM) {
    if (this.points.length < 2 || distanceM === null || distanceM === undefined) {
      this.clearProbe(false);
      return;
    }

    const clamped = Math.min(this.total, Math.max(0, distanceM));
    this.cursorAt = clamped;
    const x = this.xOf(clamped);
    this.cursorLine.setAttribute("x1", String(x));
    this.cursorLine.setAttribute("x2", String(x));
    this.root.classList.add("probing");

    this.tip.hidden = false;
    this.tip.textContent = this.labelAt(clamped);
    this.tip.style.left = `${(x / VIEW_W) * 100}%`;

    // Announced, so the readout is available without seeing the tooltip.
    this.svg.setAttribute("aria-valuenow", String(Math.round(clamped)));
    this.svg.setAttribute("aria-valuetext", this.labelAt(clamped));
  }

  /** Hide the cursor, and by default tell the listener the pointer has gone. */
  clearProbe(notify = true) {
    this.root.classList.remove("probing");
    this.tip.hidden = true;
    if (notify) this.onProbe(null);
  }

  probeFromPointer(event) {
    if (this.points.length < 2) return;
    const box = this.svg.getBoundingClientRect();
    if (box.width <= 0) return;

    const share = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const distance = share * this.total;
    this.showAt(distance);
    this.onProbe(distance);
  }

  probeFromKey(event) {
    if (this.points.length < 2) return;

    const step = this.total / 50;
    const jump = this.total / 10;
    const current = this.cursorAt ?? 0;
    const moves = {
      ArrowRight: current + step,
      ArrowUp: current + step,
      ArrowLeft: current - step,
      ArrowDown: current - step,
      PageUp: current + jump,
      PageDown: current - jump,
      Home: 0,
      End: this.total,
    };

    if (!(event.key in moves)) return;
    event.preventDefault();

    const target = Math.min(this.total, Math.max(0, moves[event.key]));
    this.showAt(target);
    this.onProbe(target);
  }

  labelAt(distanceM) {
    const rise = Math.round(this.heightAt(distanceM));
    const along =
      distanceM < 1000 ? `${Math.round(distanceM)} m` : `${(distanceM / 1000).toFixed(1)} km`;
    return `${along} · ${rise >= 0 ? "+" : ""}${rise} m`;
  }

  heightAt(distanceM) {
    for (let i = 1; i < this.points.length; i += 1) {
      const previous = this.points[i - 1];
      const current = this.points[i];
      if (distanceM <= current.at_m) {
        const span = current.at_m - previous.at_m;
        const t = span <= 0 ? 0 : (distanceM - previous.at_m) / span;
        return previous.height_m + (current.height_m - previous.height_m) * t;
      }
    }
    return this.points[this.points.length - 1].height_m;
  }

  xOf(distanceM) {
    return this.total <= 0 ? 0 : (distanceM / this.total) * VIEW_W;
  }

  yOf(height) {
    const share = (height - this.low) / this.span;
    return VIEW_H - PAD_BOTTOM - share * (VIEW_H - PAD_TOP - PAD_BOTTOM);
  }
}
