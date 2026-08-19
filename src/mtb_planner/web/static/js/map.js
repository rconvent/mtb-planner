// The map, and the other half of the linked hover.
//
// MapLibre rather than Leaflet because hillshade and terrain are not decoration
// in a mountain-bike tool, they are the product: native raster-DEM hillshade,
// GPU line rendering for a four-thousand-point route, and full control of the
// style. Leaflet needs a plugin for each and draws to SVG at a visible cost.
//
// The route is drawn as one feature per edge rather than a single line with a
// gradient, because a dash pattern cannot vary along one feature — and the dash
// is what marks difficulty that was inferred rather than surveyed. Two layers
// over one source, filtered on that flag, plus a halo underneath so the line
// reads against both paper and hillshade.
//
// The draw-in is a filter on distance along the route rather than an opacity
// fade, so the line grows the way a ride is ridden. It is skipped outright when
// the viewer has asked for reduced motion.

import { cssColour, techColourExpression } from "./ramp.js";
import { bounds, nearestDistance, positionAt, routeFeature, segmentFeatures } from "./route.js";

const SEGMENTS = "route-segments";
const HALO = "route-halo";
const NETWORK = "trail-network";
const DRAW_MS = 600;

// Way classes, from features/enums.py. Duplicated because the map has to style
// by class and a paint expression cannot import a Python enum; a test asserts
// the two agree.
const CLASS = {
  SINGLETRACK: 0,
  DOUBLETRACK: 1,
  FIREROAD: 2,
  GRAVEL_ROAD: 3,
  PAVED_MINOR: 4,
  PAVED_MAJOR: 5,
  URBAN: 6,
  STEPS: 7,
  FERRY: 8,
  UNKNOWN: 9,
};

const TRAIL_CLASSES = [
  CLASS.SINGLETRACK,
  CLASS.DOUBLETRACK,
  CLASS.FIREROAD,
  CLASS.GRAVEL_ROAD,
  CLASS.STEPS,
];

const REDUCED_MOTION =
  typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

export class RouteMap {
  /**
   * @param {string} containerId element to mount into
   * @param {object} options {styleUrl, center, zoom, onPick, onProbe}
   */
  constructor(containerId, options = {}) {
    this.onPick = options.onPick ?? (() => {});
    this.onProbe = options.onProbe ?? (() => {});
    this.onNetwork = options.onNetwork ?? (() => {});
    this.route = null;
    this.ready = false;
    this.networkTimer = 0;

    this.map = new maplibregl.Map({
      container: containerId,
      style: options.styleUrl,
      center: options.center ?? [6.13, 49.61],
      zoom: options.zoom ?? 11,
      attributionControl: { compact: true },
    });

    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    this.map.addControl(
      new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }),
      "bottom-left"
    );

    this.marker = new maplibregl.Marker({ color: readToken("--ink") });
    this.endMarker = new maplibregl.Marker({ color: readToken("--tech-5") });
    this.endAttached = false;
    this.probeMarker = new maplibregl.Marker({ color: readToken("--accent") });
    this.probeAttached = false;

    this.map.on("load", () => {
      this.addLayers();
      this.ready = true;
      this.refreshNetwork();
      this.map.on("moveend", () => this.refreshNetwork());
      this.map.on("click", (event) => this.onPick(event.lngLat));
      this.map.on("mousemove", `${SEGMENTS}-surveyed`, (event) => this.reportProbe(event.lngLat));
      this.map.on("mousemove", `${SEGMENTS}-estimated`, (event) => this.reportProbe(event.lngLat));
      this.map.on("mouseleave", `${SEGMENTS}-surveyed`, () => this.onProbe(null));
      this.map.on("mouseleave", `${SEGMENTS}-estimated`, () => this.onProbe(null));
    });
  }

  /**
   * Whether a basemap is drawn underneath.
   *
   * It changes what the trail overlay is *for*. With no basemap the overlay is
   * the map, and it has to carry roads as well so a rider can orient at all.
   * With one, the roads are already there and better drawn, so repeating them
   * is just a second set of lines over the first.
   */
  hasBasemap() {
    return Boolean(this.map.getSource("basemap"));
  }

  addLayers() {
    const empty = { type: "FeatureCollection", features: [] };
    this.map.addSource(NETWORK, { type: "geojson", data: empty });
    this.map.addSource(SEGMENTS, { type: "geojson", data: empty });
    this.map.addSource(HALO, { type: "geojson", data: empty });

    this.addNetworkLayers();

    // Under the line, so it reads against paper and hillshade alike.
    this.map.addLayer({
      id: HALO,
      type: "line",
      source: HALO,
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": readToken("--paper-raised"),
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 6, 15, 10],
        "line-opacity": 0.85,
      },
    });

    const width = ["interpolate", ["linear"], ["zoom"], 10, 2.5, 15, 5];

    this.map.addLayer({
      id: `${SEGMENTS}-surveyed`,
      type: "line",
      source: SEGMENTS,
      filter: ["==", ["get", "surveyed"], true],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": techColourExpression("tech"), "line-width": width },
    });

    // Dashed where the difficulty was inferred rather than mapped. At a few
    // per cent survey coverage, drawing the two identically would claim more
    // than the data supports.
    this.map.addLayer({
      id: `${SEGMENTS}-estimated`,
      type: "line",
      source: SEGMENTS,
      filter: ["!=", ["get", "surveyed"], true],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: {
        "line-color": techColourExpression("tech"),
        "line-width": width,
        "line-dasharray": [2, 1.4],
      },
    });
  }

  /**
   * The trail network itself, under everything else.
   *
   * Two layers, and the split is the whole point of the design direction: roads
   * are hairlines that exist only to orient, trails are drawn properly because
   * they are the subject. Every general-purpose map does the opposite.
   */
  addNetworkLayers() {
    // Roads only when nothing else is drawing them.
    this.map.addLayer({
      id: `${NETWORK}-roads`,
      type: "line",
      source: NETWORK,
      filter: ["!", ["in", ["get", "class"], ["literal", TRAIL_CLASSES]]],
      layout: {
        "line-cap": "round",
        "line-join": "round",
        visibility: this.hasBasemap() ? "none" : "visible",
      },
      paint: {
        "line-color": readToken("--ink-faint"),
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.4, 16, 1.2],
        "line-opacity": 0.55,
      },
    });

    this.map.addLayer({
      id: `${NETWORK}-trails`,
      type: "line",
      source: NETWORK,
      filter: ["in", ["get", "class"], ["literal", TRAIL_CLASSES]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": readToken("--ink-muted"),
        // Singletrack thinner than a fire road, as on the ground.
        "line-width": [
          "interpolate",
          ["linear"],
          ["zoom"],
          10,
          ["case", ["==", ["get", "class"], CLASS.SINGLETRACK], 0.6, 0.9],
          16,
          ["case", ["==", ["get", "class"], CLASS.SINGLETRACK], 1.6, 2.4],
        ],
        // Lighter over a basemap: there the overlay is emphasis on trails the
        // ground already shows, not the only record of them.
        "line-opacity": this.hasBasemap() ? 0.42 : 0.8,
      },
    });

    // A signed mountain-bike route is somebody physically marking the ground,
    // which is worth more than anything inferred, so it gets its own weight.
    this.map.addLayer({
      id: `${NETWORK}-signed`,
      type: "line",
      source: NETWORK,
      filter: ["==", ["get", "signed"], true],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: {
        "line-color": readToken("--ink"),
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.9, 16, 2.6],
        "line-opacity": this.hasBasemap() ? 0.5 : 0.7,
      },
    });
  }

  /**
   * Fetch the trails in view.
   *
   * Debounced, and skipped while the map is still moving, because a pan fires
   * this continuously and the payload is the largest the app requests.
   */
  refreshNetwork() {
    if (!this.ready) return;
    clearTimeout(this.networkTimer);
    this.networkTimer = setTimeout(() => {
      const bounds = this.map.getBounds();
      const zoom = Math.round(this.map.getZoom());
      const query = new URLSearchParams({
        west: bounds.getWest().toFixed(5),
        south: bounds.getSouth().toFixed(5),
        east: bounds.getEast().toFixed(5),
        north: bounds.getNorth().toFixed(5),
        zoom: String(zoom),
      });

      fetch(`/api/network?${query}`)
        .then((response) => {
          if (!response.ok) return null;
          // Surfaced, not swallowed: a thinned map looks exactly like a gap in
          // the trail network, and that is the wrong thing to believe.
          this.onNetwork({
            drawn: Number(response.headers.get("X-Network-Drawn") || 0),
            total: Number(response.headers.get("X-Network-Total") || 0),
            truncated: response.headers.get("X-Network-Truncated") === "1",
          });
          return response.json();
        })
        .then((data) => {
          if (data && this.map.getSource(NETWORK)) {
            this.map.getSource(NETWORK).setData(data);
          }
        })
        .catch(() => {
          // A missing basemap is a degraded map, not a broken one: the route is
          // still drawn and still correct, so this stays quiet.
        });
    }, 250);
  }

  /** Draw a route, fit the view to it, and animate it in. */
  show(route, { fit = true } = {}) {
    if (!this.ready) {
      this.map.once("load", () => this.show(route, { fit }));
      return;
    }

    this.route = route;
    this.map.getSource(SEGMENTS).setData(segmentFeatures(route));
    this.map.getSource(HALO).setData(routeFeature(route));

    const start = route?.geometry?.coordinates?.[0];
    if (start) this.marker.setLngLat(start).addTo(this.map);

    if (fit) {
      const box = bounds(route);
      if (box) {
        // Padded away from the rail and the dock, so the route is not drawn
        // under a panel.
        this.map.fitBounds(box, {
          padding: { top: 60, right: 380, bottom: 60, left: 380 },
          duration: REDUCED_MOTION ? 0 : 700,
        });
      }
    }

    this.drawIn(route);
  }

  /** Reveal the line progressively, as a filter on distance along the route. */
  drawIn(route) {
    const total = route?.stats?.distance_m ?? 0;
    const layers = [`${SEGMENTS}-surveyed`, `${SEGMENTS}-estimated`];

    if (REDUCED_MOTION || total <= 0) {
      this.setReveal(layers, Infinity);
      return;
    }

    const began = performance.now();
    const step = () => {
      const elapsed = performance.now() - began;
      const share = Math.min(1, elapsed / DRAW_MS);
      // Ease out, so the line lands rather than stopping dead.
      const eased = 1 - (1 - share) ** 3;
      this.setReveal(layers, eased * total);
      if (share < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  setReveal(layers, upTo) {
    for (const id of layers) {
      if (!this.map.getLayer(id)) continue;
      const base = id.endsWith("surveyed")
        ? ["==", ["get", "surveyed"], true]
        : ["!=", ["get", "surveyed"], true];
      this.map.setFilter(
        id,
        Number.isFinite(upTo) ? ["all", base, ["<=", ["get", "at_m"], upTo]] : base
      );
    }
  }

  /** Put the probe marker at a distance along the route. */
  showProbeAt(distanceM) {
    if (!this.route || distanceM === null || distanceM === undefined) {
      if (this.probeAttached) {
        this.probeMarker.remove();
        this.probeAttached = false;
      }
      return;
    }

    const point = positionAt(this.route, distanceM);
    if (!point) return;
    this.probeMarker.setLngLat([point.lon, point.lat]);
    if (!this.probeAttached) {
      this.probeMarker.addTo(this.map);
      this.probeAttached = true;
    }
  }

  reportProbe(lngLat) {
    if (!this.route) return;
    this.onProbe(nearestDistance(this.route, { lon: lngLat.lng, lat: lngLat.lat }));
  }

  /**
   * Show or hide the finish pin.
   *
   * A loop finishes where it starts, so a second pin there would be a lie about
   * the shape of the ride.
   */
  showFinish(point) {
    if (!point) {
      if (this.endAttached) {
        this.endMarker.remove();
        this.endAttached = false;
      }
      return;
    }
    this.endMarker.setLngLat([point.lon, point.lat]);
    if (!this.endAttached) {
      this.endMarker.addTo(this.map);
      this.endAttached = true;
    }
  }

  /** Clear the drawn route. */
  clear() {
    if (!this.ready) return;
    const empty = { type: "FeatureCollection", features: [] };
    this.map.getSource(SEGMENTS).setData(empty);
    this.map.getSource(HALO).setData(empty);
    this.route = null;
    this.showProbeAt(null);
  }
}

/**
 * Read a design token, converted to something the map can paint with.
 *
 * The tokens are oklch, which MapLibre rejects outright — it throws on the
 * layer and the route silently does not draw. Converting here keeps one palette
 * in CSS and lets the canvas have it in sRGB.
 */
function readToken(name) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return cssColour(value, "black");
}
