"""Geofabrik region catalogue.

Region slugs are the path Geofabrik itself uses, so any region on the site
works without this file having to enumerate the world. The default set below
covers the area of interest: Benelux as one file — which avoids a seam between
Belgium and Luxembourg — plus the French and German border strips that carry
the Ardennes network across the frontier.
"""

from __future__ import annotations

from dataclasses import dataclass

GEOFABRIK_BASE = "https://download.geofabrik.de"

#: Geofabrik publishes no combined Benelux extract — `europe/benelux` returns
#: the download index page with a 200, which is how it survived here as a
#: nine-kilobyte HTML file named `.pbf`. The three countries are separate
#: downloads, and the seam between them is handled the same way as every other
#: border: dedupe ways by OSM id, keep the copy with the most node refs, and
#: clip once geometry is assembled.
#:
#: The remaining regions contribute only their border strips after the bounding
#: box is applied, but must be whole files because that is how they are
#: published.
DEFAULT_AOI_REGIONS: tuple[str, ...] = (
    "europe/belgium",
    "europe/netherlands",
    "europe/luxembourg",
    "europe/france/nord-pas-de-calais",
    "europe/france/lorraine",
    "europe/france/champagne-ardenne",
    "europe/germany/nordrhein-westfalen",
    "europe/germany/rheinland-pfalz",
    "europe/germany/saarland",
)

#: Small enough to download quickly, real enough to prove the pipeline.
SMOKE_TEST_REGION = "europe/luxembourg"


@dataclass(frozen=True)
class Region:
    """A downloadable Geofabrik extract."""

    slug: str

    def __post_init__(self) -> None:
        if not self.slug or self.slug.startswith("/") or self.slug.endswith("/"):
            msg = f"Invalid region slug: {self.slug!r}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Last path component, e.g. `belgium`."""
        return self.slug.rsplit("/", 1)[-1]

    @property
    def filename(self) -> str:
        """Local file name for the extract."""
        return f"{self.name}-latest.osm.pbf"

    @property
    def url(self) -> str:
        """Download URL for the latest extract."""
        return f"{GEOFABRIK_BASE}/{self.slug}-latest.osm.pbf"

    @property
    def md5_url(self) -> str:
        """URL of the published checksum sidecar."""
        return f"{self.url}.md5"
