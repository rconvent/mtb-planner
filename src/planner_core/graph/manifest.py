"""The artifact manifest and its compatibility gate.

A server running a newer cost model against an artifact built by an older
feature pipeline does not crash — it produces plausible-looking routes that
are quietly wrong. That failure has no symptom until a rider notices the
numbers are nonsense, so the two versions are checked at load and a mismatch
refuses to start.

`schema_version` covers the on-disk layout: which arrays exist and what they
mean. `cost_model_version` covers how edge attributes are interpreted. They
move independently, because adding an array and changing a weighting are
different kinds of break. The cost model belongs to a sport, so the manifest
also records which sport built the graph: a runner's graph includes footways a
rider may not use, and serving it to a rider would route them illegally.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from planner_core import ARTIFACT_SCHEMA_VERSION, __version__
from planner_core.core.errors import ArtifactNotFoundError, IncompatibleArtifactError
from planner_core.sport import Sport, active_sport

MANIFEST_FILENAME = "manifest.json"

#: UTC, second resolution, `Z` suffix. Stable across platforms and sortable.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_timestamp() -> str:
    """Current time as an artifact timestamp."""
    return datetime.now(UTC).strftime(TIMESTAMP_FORMAT)


class SourceExtract(BaseModel):
    """One input file that contributed to a build."""

    model_config = ConfigDict(frozen=True)

    name: str
    bytes: int = Field(ge=0)
    md5: str | None = None


class GraphCounts(BaseModel):
    """Shape of the built graph, for reporting and sanity checks."""

    model_config = ConfigDict(frozen=True)

    nodes: int = Field(ge=0)
    edges: int = Field(ge=0)
    arcs: int = Field(ge=0)
    allowed_arcs: int = Field(ge=0)
    geom_points: int = Field(ge=0)
    giant_scc_nodes: int = Field(ge=0)
    giant_scc_edges: int = Field(ge=0)

    @property
    def giant_scc_edge_fraction(self) -> float:
        """Share of edges reachable within the main component."""
        return self.giant_scc_edges / self.edges if self.edges else 0.0


class Manifest(BaseModel):
    """Everything needed to decide whether an artifact may be loaded."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    cost_model_version: int
    #: Artifacts built before the planner served more than one sport carry no
    #: sport, and every one of them was built for bikes.
    sport: str = "mtb"
    artifact: str
    planner_version: str
    built_at: str = Field(description=f"UTC, formatted {TIMESTAMP_FORMAT}.")
    bbox: tuple[float, float, float, float] = Field(
        description="min_lon, min_lat, max_lon, max_lat."
    )
    counts: GraphCounts
    giant_scc: int
    contracted: bool
    sources: tuple[SourceExtract, ...] = ()
    tech_coverage: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Share of network length at each difficulty-confidence level, keyed by "
            "the lower-cased level name. The headline measure of how much of the "
            "model is inference rather than survey."
        ),
    )

    total_ascent_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Summed climb across every edge, in metres. Zero when built flat.",
    )

    @property
    def has_elevation(self) -> bool:
        """Whether the artifact was built against an elevation model."""
        return self.total_ascent_m > 0.0

    @property
    def informed_fraction(self) -> float:
        """Share of length whose difficulty rests on more than a way class."""
        return sum(share for level, share in self.tech_coverage.items() if level != "default")

    def is_compatible(self, sport: Sport | None = None) -> bool:
        """Whether this build of the planner can interpret the artifact."""
        return self.incompatibility(sport) is None

    def incompatibility(self, sport: Sport | None = None) -> str | None:
        """Describe the mismatch, or `None` when the artifact is usable."""
        sport = sport or active_sport()
        if self.sport != sport.id:
            return f"artifact was built for {self.sport}, not {sport.id}"
        problems = []
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            problems.append(
                f"artifact schema v{self.schema_version} "
                f"but this build reads v{ARTIFACT_SCHEMA_VERSION}"
            )
        if self.cost_model_version != sport.cost_model_version:
            problems.append(
                f"artifact cost model v{self.cost_model_version} "
                f"but this build expects v{sport.cost_model_version}"
            )
        return "; ".join(problems) or None

    def require_compatible(self, sport: Sport | None = None) -> None:
        """Raise unless this build can interpret the artifact."""
        sport = sport or active_sport()
        problem = self.incompatibility(sport)
        if problem is not None:
            msg = f"Cannot load {self.artifact}: {problem}. Rebuild with `{sport.cli} build`."
            raise IncompatibleArtifactError(msg)

    def write(self, directory: Path) -> Path:
        """Write the manifest into an artifact directory."""
        path = directory / MANIFEST_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def build_manifest(
    artifact: str,
    bbox: tuple[float, float, float, float],
    counts: GraphCounts,
    giant_scc: int,
    contracted: bool,
    sources: tuple[SourceExtract, ...] = (),
    tech_coverage: dict[str, float] | None = None,
    total_ascent_m: float = 0.0,
    sport: Sport | None = None,
) -> Manifest:
    """Create a manifest stamped with the current versions and time."""
    sport = sport or active_sport()
    return Manifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        cost_model_version=sport.cost_model_version,
        sport=sport.id,
        artifact=artifact,
        planner_version=__version__,
        built_at=utc_timestamp(),
        bbox=bbox,
        counts=counts,
        giant_scc=giant_scc,
        contracted=contracted,
        sources=sources,
        tech_coverage=tech_coverage or {},
        total_ascent_m=total_ascent_m,
    )


def _sibling_artifacts(directory: Path) -> list[str]:
    """Names of built artifacts beside the one that was asked for.

    The configured artifact name is a guess about what somebody will build. When
    it guesses wrong, naming the alternatives turns a dead end into a one-line
    fix, and it belongs here so the CLI and the server say the same thing.
    """
    parent = directory.parent
    if not parent.is_dir():
        return []
    return sorted(child.name for child in parent.iterdir() if (child / MANIFEST_FILENAME).is_file())


def resolve_artifact_dir(configured: Path) -> Path:
    """The artifact to actually load, given the one that was asked for.

    The configured name is a guess about what somebody will build, and it
    defaults to a name nobody has. Refusing to serve the only built graph on the
    machine because it is called something else is pedantry: it produced an
    interface that loaded, listed the profiles, and answered every plan with 503.

    So: use the configured path if it is built. If it is not, and exactly one
    artifact sits beside it, use that. **Ambiguity is never guessed** — with two
    or more the configured path comes back unchanged, and the caller fails with
    the list, because silently picking one of several is how somebody plans a
    route against last month`s graph without noticing.

    Callers compare the result with what they passed in to decide whether to say
    a substitution happened. They should.
    """
    if (configured / MANIFEST_FILENAME).is_file():
        return configured

    available = _sibling_artifacts(configured)
    if len(available) == 1:
        return configured.parent / available[0]
    return configured


def read_manifest(directory: Path) -> Manifest:
    """Read and validate a manifest without loading the arrays."""
    path = directory / MANIFEST_FILENAME
    if not path.exists():
        available = _sibling_artifacts(directory)
        sport = active_sport()
        prefix = sport.settings_cls.model_config.get("env_prefix", "")
        hint = (
            f" Built artifacts: {', '.join(available)}. Pass --artifact, or set {prefix}ARTIFACT."
            if available
            else f" Nothing is built yet; run `{sport.cli} build` first."
        )
        msg = f"No routing artifact at {directory} (missing {MANIFEST_FILENAME}).{hint}"
        raise ArtifactNotFoundError(msg)
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, json.JSONDecodeError) as exc:
        msg = f"Manifest at {path} is not readable: {exc}"
        raise IncompatibleArtifactError(msg) from exc
