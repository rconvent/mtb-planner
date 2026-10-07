"""The `mtb` command line entry point.

Driven in-process through typer's runner: spawning subprocesses would need
`COVERAGE_PROCESS_START` plumbing on Windows for no added confidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mtb_planner.cli import app
from planner_core import __version__

pytestmark = pytest.mark.integration

runner = CliRunner()


def test_version_flag_reports_the_package_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_bare_invocation_shows_help_rather_than_failing_silently():
    result = runner.invoke(app, [])

    assert "serve" in result.stdout


def test_serve_command_is_registered():
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--reload" in result.stdout


# --------------------------------------------------------------- documentation

README = Path(__file__).resolve().parents[2] / "README.md"

#: Commands whose first positional argument names a subcommand rather than a
#: value, so `mtb fetch osm` has to be resolved as one unit.
_GROUPS = {"fetch"}

#: How a README line introduces a command.
_PREFIX = "uv run mtb "


def documented_invocations() -> list[list[str]]:
    """Every `uv run mtb ...` line in the README, as argument lists."""
    found = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith(_PREFIX):
            continue
        body = line[len(_PREFIX) :].split("#")[0].split("`")[0]
        parts = body.split()
        if parts and parts != ["--help"]:
            found.append(parts)
    return found


def _help_for(parts: list[str]) -> str:
    """Help text for the command an invocation names."""
    command = [parts[0]]
    if parts[0] in _GROUPS and len(parts) > 1 and not parts[1].startswith("-"):
        command.append(parts[1])
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, f"mtb {' '.join(command)} is not a command"
    return result.stdout


def test_the_readme_documents_some_commands():
    # If the parser stops finding anything, the tests below pass vacuously.
    assert len(documented_invocations()) >= 8


def test_every_command_the_readme_shows_exists():
    for parts in documented_invocations():
        _help_for(parts)


def test_every_option_the_readme_shows_exists():
    """The README is the first thing anyone runs, so a fictional flag is a bug.

    This has caught two: `mtb inspect --tag-stats`, which was designed and never
    built, and `mtb fetch osm --config`, where the regions are positional. Both
    failed on the first command a new reader typed.
    """
    missing = []
    for parts in documented_invocations():
        # Typer wraps help at the terminal width, so compare without spacing.
        flat = "".join(_help_for(parts).split())
        for flag in (part for part in parts if part.startswith("--")):
            if flag not in flat:
                missing.append(f"mtb {parts[0]}: {flag}")

    assert not missing, "options in README.md that do not exist: " + "; ".join(missing)


def test_no_documented_command_hides_a_placeholder():
    """Every command in the README has to run exactly as written.

    A block of shell lines gets copy-pasted, so a placeholder that looks like a
    real value is a trap. `--bbox W,S,E,N` was one: the flag exists, so the
    option check above passed it happily, and it failed only when somebody ran
    it.

    Placeholders read as bare upper-case tokens, which no real region slug,
    path, coordinate or flag in this project uses.
    """
    suspects = []
    for parts in documented_invocations():
        for token in parts:
            if token.startswith("-"):
                continue
            letters = [character for character in token if character.isalpha()]
            if letters and all(character.isupper() for character in letters):
                suspects.append(f"{' '.join(parts)}  ->  {token}")

    assert not suspects, "placeholders in runnable README commands: " + "; ".join(suspects)


def test_no_documented_command_depends_on_unset_configuration():
    """A documented command must not need an environment variable to be set.

    `mtb route --from ... --to ...` was documented without `--artifact`, so it
    fell back to the configured default and failed with "No routing artifact at
    data/graph/aoi-v1" for anyone who had not exported MTB_ARTIFACT. The flag
    existed and the arguments parsed, so both checks above passed it; it only
    failed when run.

    Any command that loads a graph therefore has to say which one, in the line
    itself.
    """
    needs_a_graph = {"route", "inspect"}
    silent = []

    for parts in documented_invocations():
        if parts[0] not in needs_a_graph:
            continue
        names_one = any(
            token in {"--artifact", "--pbf"} or token.startswith("data") for token in parts
        )
        if not names_one:
            silent.append(" ".join(parts))

    assert not silent, "these README commands rely on MTB_ARTIFACT being set: " + "; ".join(silent)


def test_a_missing_artifact_names_the_ones_that_exist(tmp_path):
    """The failure has to carry its own fix.

    Whatever the docs say, somebody will point the server at an artifact that is
    not there. Listing what is beside it turns a dead end into a one-line
    correction, and it is worded once so the CLI and the server agree.
    """
    from planner_core.core.errors import ArtifactNotFoundError
    from planner_core.graph.manifest import read_manifest

    built = tmp_path / "lux-v1"
    built.mkdir()
    (built / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactNotFoundError) as caught:
        read_manifest(tmp_path / "aoi-v1")

    message = str(caught.value)
    assert "lux-v1" in message
    assert "--artifact" in message or "MTB_ARTIFACT" in message


def test_a_missing_artifact_with_nothing_built_says_to_build(tmp_path):
    from planner_core.core.errors import ArtifactNotFoundError
    from planner_core.graph.manifest import read_manifest

    with pytest.raises(ArtifactNotFoundError, match="mtb build"):
        read_manifest(tmp_path / "aoi-v1")


def invocations_by_block() -> list[list[list[str]]]:
    """Documented commands, grouped by the fenced block they appear in."""
    blocks: list[list[list[str]]] = []
    inside = False
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            inside = not inside
            if inside:
                blocks.append([])
            continue
        if inside and line.startswith(_PREFIX):
            body = line[len(_PREFIX) :].split("#")[0].split("`")[0]
            parts = body.split()
            if parts and parts != ["--help"]:
                blocks[-1].append(parts)
    return [block for block in blocks if block]


def test_no_block_offers_two_ways_of_doing_the_same_thing():
    """A block of shell lines gets run wholesale, so alternatives cost real work.

    Listing `fetch osm europe/luxembourg` next to a bare `fetch osm` reads as
    "either of these" to a person and "both of these" to a script. Somebody ran
    it and downloaded one extract followed by all seven, then four DEM tiles
    followed by forty-two.

    Repeating a command identically in a later block is fine; offering two
    different forms of it inside one runnable block is not.
    """
    clashes = []
    for block in invocations_by_block():
        seen: dict[str, set[str]] = {}
        for parts in block:
            key = " ".join(parts[:2]) if parts[0] in _GROUPS else parts[0]
            seen.setdefault(key, set()).add(" ".join(parts))
        clashes += [f"{key}: {sorted(uses)}" for key, uses in seen.items() if len(uses) > 1]

    assert not clashes, "one runnable block offers competing commands: " + "; ".join(clashes)


def test_the_readme_shows_more_than_one_area():
    """One worked example is a recipe; two show which parts are arguments.

    They have to be separate blocks. Belgium beside Luxembourg in the same
    runnable block would download 692 MB on top of 47 MB for anyone who ran it
    top to bottom, which is the mistake `test_no_block_offers_two_ways_of_doing_
    the_same_thing` exists to prevent.
    """
    regions = set()
    for block in invocations_by_block():
        for parts in block:
            if parts[:2] == ["fetch", "osm"] and len(parts) > 2:
                regions.add(parts[2])

    assert len(regions) >= 2, f"only one area is documented: {regions}"


def test_serving_does_not_require_an_environment_variable():
    # `mtb route` took --artifact from the start; `mtb serve` did not, so the
    # only way to serve a built graph was to remember to export MTB_ARTIFACT.
    result = runner.invoke(app, ["serve", "--help"])

    assert "--artifact" in result.stdout


def test_the_readme_names_the_artifact_environment_variable():
    # The most common way to end up with a server that starts but cannot plan.
    assert "MTB_ARTIFACT" in README.read_text(encoding="utf-8")
