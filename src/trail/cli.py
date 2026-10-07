"""The `trail` command line."""

from planner_core.cli.main import build_cli
from trail.sport import TRAIL

app = build_cli(TRAIL)

if __name__ == "__main__":
    app()
