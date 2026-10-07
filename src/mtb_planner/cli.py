"""The `mtb` command line."""

from mtb_planner.sport import MTB
from planner_core.cli.main import build_cli

app = build_cli(MTB)

if __name__ == "__main__":
    app()
