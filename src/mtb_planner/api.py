"""The bike planner's ASGI app, for uvicorn."""

from mtb_planner.sport import MTB
from planner_core.api.app import create_app

app = create_app(MTB)
