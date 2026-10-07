"""The trail planner's ASGI app, for uvicorn."""

from planner_core.api.app import create_app
from trail.sport import TRAIL

app = create_app(TRAIL)
