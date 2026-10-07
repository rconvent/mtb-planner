"""Logging configuration.

`build_config` is pure, so it can be asserted on without mutating global
logging state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from planner_core.config.logging import build_config, get_logger

pytestmark = pytest.mark.unit


def test_console_only_by_default():
    config = build_config(level="INFO")

    assert set(config["handlers"]) == {"console"}
    assert config["loggers"]["planner_core"]["level"] == "INFO"


def test_a_file_handler_is_added_when_a_path_is_given():
    config = build_config(level="DEBUG", log_file=Path("out/mtb.log"))

    assert set(config["handlers"]) == {"console", "file"}
    assert config["handlers"]["file"]["filename"] == str(Path("out/mtb.log"))
    assert config["loggers"]["planner_core"]["handlers"] == ["console", "file"]


def test_package_logger_does_not_propagate_to_root():
    # Otherwise every record would be emitted twice.
    assert build_config()["loggers"]["planner_core"]["propagate"] is False


def test_get_logger_namespaces_bare_module_names():
    assert get_logger("ingest.pbf_reader").name == "planner_core.ingest.pbf_reader"


def test_get_logger_leaves_already_namespaced_names_alone():
    assert get_logger("planner_core.graph.builder").name == "planner_core.graph.builder"
