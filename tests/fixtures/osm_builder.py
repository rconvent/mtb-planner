"""Write small OSM files for tests.

Produces both `.osm` XML and `.osm.pbf`. The XML form is what makes a fixture
reviewable; the PBF form is what the production reader actually consumes, and
testing against anything else would leave the real code path unexercised.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import osmium
from osmium.osm import osm_entity_bits as EntityBits


@dataclass(frozen=True)
class FixtureNode:
    """A node in a synthetic extract."""

    id: int
    lat: float
    lon: float
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FixtureWay:
    """A way in a synthetic extract."""

    id: int
    nodes: Sequence[int]
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FixtureRelation:
    """A relation in a synthetic extract. Members are `(type, ref, role)`."""

    id: int
    members: Sequence[tuple[str, int, str]]
    tags: Mapping[str, str] = field(default_factory=dict)


def _add_tags(parent: ET.Element, tags: Mapping[str, str]) -> None:
    for key, value in tags.items():
        ET.SubElement(parent, "tag", {"k": key, "v": value})


def write_osm_xml(
    path: Path,
    nodes: Sequence[FixtureNode],
    ways: Sequence[FixtureWay] = (),
    relations: Sequence[FixtureRelation] = (),
) -> Path:
    """Write an OSM 0.6 XML document."""
    root = ET.Element("osm", {"version": "0.6", "generator": "mtb-planner-tests"})

    for node in nodes:
        element = ET.SubElement(
            root,
            "node",
            {"id": str(node.id), "lat": f"{node.lat:.7f}", "lon": f"{node.lon:.7f}"},
        )
        _add_tags(element, node.tags)

    for way in ways:
        element = ET.SubElement(root, "way", {"id": str(way.id)})
        for ref in way.nodes:
            ET.SubElement(element, "nd", {"ref": str(ref)})
        _add_tags(element, way.tags)

    for relation in relations:
        element = ET.SubElement(root, "relation", {"id": str(relation.id)})
        for member_type, ref, role in relation.members:
            ET.SubElement(element, "member", {"type": member_type, "ref": str(ref), "role": role})
        _add_tags(element, relation.tags)

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def xml_to_pbf(xml_path: Path, pbf_path: Path) -> Path:
    """Convert an OSM XML file to `.osm.pbf`."""
    if pbf_path.exists():
        pbf_path.unlink()
    pbf_path.parent.mkdir(parents=True, exist_ok=True)

    writer = osmium.SimpleWriter(str(pbf_path))
    try:
        entities = EntityBits.NODE | EntityBits.WAY | EntityBits.RELATION
        for obj in osmium.FileProcessor(str(xml_path), entities):
            writer.add(obj)
    finally:
        writer.close()
    return pbf_path


def write_fixture_pbf(
    directory: Path,
    stem: str,
    nodes: Sequence[FixtureNode],
    ways: Sequence[FixtureWay] = (),
    relations: Sequence[FixtureRelation] = (),
) -> Path:
    """Write a fixture as XML and convert it to PBF, returning the PBF path."""
    xml_path = write_osm_xml(directory / f"{stem}.osm", nodes, ways, relations)
    return xml_to_pbf(xml_path, directory / f"{stem}.osm.pbf")
