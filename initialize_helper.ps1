uv run mtb fetch osm europe/belgium
uv run mtb fetch dem --bbox 2.5,49.4,6.5,51.6
uv run mtb build --pbf data\osm\belgium-latest.osm.pbf --artifact be-v1
uv run mtb inspect data\graph\be-v1
uv run mtb serve --artifact be-v1
# Trail running reuses the same downloads; only the graph is built separately.
# uv run trail build --pbf data\osm\belgium-latest.osm.pbf --artifact be-v1
# uv run trail serve --artifact be-v1
