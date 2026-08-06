"""Deriving rideable character from OSM tags.

Explicit technicality tagging covers a small minority of the network — about
5% of Luxembourg's trail ways and 3.4% of Belgium's — so inference from
surface, tracktype, smoothness and width is the primary engine here, not a
fallback. Every derived difficulty carries a confidence level so the interface
can distinguish surveyed from estimated rather than implying knowledge it does
not have.
"""
