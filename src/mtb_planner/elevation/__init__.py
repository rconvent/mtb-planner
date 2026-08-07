"""Elevation: sampling a digital elevation model onto the routing graph.

Climb is what makes a time budget mean anything — two hours of Ardennes
singletrack covers a fraction of two hours of Flemish gravel — so this is the
input the speed model rests on.

It is also the easiest thing in the project to get quietly, badly wrong. A
30 m raster read straight along a way reports the valley floor under every
viaduct and turns sensor noise into climbing that was never ridden. The
smoothing, dead-band and bridge handling here are not refinements; without
them a flat 40 km ride reports around a kilometre of ascent.
"""
