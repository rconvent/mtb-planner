"""Physical constants and unit conversions.

Named constants only — no magic numbers scattered through the pipeline.
Internally the project is strictly SI: metres, seconds, metres per second.
Conversions happen at the API and CLI boundaries, nowhere else.
"""

from __future__ import annotations

from typing import Final

# --- geodesy -------------------------------------------------------------

#: Mean Earth radius (IUGG). Adequate for trail-scale distances; the routing
#: graph stores real geometry, so this only ever measures short segments.
EARTH_RADIUS_M: Final = 6_371_008.8

#: OpenStreetMap stores coordinates as integers in 1e-7 degrees. The graph
#: uses the same encoding: exact, and 4 bytes like float32 but without the
#: ~0.4 m error float32 introduces at Belgian latitudes.
COORD_SCALE: Final = 1e7

# --- physics -------------------------------------------------------------

GRAVITY_M_S2: Final = 9.81
AIR_DENSITY_KG_M3: Final = 1.225

# --- speed bounds --------------------------------------------------------

#: Absolute floor, roughly a slow hike-a-bike push.
MIN_SPEED_M_S: Final = 0.6

#: Absolute ceiling (~43 km/h). Caps the descent model, and gives the A*
#: heuristic an admissible lower bound on traversal time.
MAX_SPEED_M_S: Final = 12.0

#: Enforced speed when a way is rideable only on foot (`bicycle=dismount`,
#: or technicality above the rider's declared ceiling).
DISMOUNT_SPEED_M_S: Final = 1.1

# --- conversions ---------------------------------------------------------

SECONDS_PER_MINUTE: Final = 60
SECONDS_PER_HOUR: Final = 3600
METRES_PER_KM: Final = 1000


def km_to_m(km: float) -> float:
    """Kilometres to metres."""
    return km * METRES_PER_KM


def m_to_km(m: float) -> float:
    """Metres to kilometres."""
    return m / METRES_PER_KM


def minutes_to_s(minutes: float) -> float:
    """Minutes to seconds."""
    return minutes * SECONDS_PER_MINUTE


def s_to_minutes(seconds: float) -> float:
    """Seconds to minutes."""
    return seconds / SECONDS_PER_MINUTE


def m_s_to_km_h(m_s: float) -> float:
    """Metres per second to kilometres per hour."""
    return m_s * SECONDS_PER_HOUR / METRES_PER_KM


def km_h_to_m_s(km_h: float) -> float:
    """Kilometres per hour to metres per second."""
    return km_h * METRES_PER_KM / SECONDS_PER_HOUR
