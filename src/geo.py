# src/geo.py — Vectorized geodetic <-> ENU utilities (pure NumPy)

from __future__ import annotations
import numpy as np

# ----------------------------
# WGS84 ellipsoid parameters
# ----------------------------
_A = 6378137.0                     # semi-major axis (meters)
_F = 1.0 / 298.257223563           # flattening
_E2 = _F * (2.0 - _F)              # eccentricity squared

# ----------------------------
# Internal helpers
# ----------------------------
def _deg2rad(d) -> np.ndarray:
    """Degrees → radians (vectorized)."""
    return np.asarray(d, dtype=float) * (np.pi / 180.0)

def _geodetic_to_ecef(lat, lon, h):
    """
    Geodetic (lat, lon in degrees; h in meters) → ECEF (x, y, z in meters).
    lat, lon, h can be arrays or scalars; returns arrays with matching shape.
    """
    lat = _deg2rad(lat)
    lon = _deg2rad(lon)

    sl, cl = np.sin(lat), np.cos(lat)
    sb, cb = np.sin(lon), np.cos(lon)

    N = _A / np.sqrt(1.0 - _E2 * sl * sl)    # prime vertical radius of curvature

    x = (N + h) * cl * cb
    y = (N + h) * cl * sb
    z = (N * (1.0 - _E2) + h) * sl
    return x, y, z

def _enu_rotation(lat0, lon0) -> np.ndarray:
    """
    Rotation matrix (3x3) mapping ECEF deltas to local ENU at anchor (lat0, lon0).
    lat0, lon0 in degrees (scalars).
    """
    lat0 = float(_deg2rad(lat0))
    lon0 = float(_deg2rad(lon0))

    sl, cl = np.sin(lat0), np.cos(lat0)
    sb, cb = np.sin(lon0), np.cos(lon0)

    # Rows are unit vectors of E, N, U expressed in ECEF
    return np.array([
        [-sb,        cb,       0.0],
        [-sl * cb,  -sl * sb,  cl ],
        [ cl * cb,   cl * sb,  sl ],
    ], dtype=float)

# ----------------------------
# Public API
# ----------------------------
def latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert arrays of geodetic coordinates to local ENU, anchored at (lat0, lon0, alt0).

    Inputs
    ------
    lat, lon, alt : array-like or scalars
        Latitude [deg], Longitude [deg], Altitude above ellipsoid [m].
        Arrays must be 1D and of the same length.
        NaNs in `alt` are treated as 0.0 (you can pre-fill with baro altitude if desired).
    lat0, lon0, alt0 : scalars
        Anchor point (degrees, meters).

    Returns
    -------
    E, N, U : np.ndarray
        East, North, Up coordinates [m], same shape as `lat`.
    """
    # Ensure arrays
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)

    # Handle NaNs in altitude robustly
    alt = np.nan_to_num(alt, nan=0.0)
    alt0 = 0.0 if (alt0 is None or (isinstance(alt0, float) and np.isnan(alt0))) else float(alt0)

    # Vectorized geodetic -> ECEF
    x, y, z = _geodetic_to_ecef(lat, lon, alt)

    # Anchor ECEF (scalars)
    x0, y0, z0 = _geodetic_to_ecef(float(lat0), float(lon0), alt0)

    # ECEF deltas
    dx, dy, dz = x - x0, y - y0, z - z0

    # Rotate to ENU (matrix multiply, all points at once)
    R = _enu_rotation(lat0, lon0)          # (3, 3)
    enu = R @ np.vstack((dx, dy, dz))      # (3, N)

    E, N, U = enu[0], enu[1], enu[2]
    return E, N, U

def first_point_is_origin(E, N, U, atol: float = 1e-6) -> bool:
    """
    Convenience check: for a per-flight ENU computed with an anchor at the first
    valid point, the first ENU should be ~ (0,0,0).
    """
    if len(E) == 0:
        return True
    return (abs(E[0]) <= atol) and (abs(N[0]) <= atol) and (abs(U[0]) <= atol)
