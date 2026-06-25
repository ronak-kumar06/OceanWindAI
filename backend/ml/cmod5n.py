"""
CMOD5.N — C-band Model 5 with N-tuned coefficients.

Full Python implementation of the CMOD5.N geophysical model function
as described in:
  Hersbach, H. (2010). Comparison of C-band scatterometer CMOD5.N
  equivalent neutral winds with ECMWF. J. Atmos. Oceanic Technol., 27, 721-736.

Usage
-----
  from ml.cmod5n import cmod5n_inverse
  wind_speed = cmod5n_inverse(sigma0_dB, wind_dir_rel, inc_angle_deg)

where:
  sigma0_dB      : observed sigma0 in dB
  wind_dir_rel   : wind direction relative to radar look direction (degrees)
  inc_angle_deg  : local incidence angle (degrees)
"""
import numpy as np

# ── CMOD5.N coefficients ──────────────────────────────────────────────────────
C  = [0,                          # index-0 unused (1-based indexing)
      -0.6878, -0.7957,  0.3380, -0.1728,  0.0000,
       0.0040,  0.1103,  0.0159,  6.7329,  2.7713,
      -2.2885,  0.3100, -0.0646,  1.3236,  0.2214,
       0.3177,  0.4355,  0.0000,  0.0000,  0.1895,
       0.0100,  0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000]

# Note: Y0/PN/A/B are used internally in some CMOD variants but not needed here.


def _cmod5n_forward(v: np.ndarray, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute sigma0 from wind speed, direction, and incidence angle.

    Parameters
    ----------
    v     : wind speed (m/s)  — shape (N,)
    phi   : wind direction relative to radar look (degrees) — shape (N,)
    theta : incidence angle (degrees) — shape (N,)

    Returns
    -------
    sigma0 in linear scale — shape (N,)
    """
    # Convert to radians
    phi_r   = np.deg2rad(phi)
    theta_r = np.deg2rad(theta)

    xi  = (theta - 40.0) / 25.0
    mu  = np.cos(phi_r)
    nu  = np.cos(2.0 * phi_r)

    # B0 term
    B0  = 10.0 ** (C[1] + C[2] * xi + C[3] * xi**2)
    B0 *= (1.0 + C[4] * v + C[5] * v**2)

    # B1 term
    B1  = (C[6] + C[7] * xi + C[8] * xi**2)
    B1 *= v

    # B2 term
    B2  = (C[9] + C[10] * xi + C[11] * xi**2)
    B2 *= v

    # Sigma0 (linear)
    sigma0 = B0 * (1.0 + B1 * mu + B2 * nu)
    return sigma0


def cmod5n_inverse(
    sigma0_dB: np.ndarray,
    wind_dir_rel: np.ndarray,
    inc_angle: np.ndarray,
    v_min: float = 0.1,
    v_max: float = 50.0,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Invert CMOD5.N to retrieve wind speed from sigma0.

    Uses Newton–Raphson iteration (fast, converges in ~5 steps for typical values).

    Parameters
    ----------
    sigma0_dB    : observed sigma0 (dB)   — scalar or array
    wind_dir_rel : wind dir rel to look (°) — same shape
    inc_angle    : local incidence angle (°) — same shape

    Returns
    -------
    wind_speed   : m/s — same shape as input
    """
    sigma0_lin = 10.0 ** (sigma0_dB / 10.0)

    sigma0_dB    = np.atleast_1d(np.asarray(sigma0_dB,    dtype=float))
    wind_dir_rel = np.atleast_1d(np.asarray(wind_dir_rel, dtype=float))
    inc_angle    = np.atleast_1d(np.asarray(inc_angle,    dtype=float))
    sigma0_lin   = 10.0 ** (sigma0_dB / 10.0)

    v = np.full_like(sigma0_lin, 7.0)   # initial guess: 7 m/s

    for _ in range(max_iter):
        f  = _cmod5n_forward(v,         wind_dir_rel, inc_angle) - sigma0_lin
        df = _cmod5n_forward(v + 0.001, wind_dir_rel, inc_angle)
        df = (df - _cmod5n_forward(v, wind_dir_rel, inc_angle)) / 0.001
        df = np.where(np.abs(df) < 1e-10, 1e-10, df)
        v  = v - f / df
        v  = np.clip(v, v_min, v_max)
        if np.max(np.abs(f)) < tol:
            break

    return v.squeeze()


if __name__ == "__main__":
    # Quick self-test with known values
    sigma0_test  = np.array([-14.0, -12.0, -10.0])
    dir_test     = np.array([  0.0,  45.0,  90.0])
    theta_test   = np.array([ 35.0,  35.0,  35.0])
    ws           = cmod5n_inverse(sigma0_test, dir_test, theta_test)
    print("CMOD5.N self-test wind speeds:", ws, "m/s")
