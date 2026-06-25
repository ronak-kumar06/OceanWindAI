"""
Scientific quiver-plot visualization for OceanWind AI.

Generates a publication-quality wind field map exactly matching the style
of the reference image: light CartoDB basemap, jet colormap quiver arrows,
lat/lon grid, horizontal colorbar, and a data-source annotation.
"""
import os
import uuid
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)

try:
    import contextily as cx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    logger.warning("contextily not available — basemap will be plain white")


def _fmt_lat(val: float, _pos) -> str:
    return f"{abs(val):.1f}°{'N' if val >= 0 else 'S'}"

def _fmt_lon(val: float, _pos) -> str:
    return f"{abs(val):.1f}°{'E' if val >= 0 else 'W'}"


def generate_quiver_plot(
    lons: np.ndarray,
    lats: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    speed: np.ndarray,
    bbox: list,
    title: str = "Sentinel-1 SAR Wind Field",
    data_source: str = "Simulated Sentinel-1 GRD (Mock)",
    static_dir: str = "static",
) -> str:
    """
    Render a scientific quiver plot and save it as a PNG.

    Parameters
    ----------
    lons, lats : 1-D arrays of grid longitudes and latitudes
    u, v       : 2-D arrays (len(lats), len(lons)) of U/V wind components
    speed      : 2-D speed array — used for Jet colour mapping
    bbox       : [min_lon, min_lat, max_lon, max_lat]
    title      : plot title string
    data_source: annotation label (e.g. "Sentinel-1 IW GRD (Live)")
    static_dir : directory to save the PNG

    Returns
    -------
    str — filename of the saved PNG (not the full path)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    LON, LAT = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")

    # ── Basemap ───────────────────────────────────────────────────────────────
    if HAS_CTX:
        try:
            cx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=cx.providers.CartoDB.Positron,
                alpha=0.85,
            )
        except Exception as exc:
            logger.warning("Basemap fetch failed: %s", exc)

    # ── Background Color (Wind Speed) ─────────────────────────────────────────
    norm  = mcolors.Normalize(vmin=2.0, vmax=12.0)
    cmap  = plt.cm.jet

    # Create a colored background for the wind speed
    pm = ax.pcolormesh(
        LON, LAT, speed,
        cmap=cmap, norm=norm,
        shading='auto',
        alpha=0.85
    )

    # ── Quiver plot (Direction) ───────────────────────────────────────────────
    # Draw black arrows on top
    q = ax.quiver(
        LON, LAT, u, v,
        color='black',
        scale=120,
        width=0.003,
        headwidth=4,
        headlength=4,
        headaxislength=3.5,
        alpha=0.92,
    )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.tick_params(labelsize=9)
    ax.xaxis.tick_top()  # Move x-axis labels to the top to match Image 2
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.07,
                        aspect=40, fraction=0.03)
    cbar.set_label("Wind Speed (m/s)", fontsize=11, weight="bold")
    cbar.ax.tick_params(labelsize=9)

    # ── Title & annotation ────────────────────────────────────────────────────
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.annotate(
        f"Data source: {data_source}",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=7, color="gray",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
    )
    ax.annotate(
        "OceanWind AI — ResNet + CMOD5.N",
        xy=(0.99, 0.01), xycoords="axes fraction",
        fontsize=7, color="gray", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(static_dir, exist_ok=True)
    filename = f"windfield_{uuid.uuid4().hex[:10]}.png"
    filepath = os.path.join(static_dir, filename)
    plt.savefig(filepath, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)

    logger.info("Quiver plot saved: %s", filepath)
    return filename
