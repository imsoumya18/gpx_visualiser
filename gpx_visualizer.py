#!/usr/bin/env python3
"""Visualize a GPX track: elevation/speed profile (PNG) + interactive route map (HTML)."""

import argparse
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gpxpy
import matplotlib.pyplot as plt
import numpy as np
from folium import Map, Marker, PolyLine
from folium.features import Icon
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()


def load_points(gpx_path: Path):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            points.extend(segment.points)

    if not points:
        for route in gpx.routes:
            points.extend(route.points)

    if not points:
        raise ValueError("No track or route points found in GPX file")

    return gpx, points


def build_stats(points):
    lats = np.array([p.latitude for p in points])
    lons = np.array([p.longitude for p in points])
    elevations = np.array([p.elevation if p.elevation is not None else np.nan for p in points])

    distances = [0.0]
    for prev, curr in zip(points, points[1:]):
        distances.append(distances[-1] + prev.distance_3d(curr) or prev.distance_2d(curr) or 0.0)
    distances = np.array(distances) / 1000.0  # km

    times = [p.time for p in points]
    has_time = all(t is not None for t in times)
    speeds = np.full(len(points), np.nan)
    if has_time:
        for i in range(1, len(points)):
            dt = (times[i] - times[i - 1]).total_seconds()
            dd = distances[i] - distances[i - 1]
            speeds[i] = (dd / (dt / 3600.0)) if dt > 0 else 0.0

    return {
        "lats": lats,
        "lons": lons,
        "elevations": elevations,
        "distances": distances,
        "speeds": speeds,
        "has_time": has_time,
    }


def print_summary(stats):
    total_distance = stats["distances"][-1]
    elevations = stats["elevations"]
    valid_elev = elevations[~np.isnan(elevations)]
    gain = np.sum(np.diff(valid_elev).clip(min=0)) if len(valid_elev) > 1 else 0.0

    print(f"Points:         {len(stats['lats'])}")
    print(f"Total distance: {total_distance:.2f} km")
    if len(valid_elev):
        print(f"Elevation:      min {valid_elev.min():.0f} m, max {valid_elev.max():.0f} m, gain {gain:.0f} m")
    if stats["has_time"]:
        valid_speed = stats["speeds"][~np.isnan(stats["speeds"])]
        if len(valid_speed):
            print(f"Avg speed:      {valid_speed.mean():.1f} km/h")
            print(f"Max speed:      {valid_speed.max():.1f} km/h")


def plot_profile(stats, output_path: Path):
    has_elev = not np.all(np.isnan(stats["elevations"]))
    has_speed = stats["has_time"] and not np.all(np.isnan(stats["speeds"]))

    n_rows = sum([has_elev, has_speed]) or 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    idx = 0
    if has_elev:
        ax = axes[idx]
        ax.plot(stats["distances"], stats["elevations"], color="#2E7D32")
        ax.fill_between(stats["distances"], stats["elevations"], alpha=0.15, color="#2E7D32")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Elevation profile")
        ax.grid(True, alpha=0.3)
        idx += 1

    if has_speed:
        ax = axes[idx]
        ax.plot(stats["distances"], stats["speeds"], color="#1565C0")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed profile")
        ax.grid(True, alpha=0.3)
        idx += 1

    axes[-1].set_xlabel("Distance (km)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_3d(stats, output_path: Path):
    lats, lons, elevs = stats["lats"], stats["lons"], stats["elevations"]
    if np.all(np.isnan(elevs)):
        return False
    elevs = np.nan_to_num(elevs, nan=np.nanmean(elevs))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    points = np.array([lons, lats, elevs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap="terrain", linewidth=3)
    lc.set_array(elevs)
    ax.add_collection3d(lc)

    base = elevs.min() - (elevs.max() - elevs.min()) * 0.1 - 1
    ax.plot(lons, lats, base, color="gray", alpha=0.3, linewidth=1)
    ax.scatter(*next(zip(lons, lats, elevs)), color="green", s=60, label="Start")
    ax.scatter(lons[-1], lats[-1], elevs[-1], color="red", s=60, label="End")

    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.set_zlim(base, elevs.max() + (elevs.max() - elevs.min()) * 0.1 + 1)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Elevation (m)")
    ax.set_title("3D Elevation Track")
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(lc, ax=ax, shrink=0.6, label="Elevation (m)", pad=0.1)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_CACHE_DIR = Path.home() / ".cache" / "gpx_visualiser" / "terrarium_tiles"


def _lonlat_to_pixel(lon, lat, zoom):
    """Standard slippy-map global pixel coordinates at a given zoom (256px tiles)."""
    lat_rad = np.radians(lat)
    n = 2.0 ** zoom
    px = (lon + 180.0) / 360.0 * n * 256
    py = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n * 256
    return px, py


def choose_terrarium_zoom(lat_min, lat_max, lon_min, lon_max, max_tiles=100):
    """Pick the finest zoom level whose tile count for this bbox stays under max_tiles."""
    for zoom in range(14, 0, -1):
        x0p, y0p = _lonlat_to_pixel(lon_min, lat_max, zoom)
        x1p, y1p = _lonlat_to_pixel(lon_max, lat_min, zoom)
        n_tiles = (int(x1p // 256) - int(x0p // 256) + 1) * (int(y1p // 256) - int(y0p // 256) + 1)
        if n_tiles <= max_tiles:
            return zoom
    return 1


def _decode_terrarium_png(path):
    img = plt.imread(path)
    rgb = np.round(img[..., :3] * 255).astype(np.int64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768


def _fetch_one_tile(z, x, y):
    TERRARIUM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TERRARIUM_CACHE_DIR / f"{z}_{x}_{y}.png"
    if not cache_path.exists():
        url = TERRARIUM_URL.format(z=z, x=x, y=y)
        with urllib.request.urlopen(url, timeout=15, context=_SSL_CONTEXT) as resp:
            data = resp.read()
        cache_path.write_bytes(data)
    return (x, y), _decode_terrarium_png(cache_path)


def fetch_elevation_grid(lat_min, lat_max, lon_min, lon_max, grid_size=60, zoom=None, max_workers=16):
    """Fetch a regular DEM grid by mosaicking public AWS Terrarium elevation tiles locally.
    No API key, no rate limit (fair-use public dataset) -- tiles fetch in parallel and are
    cached on disk under ~/.cache/gpx_visualiser so repeat/overlapping runs skip re-downloading."""
    if zoom is None:
        zoom = choose_terrarium_zoom(lat_min, lat_max, lon_min, lon_max)

    x0p, y0p = _lonlat_to_pixel(lon_min, lat_max, zoom)
    x1p, y1p = _lonlat_to_pixel(lon_max, lat_min, zoom)
    tx0, ty0 = int(x0p // 256), int(y0p // 256)
    tx1, ty1 = int(x1p // 256), int(y1p // 256)
    n_tile_x, n_tile_y = tx1 - tx0 + 1, ty1 - ty0 + 1
    tiles = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    print(f"  zoom {zoom}: fetching {len(tiles)} tiles ({n_tile_x}x{n_tile_y}) in parallel ...")
    mosaic = np.full((n_tile_y * 256, n_tile_x * 256), np.nan)
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_tile, zoom, tx, ty): (tx, ty) for tx, ty in tiles}
        for future in as_completed(futures):
            tx, ty = futures[future]
            try:
                (tx, ty), tile_elev = future.result()
                row0, col0 = (ty - ty0) * 256, (tx - tx0) * 256
                mosaic[row0:row0 + 256, col0:col0 + 256] = tile_elev
            except Exception as e:
                failed += 1
                print(f"\n  warning: tile z{zoom}/{tx}/{ty} failed ({e})")
            done += 1
            sys.stdout.write(f"\r  tiles fetched: {done}/{len(tiles)}          ")
            sys.stdout.flush()
    print()
    if failed == len(tiles):
        raise RuntimeError("Could not fetch any elevation tiles")

    lat_1d = np.linspace(lat_min, lat_max, grid_size)
    lon_1d = np.linspace(lon_min, lon_max, grid_size)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    px, py = _lonlat_to_pixel(lon_grid.ravel(), lat_grid.ravel(), zoom)
    local_x = np.clip(px - tx0 * 256, 0, mosaic.shape[1] - 1.001)
    local_y = np.clip(py - ty0 * 256, 0, mosaic.shape[0] - 1.001)
    col0 = local_x.astype(int)
    row0 = local_y.astype(int)
    tcol = local_x - col0
    trow = local_y - row0
    z00 = mosaic[row0, col0]
    z01 = mosaic[row0, col0 + 1]
    z10 = mosaic[row0 + 1, col0]
    z11 = mosaic[row0 + 1, col0 + 1]
    elevations = (
        z00 * (1 - tcol) * (1 - trow) + z01 * tcol * (1 - trow)
        + z10 * (1 - tcol) * trow + z11 * tcol * trow
    )

    if np.any(np.isnan(elevations)):
        valid = ~np.isnan(elevations)
        flat_lats, flat_lons = lat_grid.ravel(), lon_grid.ravel()
        for i in np.where(~valid)[0]:
            dists = (flat_lats[valid] - flat_lats[i]) ** 2 + (flat_lons[valid] - flat_lons[i]) ** 2
            elevations[i] = elevations[valid][np.argmin(dists)]

    Z = elevations.reshape(grid_size, grid_size)
    return lat_1d, lon_1d, Z


def latlon_to_xy(lats, lons, lat0, lon0):
    """Rough equirectangular projection to local meters, adequate for small areas."""
    R = 6371000.0
    x = R * np.radians(lons - lon0) * np.cos(np.radians(lat0))
    y = R * np.radians(lats - lat0)
    return x, y


def bilinear_sample(lat_1d, lon_1d, Z, lats_q, lons_q):
    lat_idx = np.clip(np.searchsorted(lat_1d, lats_q) - 1, 0, len(lat_1d) - 2)
    lon_idx = np.clip(np.searchsorted(lon_1d, lons_q) - 1, 0, len(lon_1d) - 2)
    lat0v, lat1v = lat_1d[lat_idx], lat_1d[lat_idx + 1]
    lon0v, lon1v = lon_1d[lon_idx], lon_1d[lon_idx + 1]
    tlat = np.clip((lats_q - lat0v) / (lat1v - lat0v), 0, 1)
    tlon = np.clip((lons_q - lon0v) / (lon1v - lon0v), 0, 1)

    z00 = Z[lat_idx, lon_idx]
    z01 = Z[lat_idx, lon_idx + 1]
    z10 = Z[lat_idx + 1, lon_idx]
    z11 = Z[lat_idx + 1, lon_idx + 1]

    z0 = z00 * (1 - tlon) + z01 * tlon
    z1 = z10 * (1 - tlon) + z11 * tlon
    return z0 * (1 - tlat) + z1 * tlat


def densify_path(x, y, spacing):
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    total = d[-1]
    if total <= 0:
        return x, y
    n = max(int(total / spacing), len(x) * 2)
    d_new = np.linspace(0, total, n)
    return np.interp(d_new, d, x), np.interp(d_new, d, y)


def upsample_grid(lat_1d, lon_1d, Z, factor):
    """Bilinearly densify a regular lat/lon elevation grid so the mesh renders smoothly
    instead of showing large flat facets, without costing extra API requests."""
    if factor <= 1:
        return lat_1d, lon_1d, Z
    fine_lat = np.linspace(lat_1d[0], lat_1d[-1], (len(lat_1d) - 1) * factor + 1)
    fine_lon = np.linspace(lon_1d[0], lon_1d[-1], (len(lon_1d) - 1) * factor + 1)
    lon_grid, lat_grid = np.meshgrid(fine_lon, fine_lat)
    Z_fine = bilinear_sample(lat_1d, lon_1d, Z, lat_grid.ravel(), lon_grid.ravel()).reshape(lat_grid.shape)
    return fine_lat, fine_lon, Z_fine


def _clamp_terrain_resolution(grid_size, upsample):
    # Each mesh face is built as a plain Python list/tuple (not a compact numpy array), so face
    # count directly drives RAM: ~1KB/face once the matching color and matplotlib's internal
    # copies are counted. grid_size and upsample multiply together into the final mesh size, so
    # e.g. grid=1000 with the default upsample=3 silently builds an ~18M-face mesh (tens of GB,
    # and matplotlib's Poly3DCollection isn't built to draw that many polygons anyway). Clamp
    # both to keep this from ever OOM-ing a machine.
    MAX_GRID_SIZE = 300
    if grid_size > MAX_GRID_SIZE:
        print(
            f"Note: --terrain-grid {grid_size} only resamples the already-fetched tiles more "
            f"densely, it doesn't add real detail beyond the source data -- capping to "
            f"{MAX_GRID_SIZE}. Use --terrain-zoom for genuinely finer source detail instead."
        )
        grid_size = MAX_GRID_SIZE

    MAX_EFFECTIVE_SIZE = 450
    effective_size = (grid_size - 1) * upsample + 1
    if effective_size > MAX_EFFECTIVE_SIZE:
        safe_upsample = max(1, (MAX_EFFECTIVE_SIZE - 1) // max(grid_size - 1, 1))
        print(
            f"Warning: grid_size={grid_size} x upsample={upsample} would build a "
            f"{effective_size}x{effective_size} mesh (~{2 * (effective_size - 1) ** 2:,} polygons) "
            f"-- that can use tens of GB of RAM and take minutes to render. Capping upsample to "
            f"{safe_upsample} instead."
        )
        upsample = safe_upsample
    return grid_size, upsample


def _grid_quads(xg, yg, zg):
    return [
        [(xg[i, j], yg[i, j], zg[i, j]), (xg[i, j + 1], yg[i, j + 1], zg[i, j + 1]),
         (xg[i + 1, j + 1], yg[i + 1, j + 1], zg[i + 1, j + 1]), (xg[i + 1, j], yg[i + 1, j], zg[i + 1, j])]
        for i in range(xg.shape[0] - 1) for j in range(xg.shape[1] - 1)
    ]


def _prepare_terrain_mesh(stats, grid_size, padding, exaggeration, upsample, zoom):
    """Fetch elevation, shade it, and build the static mesh geometry (terrain top + side walls
    + base) shared by both the static PNG render and the walk animation."""
    lats, lons, elevs = stats["lats"], stats["lons"], stats["elevations"]

    lat_span = max(lats.max() - lats.min(), 0.001)
    lon_span = max(lons.max() - lons.min(), 0.001)
    lat_min = lats.min() - lat_span * padding
    lat_max = lats.max() + lat_span * padding
    lon_min = lons.min() - lon_span * padding
    lon_max = lons.max() + lon_span * padding

    grid_size, upsample = _clamp_terrain_resolution(grid_size, upsample)

    print(f"Fetching terrain grid ({grid_size}x{grid_size}) from local elevation tiles ...")
    lat_1d, lon_1d, Z_raw = fetch_elevation_grid(lat_min, lat_max, lon_min, lon_max, grid_size=grid_size, zoom=zoom)
    print("Smoothing mesh ...")
    lat_1d, lon_1d, Z_raw = upsample_grid(lat_1d, lon_1d, Z_raw, upsample)
    grid_size = len(lat_1d)

    lat0, lon0 = lats.mean(), lons.mean()
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    X, Y = latlon_to_xy(lat_grid, lon_grid, lat0, lon0)

    relief = max(Z_raw.max() - Z_raw.min(), 1.0)
    # Exaggerate only the relief above the base (pivoting at the minimum), so the block's
    # side-wall thickness added below can stay a fixed, un-exaggerated depth.
    Z = Z_raw.min() + (Z_raw - Z_raw.min()) * exaggeration

    track_x, track_y = latlon_to_xy(lats, lons, lat0, lon0)
    terrain_under_track = bilinear_sample(lat_1d, lon_1d, Z_raw, lats, lons)
    if np.all(np.isnan(elevs)):
        track_z_raw = terrain_under_track
    else:
        track_z_raw = np.where(np.isnan(elevs), terrain_under_track, elevs)
    track_z = Z_raw.min() + (track_z_raw - Z_raw.min()) * exaggeration
    track_z = track_z + relief * exaggeration * 0.03 + 1  # lift path above the surface so it doesn't z-fight

    # Matte charcoal "3D-print" material: directional hillshade for relief + horizontal
    # banding (constant-elevation stripes) to mimic visible print/contour layer lines.
    dx = (X[0, -1] - X[0, 0]) / (grid_size - 1)
    dy = (Y[-1, 0] - Y[0, 0]) / (grid_size - 1)
    ls = LightSource(azdeg=315, altdeg=45)
    intensity = ls.hillshade(Z_raw, vert_exag=exaggeration, dx=dx, dy=dy)
    intensity = 0.3 + 0.7 * intensity  # ambient fill so away-facing slopes don't go pure black
    dark, light = np.array([0.09, 0.09, 0.10]), np.array([0.62, 0.62, 0.65])
    base_rgb = dark + intensity[..., None] * (light - dark)
    band_height = relief / 45.0
    banding = 0.88 + 0.12 * np.cos(2 * np.pi * Z_raw / band_height)
    base_rgb = np.clip(base_rgb * banding[..., None], 0, 1)

    cellsize = max(min((X.max() - X.min()) / grid_size, (Y.max() - Y.min()) / grid_size), 1.0)

    print("Building 3D mesh ...")
    surface_faces = _grid_quads(X, Y, Z)

    base_z = Z.min() - relief * 0.12
    def wall_quads(xs, ys, zs):
        return [
            [(xs[i], ys[i], zs[i]), (xs[i + 1], ys[i + 1], zs[i + 1]),
             (xs[i + 1], ys[i + 1], base_z), (xs[i], ys[i], base_z)]
            for i in range(len(xs) - 1)
        ]

    wall_faces = (
        wall_quads(X[0, :], Y[0, :], Z[0, :])
        + wall_quads(X[-1, :], Y[-1, :], Z[-1, :])
        + wall_quads(X[:, 0], Y[:, 0], Z[:, 0])
        + wall_quads(X[:, -1], Y[:, -1], Z[:, -1])
    )
    # The bottom cap must be subdivided to match the surface grid, not one giant quad: Poly3DCollection
    # depth-sorts faces by each face's own centroid (painter's algorithm), and one huge quad gets a
    # single centroid that can wrongly place the whole thing in front of nearer, smaller surface faces.
    bottom_faces = _grid_quads(X, Y, np.full_like(Z, base_z))
    wall_color = (0.08, 0.08, 0.09, 1.0)
    wall_bottom_colors = [wall_color] * (len(wall_faces) + len(bottom_faces))

    return {
        "X": X, "Y": Y, "Z": Z, "Z_raw": Z_raw, "base_rgb": base_rgb,
        "track_x": track_x, "track_y": track_y, "track_z": track_z, "track_z_raw": track_z_raw,
        "cellsize": cellsize, "relief": relief, "base_z": base_z,
        "surface_faces": surface_faces, "wall_faces": wall_faces, "bottom_faces": bottom_faces,
        "wall_bottom_colors": wall_bottom_colors,
    }


def _face_colors_from_vertex_rgb(rgb):
    rgba = np.dstack([rgb, np.ones(rgb.shape[:2])])
    face_rgba = (rgba[:-1, :-1] + rgba[:-1, 1:] + rgba[1:, :-1] + rgba[1:, 1:]) / 4.0
    return face_rgba.reshape(-1, 4)


def _setup_terrain_axes(ax, fig, m):
    X, Y, Z, base_z = m["X"], m["Y"], m["Z"], m["base_z"]
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = max(Z.max() - base_z, 1)
    ax.set_box_aspect((x_range, y_range, z_range))
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(base_z, Z.max())
    ax.margins(0)
    ax.set_axis_off()
    ax.view_init(elev=48, azim=-55)
    fig.patch.set_facecolor("#dcdcdc")
    fig.subplots_adjust(left=-0.05, right=1.05, top=1.08, bottom=-0.08)


def plot_terrain_relief(
    stats, output_path: Path, grid_size=60, padding=0.3, exaggeration=1.3, upsample=3, zoom=None,
):
    """Render a shaded 3D terrain block (like a topographic model) with the track glowing on top."""
    m = _prepare_terrain_mesh(stats, grid_size, padding, exaggeration, upsample, zoom)
    X, Y, Z = m["X"], m["Y"], m["Z"]
    track_x, track_y = m["track_x"], m["track_y"]

    # Bake the path directly into the surface's own face colors instead of drawing it as a
    # separate 3D line. matplotlib's 3D renderer has no true z-buffer between collections, so
    # a floating line over a large surface mesh gets sorted incorrectly and disappears behind
    # it. Painting the path color onto the mesh cells nearest the track guarantees it is
    # depth-sorted as part of the same surface and always shows through correctly.
    dense_x, dense_y = densify_path(track_x, track_y, m["cellsize"] / 2)
    dist = np.full(X.shape, np.inf)
    chunk = 150
    for start in range(0, len(dense_x), chunk):
        end = start + chunk
        ddx = X[..., None] - dense_x[start:end]
        ddy = Y[..., None] - dense_y[start:end]
        dist = np.minimum(dist, np.sqrt(ddx**2 + ddy**2).min(axis=-1))

    # Hard-edged path: a crisp cutoff at path_width, no gradient/opacity falloff.
    path_width = m["cellsize"] * 1.1
    orange = np.array([1.0, 0.55, 0.10])
    rgb = m["base_rgb"].copy()
    rgb[dist <= path_width] = orange
    surface_colors = _face_colors_from_vertex_rgb(np.clip(rgb, 0, 1))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_faces = m["surface_faces"] + m["wall_faces"] + m["bottom_faces"]
    all_colors = list(map(tuple, surface_colors)) + m["wall_bottom_colors"]
    ax.add_collection3d(Poly3DCollection(
        all_faces, facecolors=all_colors, edgecolor=(0, 0, 0, 0.15), linewidth=0.1,
    ))

    start_dist = np.hypot(X - track_x[0], Y - track_y[0])
    end_dist = np.hypot(X - track_x[-1], Y - track_y[-1])
    start_i, start_j = np.unravel_index(np.argmin(start_dist), start_dist.shape)
    end_i, end_j = np.unravel_index(np.argmin(end_dist), end_dist.shape)
    ax.scatter(
        [X[start_i, start_j]], [Y[start_i, start_j]], [Z[start_i, start_j] + m["relief"] * 0.01],
        color="#2ECC71", s=70, depthshade=False,
    )
    ax.scatter(
        [X[end_i, end_j]], [Y[end_i, end_j]], [Z[end_i, end_j] + m["relief"] * 0.01],
        color="#E74C3C", s=70, depthshade=False,
    )

    _setup_terrain_axes(ax, fig, m)
    print("Rendering (this can take a while for large/high-upsample grids) ...")
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def _shortest_angle_step(current, target, max_step):
    """Ease `current` toward `target` (degrees), taking the short way around the 360 wrap."""
    delta = (target - current + 180) % 360 - 180
    delta = np.clip(delta, -max_step, max_step)
    return current + delta


def _lerp_angle(a, b, t):
    """Interpolate directly from angle a to b (degrees) by fraction t, shortest way around."""
    delta = (b - a + 180) % 360 - 180
    return a + delta * t


def _smoothstep(t):
    return t * t * (3 - 2 * t)


def render_walk_video(
    stats, output_path: Path, grid_size=60, padding=0.3, exaggeration=1.3, upsample=3, zoom=None,
    n_frames=90, fps=15,
):
    """Animate the track being progressively walked across the 3D terrain relief: a tightly
    zoomed chase camera pans/rotates to follow the current position, then pulls back to reveal
    the whole terrain block at the end. A marker and HUD (distance/elevation + elevation-profile
    strip) track progress. Saved as an mp4."""
    import imageio.v2 as imageio

    m = _prepare_terrain_mesh(stats, grid_size, padding, exaggeration, upsample, zoom)
    X, Y = m["X"], m["Y"]
    track_x, track_y, track_z = m["track_x"], m["track_y"], m["track_z"]
    track_z_raw = m["track_z_raw"]
    cellsize = m["cellsize"]

    # Densify the whole path ONCE with a shared arc-length parameterization (including z), so
    # each frame's "revealed so far" is always a strict, stable prefix of this same array.
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(track_x), np.diff(track_y)))])
    total = max(d[-1], 1e-6)
    n_dense = max(int(total / (cellsize / 2)), len(track_x) * 2, n_frames)
    d_new = np.linspace(0, total, n_dense)
    dense_x = np.interp(d_new, d, track_x)
    dense_y = np.interp(d_new, d, track_y)
    dense_z = np.interp(d_new, d, track_z)
    dense_z_raw = np.interp(d_new, d, track_z_raw)
    total_distance_km = stats["distances"][-1]

    path_width = cellsize * 1.1
    orange = np.array([1.0, 0.55, 0.10])

    fig = plt.figure(figsize=(11, 9), dpi=100)  # lower dpi than the static PNG -- animated GIFs
    # get large fast, and full print resolution isn't needed for something meant to be played back
    ax = fig.add_subplot(111, projection="3d")

    all_faces = m["surface_faces"] + m["wall_faces"] + m["bottom_faces"]
    base_surface_colors = _face_colors_from_vertex_rgb(m["base_rgb"])
    initial_colors = list(map(tuple, base_surface_colors)) + m["wall_bottom_colors"]
    poly = Poly3DCollection(all_faces, facecolors=initial_colors, edgecolor=(0, 0, 0, 0.15), linewidth=0.1)
    ax.add_collection3d(poly)

    marker, = ax.plot(
        [dense_x[0]], [dense_y[0]], [dense_z[0]],
        marker="o", markersize=9, color="white", markeredgecolor="#FF8C1A", markeredgewidth=2,
    )

    # Chase camera: pan a small square window centered on the current position, and rotate azim
    # to face the direction of travel. matplotlib's 3D axes has no free first-person camera (it
    # always orbits the box set by xlim/ylim/zlim at a fixed elevation/azimuth), so this pan +
    # rotate is the closest practical approximation of "the camera follows the hiker." In the
    # final portion of the video the window widens and re-centers back out to the full terrain
    # extent, pulling back to an overview -- box_aspect is recomputed every frame to match,
    # since it's what actually controls the drawn box's proportions, not just xlim/ylim.
    follow_radius = max(total * 0.05, cellsize * 15)
    x_mid, y_mid = (X.max() + X.min()) / 2, (Y.max() + Y.min()) / 2
    overview_half_w = max(X.max() - X.min(), Y.max() - Y.min()) / 2 * 1.02
    overview_elev, overview_azim = 48, -55
    z_range = max(m["Z"].max() - m["base_z"], 1)
    ax.set_zlim(m["base_z"], m["Z"].max())
    ax.set_axis_off()
    fig.patch.set_facecolor("#dcdcdc")
    fig.subplots_adjust(left=-0.05, right=1.05, top=1.08, bottom=-0.08)

    # HUD: distance/elevation readout + a small elevation-profile strip with a moving marker.
    hud_text = fig.text(
        0.03, 0.95, "", fontsize=13, color="#222", family="monospace", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    profile_ax = fig.add_axes((0.55, 0.06, 0.42, 0.16))
    profile_distances = stats["distances"]
    profile_elevs = stats["elevations"]
    if np.all(np.isnan(profile_elevs)):
        profile_elevs = np.nan_to_num(profile_elevs, nan=0.0)
    profile_ax.plot(profile_distances, profile_elevs, color="#555", linewidth=1)
    profile_ax.fill_between(profile_distances, profile_elevs, alpha=0.15, color="#555")
    profile_marker = profile_ax.axvline(0, color="#FF8C1A", linewidth=2)
    profile_ax.set_xlim(profile_distances[0], profile_distances[-1])
    profile_ax.set_ylim(np.nanmin(profile_elevs), np.nanmax(profile_elevs) + 1)
    profile_ax.set_facecolor("none")
    profile_ax.patch.set_alpha(0.6)
    for spine in profile_ax.spines.values():
        spine.set_visible(False)
    profile_ax.tick_params(labelsize=7, length=2)

    # Driving the frame loop manually (rather than matplotlib.animation.FuncAnimation) because
    # FuncAnimation does not reliably re-capture Poly3DCollection facecolor updates per frame
    # with a static 3D camera -- it was verified to just repeat one frame's state across the
    # whole output. Direct set_facecolors() + fig.canvas.draw() + grabbing the rendered buffer,
    # called sequentially ourselves, is what's confirmed to work correctly.
    running_dist = np.full(X.shape, np.inf)
    prev_revealed = 0
    wall_bottom_colors_arr = np.array(m["wall_bottom_colors"])
    current_azim = None
    chase_end_azim = None
    lookahead = max(n_dense // 40, 3)

    # Reserve the tail of the video for pulling the camera back out to an overview; the path
    # finishes revealing itself by the start of that phase and then just holds at 100%.
    zoom_out_frames = max(int(n_frames * 0.22), 10)
    reveal_frames = max(n_frames - zoom_out_frames, 1)

    print(f"Rendering animation frames -> {output_path} (this can take a while) ...")
    writer = imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=5,
        output_params=["-pix_fmt", "yuv420p"],
    )
    for frame_i in range(n_frames):
        reveal_frame_i = min(frame_i, reveal_frames - 1)
        revealed = max(int((reveal_frame_i + 1) / reveal_frames * n_dense), 1)
        if revealed > prev_revealed:
            new_x, new_y = dense_x[prev_revealed:revealed], dense_y[prev_revealed:revealed]
            ddx = X[..., None] - new_x
            ddy = Y[..., None] - new_y
            new_dist = np.sqrt(ddx**2 + ddy**2).min(axis=-1)
            running_dist = np.minimum(running_dist, new_dist)
            prev_revealed = revealed

        rgb_frame = m["base_rgb"].copy()
        rgb_frame[running_dist <= path_width] = orange
        surface_colors = _face_colors_from_vertex_rgb(rgb_frame)
        poly.set_facecolors(np.vstack([surface_colors, wall_bottom_colors_arr]))

        idx = revealed - 1
        cx, cy, cz = dense_x[idx], dense_y[idx], dense_z[idx]
        marker.set_data_3d([cx], [cy], [cz])

        ahead = min(idx + lookahead, n_dense - 1)
        heading = np.degrees(np.arctan2(dense_y[ahead] - cy, dense_x[ahead] - cx))
        target_azim = heading - 90
        current_azim = target_azim if current_azim is None else _shortest_angle_step(current_azim, target_azim, 8)

        if frame_i < reveal_frames:
            half_w, view_cx, view_cy, view_elev, view_azim = follow_radius, cx, cy, 42, current_azim
            chase_end_azim = current_azim
        else:
            t = _smoothstep(min((frame_i - reveal_frames + 1) / zoom_out_frames, 1.0))
            half_w = follow_radius + t * (overview_half_w - follow_radius)
            view_cx = cx + t * (x_mid - cx)
            view_cy = cy + t * (y_mid - cy)
            view_elev = 42 + t * (overview_elev - 42)
            view_azim = _lerp_angle(chase_end_azim, overview_azim, t)

        ax.set_box_aspect((half_w * 2, half_w * 2, z_range))
        ax.set_xlim(view_cx - half_w, view_cx + half_w)
        ax.set_ylim(view_cy - half_w, view_cy + half_w)
        ax.view_init(elev=view_elev, azim=view_azim)

        progress = revealed / n_dense
        hud_text.set_text(
            f"Distance {progress * total_distance_km:6.2f} / {total_distance_km:.2f} km\n"
            f"Elevation {dense_z_raw[idx]:5.0f} m"
        )
        profile_marker.set_xdata([progress * total_distance_km, progress * total_distance_km])

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(buf)

        sys.stdout.write(f"\r  rendering frame {frame_i + 1}/{n_frames}          ")
        sys.stdout.flush()
    print()

    plt.close(fig)
    writer.close()


def build_map(stats, output_path: Path):
    lats, lons = stats["lats"], stats["lons"]
    coords = list(zip(lats, lons))
    center = [lats.mean(), lons.mean()]

    fmap = Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    PolyLine(coords, color="#1565C0", weight=4, opacity=0.8).add_to(fmap)

    Marker(coords[0], tooltip="Start", icon=Icon(color="green", icon="play")).add_to(fmap)
    Marker(coords[-1], tooltip="End", icon=Icon(color="red", icon="stop")).add_to(fmap)

    sw = [lats.min(), lons.min()]
    ne = [lats.max(), lons.max()]
    fmap.fit_bounds([sw, ne])

    fmap.save(str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Visualize a GPX track")
    parser.add_argument("gpx_file", type=Path, help="Path to the .gpx file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory for output files")
    parser.add_argument(
        "--terrain", action="store_true",
        help="Also render a shaded 3D terrain relief with the track glowing on top "
             "(fetches real elevation from public AWS Terrarium tiles, cached locally under "
             "~/.cache/gpx_visualiser; needs internet)",
    )
    parser.add_argument(
        "--terrain-grid", type=int, default=60,
        help="Terrain grid resolution per side sampled from the fetched tiles (default 60; cheap "
             "to raise since it's just local interpolation, not extra downloads)",
    )
    parser.add_argument(
        "--terrain-upsample", type=int, default=3,
        help="Further smooth the grid by this factor before rendering, so the mesh doesn't look "
             "blocky/pixelated (default 3; free, just a denser interpolated mesh)",
    )
    parser.add_argument(
        "--terrain-zoom", type=int, default=None,
        help="Tile zoom level to fetch (higher = more real detail per tile, more tiles needed). "
             "Default: auto-picked to keep tile count reasonable for the track's bounding box.",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Render an mp4 of the walk progressing along the 3D terrain: a chase camera "
             "follows the moving path, then pulls back to a full overview at the end. "
             "Instead of a static PNG.",
    )
    parser.add_argument(
        "--video-frames", type=int, default=90,
        help="Number of animation frames (default 90). More frames = smoother but slower to render.",
    )
    parser.add_argument(
        "--video-fps", type=int, default=15,
        help="Playback frames per second (default 15).",
    )
    args = parser.parse_args()

    if not args.gpx_file.exists():
        sys.exit(f"Error: file not found: {args.gpx_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.gpx_file.stem

    gpx, points = load_points(args.gpx_file)
    stats = build_stats(points)

    print_summary(stats)

    if args.video:
        video_path = args.output_dir / f"{stem}_walk.mp4"
        render_walk_video(
            stats, video_path, grid_size=args.terrain_grid, upsample=args.terrain_upsample,
            zoom=args.terrain_zoom, n_frames=args.video_frames, fps=args.video_fps,
        )
        print(f"\nSaved walk animation -> {video_path}")
        return

    if args.terrain:
        terrain_path = args.output_dir / f"{stem}_terrain.png"
        plot_terrain_relief(
            stats, terrain_path, grid_size=args.terrain_grid, upsample=args.terrain_upsample,
            zoom=args.terrain_zoom,
        )
        print(f"\nSaved 3D terrain relief -> {terrain_path}")
        return

    profile_path = args.output_dir / f"{stem}_profile.png"
    map_path = args.output_dir / f"{stem}_map.html"
    plot3d_path = args.output_dir / f"{stem}_3d.png"

    plot_profile(stats, profile_path)
    build_map(stats, map_path)
    has_3d = plot_3d(stats, plot3d_path)

    print(f"\nSaved profile chart -> {profile_path}")
    print(f"Saved interactive map -> {map_path}")
    if has_3d:
        print(f"Saved 3D elevation plot -> {plot3d_path}")


if __name__ == "__main__":
    main()
