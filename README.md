# gpx_visualiser

Turn a GPX track into a set of visualizations: elevation/speed charts, an interactive route
map, a 3D elevation plot, a shaded 3D terrain relief block with the track baked in, and an mp4
flythrough animation of the walk.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No API keys needed. Terrain elevation comes from the public AWS-hosted Terrarium tile dataset
(no auth, no rate limit) and is cached locally under `~/.cache/gpx_visualiser/` so repeat runs
over the same area don't re-download anything. mp4 encoding uses `imageio-ffmpeg`, which bundles
its own ffmpeg binary inside the venv — no system/Homebrew install required.

## Usage

```bash
# Default: profile chart + interactive map + 3D elevation line plot
python gpx_visualizer.py track.gpx

# Shaded 3D terrain relief (real elevation, track glowing on top)
python gpx_visualizer.py track.gpx --terrain

# mp4 flythrough: chase camera follows the walk, then pulls back to a full overview
python gpx_visualizer.py track.gpx --video
```

Outputs are written next to the input file by default (`-o` / `--output-dir` to change that),
named after the GPX file's stem:

| Mode | Output |
|---|---|
| default | `<name>_profile.png`, `<name>_map.html`, `<name>_3d.png` |
| `--terrain` | `<name>_terrain.png` |
| `--video` | `<name>_walk.mp4` |

`--video` and `--terrain` are mutually exclusive per run (whichever is passed wins, `--video`
takes priority if both are given) — each produces just its one output file, not the full default
set.

## Options

| Flag | Default | Description |
|---|---|---|
| `-o`, `--output-dir` | `.` | Directory for output files |
| `--terrain` | off | Render the 3D terrain relief PNG instead of the default charts |
| `--terrain-grid` | `60` | Grid resolution sampled from the fetched tiles. Cheap to raise — it's local interpolation, not extra downloads |
| `--terrain-upsample` | `3` | Extra smoothing factor applied before rendering, so the mesh doesn't look blocky |
| `--terrain-zoom` | auto | Tile zoom level to fetch (higher = more real detail, more tiles). Auto-picked to keep tile count reasonable for the track's bounding box |
| `--video` | off | Render the mp4 walk animation instead of the default charts |
| `--video-frames` | `90` | Number of animation frames. More = smoother but slower to render |
| `--video-fps` | `15` | Playback frame rate |

Both `--terrain` and `--video` share the `--terrain-grid` / `--terrain-upsample` / `--terrain-zoom`
flags, since they use the same underlying terrain mesh.

### A note on scale

`--terrain-grid` and `--terrain-upsample` multiply together into the final mesh size, and mesh
size drives both render time and memory. The script clamps both automatically to safe limits, so
an oversized value (e.g. `--terrain-grid 1000`) gets capped rather than exhausting memory — it'll
print a note explaining what got reduced and why. If you want more real detail rather than a
denser resample of the same data, raise `--terrain-zoom` instead.

## What each output looks like

- **Profile chart**: elevation and speed vs. distance.
- **Interactive map**: the route on an OpenStreetMap base layer (open the `.html` in a browser).
- **3D elevation plot**: the track drawn in 3D (lon/lat/elevation), colored by elevation.
- **Terrain relief**: a shaded, matte "3D-printed model" style terrain block built from real
  elevation data, with the track painted on as a solid orange line and start/end markers.
- **Walk video**: the terrain relief with the path progressively revealed, a marker moving along
  it, a chase camera that follows and rotates to face the direction of travel, a distance/
  elevation HUD, and an elevation-profile strip — pulling back to a full overview in the last
  ~20% of the video.

## Dependencies

`gpxpy`, `matplotlib`, `folium`, `numpy`, `certifi`, `Pillow`, `imageio`, `imageio-ffmpeg` — all
listed in `requirements.txt`.
