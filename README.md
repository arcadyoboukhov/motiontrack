# Video Player — Custom Motion Tracking (Research)

A lightweight Python video player designed for interactive motion-tracking annotation and research. It pairs OpenCV frame extraction with a PySide6 GUI to let you place named tracking regions (hexagon/square + central dot), play video, auto-track across frames with multiple algorithms, and export per-frame coordinates to CSV for analysis.

## Features

- Generates a timeline CSV (`Frame_num`, `Time_ms`) for the source video.
- Interactive annotation: add named dots with an enclosing region (hexagon/square) and resize/move them.
- Per-frame in-memory storage and CSV export of named dot coordinates.
- Multiple automatic tracking algorithms: KLT (Lucas–Kanade), Template Matching, CSRT, KCF (if OpenCV supports them).
- Zoom, pan, playback speed control, and keyboard shortcuts for efficient annotation.
- Fallback console mode if UI or OpenCV is unavailable.

## Dependencies

- Python 3.8+
- numpy
- opencv-python
- PySide6

Recommended install (virtualenv):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
pip install numpy opencv-python PySide6
```

## Quick Start

1. Place your video in a folder and run:

```bash
python motiontrack.py path/to/video.mp4
```

2. If you omit the path, the script attempts a PySide6 file dialog; if that fails it prompts for a console path.
3. The script writes a timeline CSV next to the video (same name, `.csv` extension). Open the GUI to annotate and save per-frame dot coordinates.

## Running (examples)

- Create timeline CSV and open UI (typical):

```bash
python motiontrack.py sample_video.mp4
```

- Console fallback only (no OpenCV UI): the script prints file info and frame count if OpenCV is missing.

## CSV format

The script writes and edits a CSV alongside the video. Columns include:

- `Frame_num`, `Time_ms` — required timeline columns
- `dot_<NAME>_X`, `dot_<NAME>_Y` — main dot coordinates per named annotation
- `Hexagon_<NAME>_Point_<N>_X` / `_Y` — six hexagon corner points for the region (if present)
- `Sqaure_<NAME>_...` — (the code currently uses the spelling `Sqaure_` for square-region columns)

Example header (simplified):

```
Frame_num,Time_ms,dot_ball_X,dot_ball_Y,Hexagon_ball_Point_1_X,Hexagon_ball_Point_1_Y,...
```

Example row for frame 0 (values shown for illustration):

```
0,0,123.456,234.567,110.000,100.000,...
```

Notes:
- Empty cells mean the dot is not present for that frame.
- The GUI stores per-frame positions in memory and writes them into CSV on Save.

## UI controls & shortcuts

- Mouse:
  - Left-click inside the frame: add a new named dot / select + drag to move / drag handles to resize.
  - Right-click over a dot: remove the dot.
  - Mouse wheel: zoom in/out.
- Keyboard:
  - Space: toggle play/pause
  - `T`: toggle tracking on/off
  - `Q` / `E`: decrease / increase playback speed
  - `W` / `A` / `S` / `D` or Arrow keys: pan the view
  - `Ctrl+C`: copy selected dot
  - `Ctrl+V`: paste duplicate of copied dot
  - `Ctrl+S`: save CSV

## Tracking algorithms

- KLT: Lucas–Kanade sparse optical flow using `cv2.calcOpticalFlowPyrLK` (tracks the central point).
- Template Matching: `cv2.matchTemplate` on a search window around the last region.
- CSRT / KCF: OpenCV object trackers created via factory names (falls back to `cv2.legacy` when necessary).

Behavior notes:
- The UI attempts to keep regions inside frame bounds and clamps coordinates.
- Tracking runs per frame when enabled and updates in-memory frame positions.

## Developer notes

- The GUI code is in `motiontrack.py` and uses PySide6 for rendering and event handling.
- CSV creation: `create_video_timeline_csv()` writes the initial `Frame_num`/`Time_ms` rows.
- There is a small naming oddity in the CSV column prefixes: `Sqaure_` (typo) — preserved to match existing code.
- `mark_csv_dirty()` is a placeholder (no-op) and can be implemented to enable autosave/flush behavior.

## Known issues & improvements

- Consider implementing `mark_csv_dirty()` to auto-flush CSV changes at intervals.
- Add unit tests and a small sample video for integration testing.
- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Improve UI labeling and add an export options dialog (format choices, precision, etc.).

## Contributing

Contributions welcome — open an issue or PR with proposed changes. Please include clear tests or reproduction steps for behavioral changes.

## License

MIT License — feel free to reuse and adapt for research and experiments.
