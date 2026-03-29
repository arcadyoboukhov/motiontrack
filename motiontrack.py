from __future__ import annotations

import argparse
import csv
import math
import numpy as np
import os
import sys
import threading
import time
from pathlib import Path
##############################################################################
# SELECT VIDEO PATH FOR MOTION TRACKING

# Constants for video processing and UI behavior    
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
DOT_RADIUS = 6
BOX_HALF_SIZE = 24
SIDE_HIT_TOLERANCE = 8
CORNER_HIT_TOLERANCE = 10
DRAG_THRESHOLD = 3
ZOOM_MIN = 0.5
ZOOM_MAX = 13.0
ZOOM_STEP = 1.1
PAN_STEP_PIXELS = 40
MAX_RENDER_WIDTH = 4096
MAX_RENDER_HEIGHT = 4096
MAX_RENDER_PIXELS = 16_000_000
PASTE_GAP_PIXELS = 12
CSV_FLUSH_INTERVAL_SECONDS = 10.0


class UIUnavailableError(RuntimeError):
	pass

# select video file 
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Select a video file (.mp4, .avi, .mov) for motion tracking."
	)
	parser.add_argument(
		"video_path",
		nargs="?",
		help="Path to input video (.mp4, .avi, .mov).",
	)
	return parser.parse_args()


def is_allowed_video_file(file_path: str) -> bool:
	return Path(file_path).suffix.lower() in ALLOWED_EXTENSIONS


def validate_video_path(file_path: str) -> str:
	if not file_path:
		raise ValueError("No file path provided.")

	normalized = os.path.abspath(os.path.expanduser(file_path.strip().strip('"')))

	if not os.path.exists(normalized):
		raise FileNotFoundError(f"File does not exist: {normalized}")
	if not os.path.isfile(normalized):
		raise ValueError(f"Not a file: {normalized}")
	if not is_allowed_video_file(normalized):
		allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
		raise ValueError(f"Unsupported file type. Allowed extensions: {allowed}")

	return normalized


def pick_file_with_pyside6() -> str | None:
	try:
		from PySide6.QtWidgets import QApplication, QFileDialog
	except Exception:
		return None

	app = QApplication.instance()
	created_app = False
	if app is None:
		try:
			app = QApplication(sys.argv)
			created_app = True
		except Exception:
			return None

	try:
		selected, _ = QFileDialog.getOpenFileName(
			None,
			"Select video file",
			"",
			"Video files (*.mp4 *.avi *.mov);;MP4 (*.mp4);;AVI (*.avi);;MOV (*.mov);;All files (*)",
		)
		return selected or None
	except Exception:
		return None
	finally:
		if created_app and app is not None:
			app.quit()


def prompt_file_from_console() -> str:
	return input("Enter path to video file (.mp4, .avi, .mov): ").strip()


def get_video_path(initial_arg: str | None) -> str:
	if initial_arg:
		return validate_video_path(initial_arg)

	selected = pick_file_with_pyside6()
	if selected:
		return validate_video_path(selected)

	print("PySide6 file dialog is unavailable or failed. Falling back to console input.")
	return validate_video_path(prompt_file_from_console())

# Format seconds into MM:SS or HH:MM:SS for display in UI.
def format_time(seconds: float) -> str:
	seconds = max(0, int(seconds))
	minutes, secs = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	if hours > 0:
		return f"{hours:02d}:{minutes:02d}:{secs:02d}"
	return f"{minutes:02d}:{secs:02d}"

# Process video frames using OpenCV if available, otherwise validate file and print basic info.
def process_video_console(video_path: str) -> None:
	try:
		import cv2
	except ImportError:
		file_size = os.path.getsize(video_path)
		print("OpenCV not found. Skipping frame processing fallback mode.")
		print(f"Video file is valid: {video_path}")
		print(f"File size (bytes): {file_size}")
		print("To enable frame processing, install OpenCV: pip install opencv-python")
		return

	capture = cv2.VideoCapture(video_path)
	if not capture.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	try:
		frame_count = 0
		while True:
			success, _frame = capture.read()
			if not success:
				break
			frame_count += 1

		if frame_count == 0:
			raise RuntimeError("Video opened but no frames were read.")

		print(f"Video loaded successfully: {video_path}")
		print(f"Total frames read: {frame_count}")
	finally:
		capture.release()
##############################################################################
# Make CSV timeline for video frames with frame number and timestamp in milliseconds.

def create_video_timeline_csv(video_path: str) -> tuple[str, int]:
	try:
		import cv2
	except ImportError as exc:
		raise RuntimeError(
			"OpenCV is required to create timeline CSV files. Install with: pip install opencv-python"
		) from exc

	capture = cv2.VideoCapture(video_path)
	if not capture.isOpened():
		raise RuntimeError(f"Could not open video for CSV generation: {video_path}")

	fps = float(capture.get(cv2.CAP_PROP_FPS))
	if fps <= 0:
		fps = 30.0
	total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

	csv_path = str(Path(video_path).with_suffix(".csv"))

	try:
		if total_frames > 0:
			frame_count = total_frames
		else:
			while True:
				success, _frame = capture.read()
				if not success:
					break
				frame_count += 1

		if frame_count == 0:
			raise RuntimeError("Video opened but no frames were read during CSV generation.")

		if os.path.exists(csv_path):
			print(f"CSV file already exists: {csv_path}")
			return csv_path, frame_count

		with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(["Frame_num", "Time_ms"])
			for frame_index in range(frame_count):
				time_ms = (frame_index / fps) * 1000.0
				writer.writerow([frame_index, int(round(time_ms))])

		print(f"CSV created: {csv_path}")
		return csv_path, frame_count
	finally:
		capture.release()


##############################################################################
# Video player UI implementation using PySide6 and OpenCV for frame extraction.
    
def run_video_player_ui(video_path: str, csv_path: str) -> None:
	try:
		from PySide6.QtCore import QTimer, Qt, QPoint
		from PySide6.QtGui import QBrush, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygon, QShortcut
		from PySide6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget
	except Exception as exc:
		raise UIUnavailableError(f"PySide6 UI is unavailable: {exc}") from exc

	try:
		import cv2
	except ImportError as exc:
		raise UIUnavailableError(
			"OpenCV is required for video UI playback. Install with: pip install opencv-python"
		) from exc

	app = QApplication.instance()
	if app is None:
		app = QApplication(sys.argv)

	capture = cv2.VideoCapture(video_path)
	if not capture.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = float(capture.get(cv2.CAP_PROP_FPS))
	if fps <= 0:
		fps = 30.0
	if total_frames <= 0:
		capture.release()
		raise RuntimeError("Video opened but contains no readable frames.")

	def load_csv_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
		with open(path, "r", newline="", encoding="utf-8") as csv_file:
			reader = csv.DictReader(csv_file)
			fieldnames = list(reader.fieldnames or ["Frame_num", "Time_ms"])
			rows = [dict(row) for row in reader]
		return fieldnames, rows

	def write_csv_rows(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
		with open(path, "w", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(rows)
# UI class for video playback and annotation, with mouse and keyboard interaction handling.
	class VideoCanvasWidget(QWidget):
		def __init__(self, ui: "VideoPlayerUI") -> None:
			super().__init__(ui.root)
			self.ui = ui
			self.setMouseTracking(True)
			self.setFocusPolicy(Qt.StrongFocus)

		def paintEvent(self, _event) -> None:
			painter = QPainter(self)
			painter.fillRect(self.rect(), QColor("black"))
			if self.ui.photo is not None:
				painter.drawPixmap(self.ui.image_offset_x, self.ui.image_offset_y, self.ui.photo)
			self.ui.paint_annotations(painter)

		def mousePressEvent(self, event) -> None:
			if event.button() == Qt.LeftButton:
				self.ui.on_right_click(event)
			elif event.button() == Qt.RightButton:
				self.ui.on_remove_click(event)

		def mouseMoveEvent(self, event) -> None:
			if event.buttons() & Qt.LeftButton:
				self.ui.on_right_drag(event)

		def mouseReleaseEvent(self, event) -> None:
			if event.button() == Qt.LeftButton:
				self.ui.on_right_release(event)

		def wheelEvent(self, event) -> None:
			self.ui.on_mouse_wheel(event)

		def keyPressEvent(self, event) -> None:
			self.ui.on_key_press(event)
# Main UI class that manages video playback, annotation, and user interactions.
	class VideoPlayerUI:
		def __init__(self) -> None:
			self.root = QWidget()
			self.root.setWindowTitle("MotionTrack Video Player")
			self.root.resize(980, 700)

			self.capture = capture
			self.total_frames = total_frames
			self.fps = fps
			self.current_frame_index = 0
			self.paused = False
			self.speed_options = [0.25, 0.5] + [float(value) for value in range(1, 21)]
			self.default_speed = 1.0
			self.default_speed_index = self.speed_options.index(self.default_speed)
			self.playback_speed = self.default_speed
			self.tracking_algorithms = ["KLT", "Template Matching", "CSRT", "KCF"]
			self.selected_tracking_algorithm = self.tracking_algorithms[0]
			self.tracking_enabled = False
			self.playback_frame_accumulator = 0.0
			self.last_update_time = time.perf_counter()
			self.is_dragging_slider = False
			self.pending_seek: int | None = None
			self.photo: QPixmap | None = None
			self.current_frame_bgr = None
			self.image_offset_x = 0
			self.image_offset_y = 0
			self.image_width = 0
			self.image_height = 0
			self.pan_x = 0
			self.pan_y = 0
			self.annotations: list[dict[str, float | str]] = []
			self.active_action: dict | None = None
			self.copy_buffer: dict[str, float | str] | None = None
			self.next_unknown_id = 1
			self.csv_path = csv_path
			self.csv_fieldnames, self.csv_rows = load_csv_rows(csv_path)
			self.dot_positions_memory: dict[int, dict[str, dict[str, float]]] = {}
			self.deleted_dot_names: set[str] = set()
			self.zoom_factor = 1.0
			self.current_scale = 1.0
			self.source_width = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1)
			self.source_height = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1)
			self.normalize_csv_rows()
			self.load_dots_from_csv()

			root_layout = QVBoxLayout(self.root)
			root_layout.setContentsMargins(10, 8, 10, 10)
			root_layout.setSpacing(8)

			self.video_canvas = VideoCanvasWidget(self)
			root_layout.addWidget(self.video_canvas, 1)

			controls = QWidget(self.root)
			controls_layout = QHBoxLayout(controls)
			controls_layout.setContentsMargins(0, 0, 0, 0)
			controls_layout.setSpacing(8)

			self.pause_button = QPushButton("Pause", controls)
			self.pause_button.setFixedWidth(80)
			self.pause_button.clicked.connect(self.toggle_pause)
			controls_layout.addWidget(self.pause_button)

			self.save_button = QPushButton("Save", controls)
			self.save_button.setFixedWidth(80)
			self.save_button.setStyleSheet("background-color: #4CAF50; color: white;")
			self.save_button.clicked.connect(self.save_to_csv)
			controls_layout.addWidget(self.save_button)

			self.track_button = QPushButton("Track: Off", controls)
			self.track_button.setFixedWidth(90)
			self.track_button.clicked.connect(self.toggle_tracking)
			controls_layout.addWidget(self.track_button)

			controls_layout.addWidget(QLabel("Algorithm", controls))
			self.algorithm_dropdown = QComboBox(controls)
			self.algorithm_dropdown.addItems(self.tracking_algorithms)
			self.algorithm_dropdown.currentTextChanged.connect(self.on_algorithm_changed)
			controls_layout.addWidget(self.algorithm_dropdown)

			controls_layout.addSpacing(8)
			controls_layout.addWidget(QLabel("Speed", controls))

			self.speed_scale = QSlider(Qt.Horizontal, controls)
			self.speed_scale.setRange(0, len(self.speed_options) - 1)
			self.speed_scale.setValue(self.default_speed_index)
			self.speed_scale.setFixedWidth(160)
			self.speed_scale.valueChanged.connect(lambda value: self.on_speed_change(str(value)))
			controls_layout.addWidget(self.speed_scale)

			self.speed_label = QLabel("1x", controls)
			controls_layout.addWidget(self.speed_label)

			controls_layout.addStretch(1)

			self.time_label = QLabel("00:00 / 00:00", controls)
			controls_layout.addWidget(self.time_label)

			root_layout.addWidget(controls)

			self.timeline = QSlider(Qt.Horizontal, self.root)
			self.timeline.setRange(0, self.total_frames - 1)
			self.timeline.valueChanged.connect(lambda value: self.on_seek(str(value)))
			self.timeline.sliderPressed.connect(self.on_slider_press)
			self.timeline.sliderReleased.connect(self.on_slider_release)
			root_layout.addWidget(self.timeline)

			QShortcut(QKeySequence("Ctrl+C"), self.root, activated=self.on_copy_hotkey)
			QShortcut(QKeySequence("Ctrl+V"), self.root, activated=self.on_paste_hotkey)
			QShortcut(QKeySequence("Ctrl+S"), self.root, activated=self.save_to_csv)

			self.root.keyPressEvent = self.on_key_press
			self.root.closeEvent = self.on_close_event

			self.show_frame_at(0)

			self.timer = QTimer(self.root)
			self.timer.timeout.connect(self.update_loop)
			self.timer.start(15)

			self.video_canvas.setFocus()

		def frame_to_photo(self, frame, scale: float) -> QPixmap:
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			h, w, _ = rgb.shape
			new_w = max(1, int(w * scale))
			new_h = max(1, int(h * scale))
			if new_w != w or new_h != h:
				rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
				h, w, _ = rgb.shape

			image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
			return QPixmap.fromImage(image)

		def event_xy(self, event) -> tuple[int, int]:
			position = event.position()
			return int(position.x()), int(position.y())

		def is_inside_image(self, x: int, y: int) -> bool:
			return (
				self.image_offset_x <= x <= self.image_offset_x + self.image_width
				and self.image_offset_y <= y <= self.image_offset_y + self.image_height
			)

		def update_image_offsets(self, canvas_w: int, canvas_h: int) -> None:
			if self.image_width <= canvas_w:
				self.pan_x = 0
				self.image_offset_x = (canvas_w - self.image_width) // 2
			else:
				min_x = canvas_w - self.image_width
				max_x = 0
				self.pan_x = max(min_x, min(self.pan_x, max_x))
				self.image_offset_x = self.pan_x

			if self.image_height <= canvas_h:
				self.pan_y = 0
				self.image_offset_y = (canvas_h - self.image_height) // 2
			else:
				min_y = canvas_h - self.image_height
				max_y = 0
				self.pan_y = max(min_y, min(self.pan_y, max_y))
				self.image_offset_y = self.pan_y

		def draw_current_frame(self) -> None:
			if self.photo is None:
				return

			canvas_w = max(1, self.video_canvas.width())
			canvas_h = max(1, self.video_canvas.height())
			self.image_width = self.photo.width()
			self.image_height = self.photo.height()
			self.update_image_offsets(canvas_w, canvas_h)
			self.video_canvas.update()
# Ensure annotation box and dot are within video frame bounds, adjusting position if necessary.
		def clamp_annotation_in_bounds(self, annotation: dict[str, float | str]) -> dict[str, float | str]:
			width = annotation["x2"] - annotation["x1"]
			height = annotation["y2"] - annotation["y1"]

			shift_x = 0.0
			shift_y = 0.0
			min_x = 0.0
			min_y = 0.0
			max_x = float(self.source_width - 1)
			max_y = float(self.source_height - 1)

			if annotation["x1"] < min_x:
				shift_x = min_x - annotation["x1"]
			elif annotation["x2"] > max_x:
				shift_x = max_x - annotation["x2"]

			if annotation["y1"] < min_y:
				shift_y = min_y - annotation["y1"]
			elif annotation["y2"] > max_y:
				shift_y = max_y - annotation["y2"]

			annotation["x1"] += shift_x
			annotation["x2"] += shift_x
			annotation["dot_x"] += shift_x
			annotation["y1"] += shift_y
			annotation["y2"] += shift_y
			annotation["dot_y"] += shift_y

			if width > max_x - min_x:
				annotation["x1"] = min_x
				annotation["x2"] = max_x
			if height > max_y - min_y:
				annotation["y1"] = min_y
				annotation["y2"] = max_y

			annotation["dot_x"], annotation["dot_y"] = self.clamp_to_image_coords(
				annotation["dot_x"], annotation["dot_y"]
			)
			return annotation

		def prompt_annotation_name(self, initial_name: str = "") -> str | None:
			name, ok = QInputDialog.getText(
				self.root,
				"Dot ID",
				"Enter dot ID name:",
				text=initial_name,
			)
			if not ok:
				return None
			name = name.strip()
			if not name:
				return None
			return name

		def is_name_taken(self, name: str, ignore_index: int | None = None) -> bool:
			target = name.strip()
			if not target:
				return False
			for index, annotation in enumerate(self.annotations):
				if ignore_index is not None and index == ignore_index:
					continue
				if str(annotation.get("name", "")).strip() == target:
					return True
			return False

		def make_unique_name(self, base_name: str, ignore_index: int | None = None) -> str:
			candidate = base_name.strip()
			if not candidate:
				candidate = self.generate_unknown_name()
			if not self.is_name_taken(candidate, ignore_index=ignore_index):
				return candidate

			while True:
				candidate = f"{candidate}_copy"
				if not self.is_name_taken(candidate, ignore_index=ignore_index):
					return candidate

		def generate_unknown_name(self) -> str:
			existing_names = {
				str(annotation.get("name", "")).strip()
				for annotation in self.annotations
			}

			while True:
				candidate = f"unknown_{self.next_unknown_id}"
				self.next_unknown_id += 1
				if candidate not in existing_names:
					return candidate

		def ensure_annotation_has_name(self, annotation: dict[str, float | str]) -> str:
			name = str(annotation.get("name", "")).strip()
			if name:
				return name
			name = self.generate_unknown_name()
			annotation["name"] = name
			return name

		def normalize_csv_rows(self) -> None:
			changed = False
			for required_field in ["Frame_num", "Time_ms"]:
				if required_field not in self.csv_fieldnames:
					self.csv_fieldnames.append(required_field)
					changed = True

			if len(self.csv_rows) < self.total_frames:
				for frame_index in range(len(self.csv_rows), self.total_frames):
					time_ms = int(round((frame_index / self.fps) * 1000.0))
					self.csv_rows.append(
						{
							"Frame_num": str(frame_index),
							"Time_ms": str(time_ms),
						}
					)
					changed = True

			if len(self.csv_rows) > self.total_frames:
				self.csv_rows = self.csv_rows[: self.total_frames]
				changed = True

			for index, row in enumerate(self.csv_rows):
				frame_num = str(index)
				if str(row.get("Frame_num", "")) != frame_num:
					row["Frame_num"] = frame_num
					changed = True
				if not str(row.get("Time_ms", "")).strip():
					row["Time_ms"] = str(int(round((index / self.fps) * 1000.0)))
					changed = True
				for fieldname in self.csv_fieldnames:
					if fieldname not in row:
						row[fieldname] = ""
						changed = True

			if changed:
				self.mark_csv_dirty()

		def load_dots_from_csv(self) -> None:
			"""Load dots from existing CSV file and jump to their first appearance."""
			dot_names = set()
			for fieldname in self.csv_fieldnames:
				if fieldname.startswith("dot_") and fieldname.endswith("_X"):
					dot_name = fieldname[4:-2]
					dot_names.add(dot_name)

			for frame_index, row in enumerate(self.csv_rows):
				for dot_name in dot_names:
					x_col = f"dot_{dot_name}_X"
					y_col = f"dot_{dot_name}_Y"
					hex_columns = [
						(f"Hexagon_{dot_name}_Point_{point_index}_X", f"Hexagon_{dot_name}_Point_{point_index}_Y")
						for point_index in range(1, 7)
					]
					ulx_col = f"Sqaure_{dot_name}_Upper_Left_X"
					uly_col = f"Sqaure_{dot_name}_Upper_Left_Y"
					urx_col = f"Sqaure_{dot_name}_Upper_Right_X"
					ury_col = f"Sqaure_{dot_name}_Upper_Right_Y"
					llx_col = f"Sqaure_{dot_name}_Lower_Left_X"
					lly_col = f"Sqaure_{dot_name}_Lower_Left_Y"
					lrx_col = f"Sqaure_{dot_name}_Lower_Right_X"
					lry_col = f"Sqaure_{dot_name}_Lower_Right_Y"
					x_str = str(row.get(x_col, "")).strip()
					y_str = str(row.get(y_col, "")).strip()
					hex_value_pairs = [
						(str(row.get(hex_x_col, "")).strip(), str(row.get(hex_y_col, "")).strip())
						for hex_x_col, hex_y_col in hex_columns
					]
					ulx_str = str(row.get(ulx_col, "")).strip()
					uly_str = str(row.get(uly_col, "")).strip()
					urx_str = str(row.get(urx_col, "")).strip()
					ury_str = str(row.get(ury_col, "")).strip()
					llx_str = str(row.get(llx_col, "")).strip()
					lly_str = str(row.get(lly_col, "")).strip()
					lrx_str = str(row.get(lrx_col, "")).strip()
					lry_str = str(row.get(lry_col, "")).strip()
					if x_str and y_str:
						try:
							dot_x = float(x_str)
							dot_y = float(y_str)
							if frame_index not in self.dot_positions_memory:
								self.dot_positions_memory[frame_index] = {}

							has_hexagon = all(hex_x and hex_y for hex_x, hex_y in hex_value_pairs)
							has_square = all([
								ulx_str, uly_str, urx_str, ury_str, llx_str, lly_str, lrx_str, lry_str
							])
							if has_hexagon:
								hex_x_values: list[float] = []
								hex_y_values: list[float] = []
								for hex_x_str, hex_y_str in hex_value_pairs:
									hex_x_values.append(float(hex_x_str))
									hex_y_values.append(float(hex_y_str))
								x1 = min(hex_x_values)
								x2 = max(hex_x_values)
								y1 = min(hex_y_values)
								y2 = max(hex_y_values)
							elif has_square:
								ulx = float(ulx_str)
								uly = float(uly_str)
								urx = float(urx_str)
								ury = float(ury_str)
								llx = float(llx_str)
								lly = float(lly_str)
								lrx = float(lrx_str)
								lry = float(lry_str)
								x1 = min(ulx, llx)
								x2 = max(urx, lrx)
								y1 = min(uly, ury)
								y2 = max(lly, lry)
							else:
								box_half = BOX_HALF_SIZE
								x1 = dot_x - box_half
								x2 = dot_x + box_half
								y1 = dot_y - box_half
								y2 = dot_y + box_half

							x1 = max(0.0, min(x1, float(self.source_width - 1)))
							x2 = max(0.0, min(x2, float(self.source_width - 1)))
							y1 = max(0.0, min(y1, float(self.source_height - 1)))
							y2 = max(0.0, min(y2, float(self.source_height - 1)))

							self.dot_positions_memory[frame_index][dot_name] = {
								"dot_x": dot_x,
								"dot_y": dot_y,
								"x1": x1,
								"y1": y1,
								"x2": x2,
								"y2": y2,
							}
						except (ValueError, TypeError):
							pass

			first_frame_with_dots = None

			for dot_name in sorted(dot_names):
				dot_x = None
				dot_y = None
				x1 = None
				y1 = None
				x2 = None
				y2 = None
				first_frame_for_dot = None

				for frame_index in sorted(self.dot_positions_memory.keys()):
					if dot_name in self.dot_positions_memory[frame_index]:
						position = self.dot_positions_memory[frame_index][dot_name]
						dot_x = position["dot_x"]
						dot_y = position["dot_y"]
						x1 = position["x1"]
						y1 = position["y1"]
						x2 = position["x2"]
						y2 = position["y2"]
						first_frame_for_dot = frame_index
						break

				if dot_x is not None and dot_y is not None and x1 is not None and y1 is not None and x2 is not None and y2 is not None:
					if first_frame_with_dots is None or first_frame_for_dot < first_frame_with_dots:
						first_frame_with_dots = first_frame_for_dot

					annotation = {
						"dot_x": dot_x,
						"dot_y": dot_y,
						"x1": x1,
						"x2": x2,
						"y1": y1,
						"y2": y2,
						"name": dot_name,
					}
					self.annotations.append(annotation)

					if dot_name.startswith("unknown_"):
						try:
							dot_id = int(dot_name[8:])
							self.next_unknown_id = max(self.next_unknown_id, dot_id + 1)
						except (ValueError, IndexError):
							pass

			if first_frame_with_dots is not None:
				self.current_frame_index = first_frame_with_dots

		def save_to_csv(self) -> None:
			"""Save all dot positions from memory to CSV file."""
			original_style = self.save_button.styleSheet()
			self.save_button.setText("Saving...")
			self.save_button.setStyleSheet("background-color: #FF9800; color: white;")

			for dot_name in self.deleted_dot_names:
				for frame_data in self.dot_positions_memory.values():
					frame_data.pop(dot_name, None)

			for dot_name in self.deleted_dot_names:
				x_column = f"dot_{dot_name}_X"
				y_column = f"dot_{dot_name}_Y"
				hex_columns = [
					f"Hexagon_{dot_name}_Point_{point_index}_{axis}"
					for point_index in range(1, 7)
					for axis in ("X", "Y")
				]
				if x_column in self.csv_fieldnames:
					self.csv_fieldnames.remove(x_column)
				if y_column in self.csv_fieldnames:
					self.csv_fieldnames.remove(y_column)
				for hex_column in hex_columns:
					if hex_column in self.csv_fieldnames:
						self.csv_fieldnames.remove(hex_column)
				for row in self.csv_rows:
					row.pop(x_column, None)
					row.pop(y_column, None)
					for hex_column in hex_columns:
						row.pop(hex_column, None)
					row.pop(f"Sqaure_{dot_name}_Upper_Left_X", None)
					row.pop(f"Sqaure_{dot_name}_Upper_Left_Y", None)
					row.pop(f"Sqaure_{dot_name}_Upper_Right_X", None)
					row.pop(f"Sqaure_{dot_name}_Upper_Right_Y", None)
					row.pop(f"Sqaure_{dot_name}_Lower_Left_X", None)
					row.pop(f"Sqaure_{dot_name}_Lower_Left_Y", None)
					row.pop(f"Sqaure_{dot_name}_Lower_Right_X", None)
					row.pop(f"Sqaure_{dot_name}_Lower_Right_Y", None)

			self.deleted_dot_names.clear()

			dot_names = set()
			for annotation in self.annotations:
				dot_name = str(annotation.get("name", "")).strip()
				if dot_name:
					dot_names.add(dot_name)
					self.ensure_dot_columns(dot_name)

			for frame_data in self.dot_positions_memory.values():
				for dot_name in frame_data.keys():
					if dot_name:
						dot_names.add(dot_name)
						self.ensure_dot_columns(dot_name)

			for frame_index in range(len(self.csv_rows)):
				row = self.csv_rows[frame_index]
				for dot_name in dot_names:
					x_column = f"dot_{dot_name}_X"
					y_column = f"dot_{dot_name}_Y"
					hex_columns = [
						f"Hexagon_{dot_name}_Point_{point_index}_{axis}"
						for point_index in range(1, 7)
						for axis in ("X", "Y")
					]

					row[x_column] = ""
					row[y_column] = ""
					for hex_column in hex_columns:
						row[hex_column] = ""

			for dot_name in dot_names:
				frames_with_position = sorted(
					frame_idx
					for frame_idx, frame_data in self.dot_positions_memory.items()
					if dot_name in frame_data
				)
				if not frames_with_position:
					continue

				for i, start_frame in enumerate(frames_with_position):
					start_data = self.dot_positions_memory[start_frame][dot_name]
					start_x = start_data["dot_x"]
					start_y = start_data["dot_y"]
					start_x1 = start_data["x1"]
					start_y1 = start_data["y1"]
					start_x2 = start_data["x2"]
					start_y2 = start_data["y2"]
					end_frame = frames_with_position[i + 1] if i + 1 < len(frames_with_position) else start_frame + 1

					x_column = f"dot_{dot_name}_X"
					y_column = f"dot_{dot_name}_Y"
					hex_columns = [
						(f"Hexagon_{dot_name}_Point_{point_index}_X", f"Hexagon_{dot_name}_Point_{point_index}_Y")
						for point_index in range(1, 7)
					]
					temp_annotation = {
						"x1": start_x1,
						"y1": start_y1,
						"x2": start_x2,
						"y2": start_y2,
					}
					hex_points = self.get_hexagon_points(temp_annotation)
					for frame_idx in range(start_frame, end_frame):
						if 0 <= frame_idx < len(self.csv_rows):
							row = self.csv_rows[frame_idx]
							row[x_column] = f"{start_x:.3f}"
							row[y_column] = f"{start_y:.3f}"
							for (hex_x_column, hex_y_column), (hex_x, hex_y) in zip(hex_columns, hex_points):
								row[hex_x_column] = f"{hex_x:.3f}"
								row[hex_y_column] = f"{hex_y:.3f}"
							row.pop(f"Sqaure_{dot_name}_Upper_Left_X", None)
							row.pop(f"Sqaure_{dot_name}_Upper_Left_Y", None)
							row.pop(f"Sqaure_{dot_name}_Upper_Right_X", None)
							row.pop(f"Sqaure_{dot_name}_Upper_Right_Y", None)
							row.pop(f"Sqaure_{dot_name}_Lower_Left_X", None)
							row.pop(f"Sqaure_{dot_name}_Lower_Left_Y", None)
							row.pop(f"Sqaure_{dot_name}_Lower_Right_X", None)
							row.pop(f"Sqaure_{dot_name}_Lower_Right_Y", None)

			self.csv_fieldnames = [
				fieldname
				for fieldname in self.csv_fieldnames
				if not (fieldname.startswith("Sqaure_") and (fieldname.endswith("_X") or fieldname.endswith("_Y")))
			]

			write_csv_rows(self.csv_path, self.csv_fieldnames, self.csv_rows)

			self.save_button.setStyleSheet("background-color: #2E7D32; color: white;")
			self.save_button.setText("Saved")
			QTimer.singleShot(1000, lambda: (self.save_button.setStyleSheet(original_style), self.save_button.setText("Save")))

		def ensure_dot_columns(self, dot_name: str) -> tuple[str, str]:
			dot_name = dot_name.strip()
			if not dot_name:
				return "", ""

			x_column = f"dot_{dot_name}_X"
			y_column = f"dot_{dot_name}_Y"
			hex_columns = [
				f"Hexagon_{dot_name}_Point_{point_index}_{axis}"
				for point_index in range(1, 7)
				for axis in ("X", "Y")
			]
			added = False
			if x_column not in self.csv_fieldnames:
				self.csv_fieldnames.append(x_column)
				added = True
			if y_column not in self.csv_fieldnames:
				self.csv_fieldnames.append(y_column)
				added = True
			for hex_column in hex_columns:
				if hex_column not in self.csv_fieldnames:
					self.csv_fieldnames.append(hex_column)
					added = True
			if added:
				for row in self.csv_rows:
					row.setdefault(x_column, "")
					row.setdefault(y_column, "")
					for hex_column in hex_columns:
						row.setdefault(hex_column, "")
			return x_column, y_column

		def update_memory_for_current_frame(self) -> None:
			"""Update in-memory dot positions for current frame."""
			frame_index = self.current_frame_index
			if frame_index < 0 or frame_index >= len(self.csv_rows):
				return

			if frame_index not in self.dot_positions_memory:
				self.dot_positions_memory[frame_index] = {}

			for annotation in self.annotations:
				dot_name = str(annotation.get("name", "")).strip()
				if not dot_name:
					continue
				self.dot_positions_memory[frame_index][dot_name] = {
					"dot_x": float(annotation["dot_x"]),
					"dot_y": float(annotation["dot_y"]),
					"x1": float(annotation["x1"]),
					"y1": float(annotation["y1"]),
					"x2": float(annotation["x2"]),
					"y2": float(annotation["y2"]),
				}

		def load_current_frame_positions(self) -> None:
			"""Load dot positions from memory for current frame if they exist."""
			frame_index = self.current_frame_index
			if frame_index < 0 or frame_index >= len(self.csv_rows):
				return

			if frame_index not in self.dot_positions_memory:
				return

			frame_data = self.dot_positions_memory[frame_index]
			for annotation in self.annotations:
				dot_name = str(annotation.get("name", "")).strip()
				if not dot_name or dot_name not in frame_data:
					continue

				position = frame_data[dot_name]
				dot_x = position["dot_x"]
				dot_y = position["dot_y"]

				annotation["dot_x"] = dot_x
				annotation["dot_y"] = dot_y
				annotation["x1"] = position.get("x1", dot_x - BOX_HALF_SIZE)
				annotation["y1"] = position.get("y1", dot_y - BOX_HALF_SIZE)
				annotation["x2"] = position.get("x2", dot_x + BOX_HALF_SIZE)
				annotation["y2"] = position.get("y2", dot_y + BOX_HALF_SIZE)

		def duplicate_active_annotation(self) -> None:
			if not self.active_action or self.active_action.get("type") != "dot":
				return
			if self.copy_buffer is None:
				return

			index = self.active_action["index"]
			if index < 0 or index >= len(self.annotations):
				return

			current = self.annotations[index]
			buffer = self.copy_buffer

			rel_x1 = buffer["x1"] - buffer["dot_x"]
			rel_x2 = buffer["x2"] - buffer["dot_x"]
			rel_y1 = buffer["y1"] - buffer["dot_y"]
			rel_y2 = buffer["y2"] - buffer["dot_y"]

			gap = PASTE_GAP_PIXELS / max(self.current_scale, 1e-6)
			new_dot_x = current["x2"] + gap
			new_dot_y = current["dot_y"]

			new_annotation = {
				"dot_x": new_dot_x,
				"dot_y": new_dot_y,
				"x1": new_dot_x + rel_x1,
				"y1": new_dot_y + rel_y1,
				"x2": new_dot_x + rel_x2,
				"y2": new_dot_y + rel_y2,
				"name": self.make_unique_name(
					f"{str(buffer.get('name', '')).strip() or self.generate_unknown_name()}_copy"
				),
			}

			new_annotation = self.clamp_annotation_in_bounds(new_annotation)
			self.annotations.append(new_annotation)
			self.ensure_dot_columns(str(new_annotation.get("name", "")).strip())
			self.draw_annotations()

		def clamp_to_image_coords(self, x: float, y: float) -> tuple[float, float]:
			max_x = max(0.0, float(self.source_width - 1))
			max_y = max(0.0, float(self.source_height - 1))
			return max(0.0, min(x, max_x)), max(0.0, min(y, max_y))

		def canvas_to_image_coords(self, x: int, y: int) -> tuple[float, float]:
			image_x = (x - self.image_offset_x) / max(self.current_scale, 1e-6)
			image_y = (y - self.image_offset_y) / max(self.current_scale, 1e-6)
			return self.clamp_to_image_coords(image_x, image_y)

		def image_to_canvas_coords(self, x: float, y: float) -> tuple[int, int]:
			canvas_x = int(self.image_offset_x + x * self.current_scale)
			canvas_y = int(self.image_offset_y + y * self.current_scale)
			return canvas_x, canvas_y

		def get_hexagon_points(self, annotation: dict[str, float | str]) -> list[tuple[float, float]]:
			x1 = float(annotation["x1"])
			y1 = float(annotation["y1"])
			x2 = float(annotation["x2"])
			y2 = float(annotation["y2"])
			width = max(0.0, x2 - x1)
			height = max(0.0, y2 - y1)
			offset_x = width * 0.25
			mid_y = y1 + (height * 0.5)

			return [
				(x1 + offset_x, y1),
				(x2 - offset_x, y1),
				(x2, mid_y),
				(x2 - offset_x, y2),
				(x1 + offset_x, y2),
				(x1, mid_y),
			]

		def point_to_segment_distance(self, px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
			dx = x2 - x1
			dy = y2 - y1
			if dx == 0 and dy == 0:
				return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
			t = ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)
			t = max(0.0, min(1.0, t))
			proj_x = x1 + t * dx
			proj_y = y1 + t * dy
			return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

		def find_dot_index_at(self, x: int, y: int) -> int | None:
			for index in range(len(self.annotations) - 1, -1, -1):
				dot_x, dot_y = self.image_to_canvas_coords(
					self.annotations[index]["dot_x"],
					self.annotations[index]["dot_y"],
				)
				if (dot_x - x) ** 2 + (dot_y - y) ** 2 <= (DOT_RADIUS + 2) ** 2:
					return index
			return None

		def find_side_at(self, x: int, y: int) -> tuple[int, str] | None:
			for index in range(len(self.annotations) - 1, -1, -1):
				annotation = self.annotations[index]
				hex_points = [self.image_to_canvas_coords(px, py) for px, py in self.get_hexagon_points(annotation)]
				edges = [
					("top", hex_points[0], hex_points[1]),
					("upper_right", hex_points[1], hex_points[2]),
					("lower_right", hex_points[2], hex_points[3]),
					("bottom", hex_points[3], hex_points[4]),
					("lower_left", hex_points[4], hex_points[5]),
					("upper_left", hex_points[5], hex_points[0]),
				]
				for side_name, start_point, end_point in edges:
					distance = self.point_to_segment_distance(
						x,
						y,
						start_point[0],
						start_point[1],
						end_point[0],
						end_point[1],
					)
					if distance <= SIDE_HIT_TOLERANCE:
						return index, side_name
			return None

		def find_corner_at(self, x: int, y: int) -> tuple[int, str] | None:
			for index in range(len(self.annotations) - 1, -1, -1):
				annotation = self.annotations[index]
				hex_points = [self.image_to_canvas_coords(px, py) for px, py in self.get_hexagon_points(annotation)]
				vertices = [
					("top_left", hex_points[0][0], hex_points[0][1]),
					("top_right", hex_points[1][0], hex_points[1][1]),
					("right", hex_points[2][0], hex_points[2][1]),
					("bottom_right", hex_points[3][0], hex_points[3][1]),
					("bottom_left", hex_points[4][0], hex_points[4][1]),
					("left", hex_points[5][0], hex_points[5][1]),
				]
				for corner_name, corner_x, corner_y in vertices:
					if (corner_x - x) ** 2 + (corner_y - y) ** 2 <= CORNER_HIT_TOLERANCE ** 2:
						return index, corner_name
			return None

		def draw_annotations(self) -> None:
			self.video_canvas.update()

		def paint_annotations(self, painter: QPainter) -> None:
			if self.photo is None:
				return

			fill_brush = QBrush(QColor(30, 144, 255, 128))
			outline_pen = QPen(QColor("#66b3ff"))
			outline_pen.setWidth(2)
			dot_pen = QPen(QColor("white"))
			dot_pen.setWidth(1)
			handle_pen = QPen(QColor("#1e90ff"))
			handle_pen.setWidth(1)
			font = QFont("Segoe UI", 9)
			font.setBold(True)

			for annotation in self.annotations:
				x1, y1 = self.image_to_canvas_coords(annotation["x1"], annotation["y1"])
				x2, y2 = self.image_to_canvas_coords(annotation["x2"], annotation["y2"])
				hex_points = [self.image_to_canvas_coords(px, py) for px, py in self.get_hexagon_points(annotation)]

				polygon = QPolygon([QPoint(point_x, point_y) for point_x, point_y in hex_points])
				painter.setBrush(fill_brush)
				painter.setPen(outline_pen)
				painter.drawPolygon(polygon)

				dot_x, dot_y = self.image_to_canvas_coords(annotation["dot_x"], annotation["dot_y"])
				painter.setBrush(QBrush(QColor("red")))
				painter.setPen(dot_pen)
				painter.drawEllipse(dot_x - DOT_RADIUS, dot_y - DOT_RADIUS, DOT_RADIUS * 2, DOT_RADIUS * 2)

				handle_radius = 3
				painter.setBrush(QBrush(QColor("white")))
				painter.setPen(handle_pen)
				for corner_x, corner_y in hex_points:
					painter.drawEllipse(corner_x - handle_radius, corner_y - handle_radius, handle_radius * 2, handle_radius * 2)

				name = str(annotation.get("name", "")).strip()
				if name:
					box_height = max(1, y2 - y1)
					inside_label_y = dot_y + DOT_RADIUS + 2
					if box_height >= 28 and inside_label_y <= y2 - 6:
						text_y = inside_label_y
					else:
						text_y = y2 + 10
					painter.setPen(QPen(QColor("white")))
					painter.setFont(font)
					painter.drawText(dot_x - 120, text_y, 240, 24, Qt.AlignHCenter | Qt.AlignTop, name)

		def add_annotation(self, x: int, y: int) -> None:
			annotation_name = ""
			while True:
				annotation_name = self.prompt_annotation_name(annotation_name)
				if annotation_name is None:
					return
				if self.is_name_taken(annotation_name):
					QMessageBox.critical(self.root, "Duplicate Dot ID", f"Dot ID '{annotation_name}' already exists. Use a unique name.")
					continue
				break

			dot_x, dot_y = self.canvas_to_image_coords(x, y)
			box_half = BOX_HALF_SIZE / max(self.current_scale, 1e-6)
			x1 = dot_x - box_half
			x2 = dot_x + box_half
			y1 = dot_y - box_half
			y2 = dot_y + box_half
			x1, y1 = self.clamp_to_image_coords(x1, y1)
			x2, y2 = self.clamp_to_image_coords(x2, y2)

			if x2 <= x1:
				x2 = min(float(self.source_width - 1), x1 + (8 / max(self.current_scale, 1e-6)))
			if y2 <= y1:
				y2 = min(float(self.source_height - 1), y1 + (8 / max(self.current_scale, 1e-6)))

			self.annotations.append(
				{
					"dot_x": dot_x,
					"dot_y": dot_y,
					"x1": x1,
					"y1": y1,
					"x2": x2,
					"y2": y2,
					"name": annotation_name,
				}
			)
			self.ensure_dot_columns(annotation_name)
			self.update_memory_for_current_frame()
			self.draw_annotations()

		def move_annotation(self, annotation_index: int, x: int, y: int, start_x: int, start_y: int, original: dict[str, float | str]) -> None:
			annotation = self.annotations[annotation_index]
			dx = (x - start_x) / max(self.current_scale, 1e-6)
			dy = (y - start_y) / max(self.current_scale, 1e-6)

			min_x = 0.0
			max_x = float(self.source_width - 1)
			min_y = 0.0
			max_y = float(self.source_height - 1)

			dx = max(min_x - original["x1"], min(dx, max_x - original["x2"]))
			dy = max(min_y - original["y1"], min(dy, max_y - original["y2"]))

			annotation["dot_x"] = original["dot_x"] + dx
			annotation["dot_y"] = original["dot_y"] + dy
			annotation["x1"] = original["x1"] + dx
			annotation["x2"] = original["x2"] + dx
			annotation["y1"] = original["y1"] + dy
			annotation["y2"] = original["y2"] + dy
			self.draw_annotations()

		def center_dot_in_annotation(self, annotation: dict[str, float | str]) -> None:
			annotation["dot_x"] = (float(annotation["x1"]) + float(annotation["x2"])) / 2.0
			annotation["dot_y"] = (float(annotation["y1"]) + float(annotation["y2"])) / 2.0

		def resize_annotation_side(self, annotation_index: int, side: str, x: int, y: int) -> None:
			x, y = self.canvas_to_image_coords(x, y)
			annotation = self.annotations[annotation_index]
			x1, y1, x2, y2 = annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]
			min_side = 8 / max(self.current_scale, 1e-6)

			if side in {"left", "upper_left", "lower_left"}:
				x1 = min(x, x2 - min_side)
			elif side in {"right", "upper_right", "lower_right"}:
				x2 = max(x, x1 + min_side)
			elif side == "top":
				y1 = min(y, y2 - min_side)
			elif side == "bottom":
				y2 = max(y, y1 + min_side)

			annotation["x1"] = x1
			annotation["y1"] = y1
			annotation["x2"] = x2
			annotation["y2"] = y2
			self.center_dot_in_annotation(annotation)
			self.draw_annotations()

		def resize_annotation_corner(self, annotation_index: int, corner: str, x: int, y: int) -> None:
			x, y = self.canvas_to_image_coords(x, y)
			annotation = self.annotations[annotation_index]
			x1, y1, x2, y2 = annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]
			min_side = 8 / max(self.current_scale, 1e-6)

			if corner == "top_left":
				x1 = min(x, x2 - min_side)
				y1 = min(y, y2 - min_side)
			elif corner == "top_right":
				x2 = max(x, x1 + min_side)
				y1 = min(y, y2 - min_side)
			elif corner == "bottom_left":
				x1 = min(x, x2 - min_side)
				y2 = max(y, y1 + min_side)
			elif corner == "bottom_right":
				x2 = max(x, x1 + min_side)
				y2 = max(y, y1 + min_side)
			elif corner == "right":
				x2 = max(x, x1 + min_side)
			elif corner == "left":
				x1 = min(x, x2 - min_side)
			else:
				return

			annotation["x1"] = x1
			annotation["y1"] = y1
			annotation["x2"] = x2
			annotation["y2"] = y2
			self.center_dot_in_annotation(annotation)
			self.draw_annotations()

		def update_time_label(self, frame_index: int) -> None:
			current_seconds = frame_index / self.fps
			total_seconds = (self.total_frames - 1) / self.fps
			self.time_label.setText(
				f"{format_time(current_seconds)} / {format_time(total_seconds)}"
			)

		def show_frame_at(self, frame_index: int) -> None:
			frame_index = max(0, min(frame_index, self.total_frames - 1))
			self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
			success, frame = self.capture.read()
			if not success:
				raise RuntimeError("Unable to read frame from video.")
			self.current_frame_bgr = frame.copy()

			self.source_height, self.source_width = frame.shape[:2]
			base_scale = min(960 / self.source_width, 540 / self.source_height, 1.0)
			requested_scale = base_scale * self.zoom_factor
			max_scale_w = MAX_RENDER_WIDTH / max(1, self.source_width)
			max_scale_h = MAX_RENDER_HEIGHT / max(1, self.source_height)
			max_scale_pixels = math.sqrt(MAX_RENDER_PIXELS / max(1, self.source_width * self.source_height))
			effective_scale = min(requested_scale, max_scale_w, max_scale_h, max_scale_pixels)
			self.current_scale = max(0.01, effective_scale)

			self.current_frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1

			self.load_current_frame_positions()

			self.photo = self.frame_to_photo(frame, self.current_scale)
			self.draw_current_frame()
			self.timeline.setValue(self.current_frame_index)
			self.update_time_label(self.current_frame_index)

		def on_mouse_wheel(self, event) -> None:
			delta = event.angleDelta().y()
			if delta > 0:
				new_zoom = min(ZOOM_MAX, self.zoom_factor * ZOOM_STEP)
			else:
				new_zoom = max(ZOOM_MIN, self.zoom_factor / ZOOM_STEP)

			if abs(new_zoom - self.zoom_factor) < 1e-9:
				return

			self.zoom_factor = new_zoom
			self.show_frame_at(self.current_frame_index)

		def on_key_press(self, event) -> None:
			if self.photo is None:
				return

			key = event.key()
			if key == Qt.Key_Space:
				self.toggle_pause()
				return
			if key == Qt.Key_T:
				self.toggle_tracking()
				return

			if self.active_action and self.active_action.get("type") == "dot":
				index = self.active_action["index"]
				if 0 <= index < len(self.annotations):
					annotation = self.annotations[index]
					current_name = str(annotation.get("name", ""))
					if key == Qt.Key_Backspace:
						annotation["name"] = current_name[:-1]
						self.active_action["moved"] = True
						self.draw_annotations()
						return
					if key in {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab, Qt.Key_Escape}:
						return
					char = event.text()
					if char and char.isprintable() and not (event.modifiers() & Qt.ControlModifier):
						annotation["name"] = current_name + char
						self.active_action["moved"] = True
						self.draw_annotations()
						return

			if key == Qt.Key_Q:
				current_index = int(self.speed_scale.value())
				new_index = max(0, current_index - 1)
				if new_index != current_index:
					self.speed_scale.setValue(new_index)
					self.on_speed_change(str(new_index))
				return
			if key == Qt.Key_E:
				current_index = int(self.speed_scale.value())
				new_index = min(len(self.speed_options) - 1, current_index + 1)
				if new_index != current_index:
					self.speed_scale.setValue(new_index)
					self.on_speed_change(str(new_index))
				return

			step = PAN_STEP_PIXELS

			if key in {Qt.Key_W, Qt.Key_Up}:
				self.pan_y += step
			elif key in {Qt.Key_S, Qt.Key_Down}:
				self.pan_y -= step
			elif key in {Qt.Key_A, Qt.Key_Left}:
				self.pan_x += step
			elif key in {Qt.Key_D, Qt.Key_Right}:
				self.pan_x -= step
			else:
				return

			self.draw_current_frame()

		def on_copy_hotkey(self) -> None:
			if self.active_action and self.active_action.get("type") == "dot":
				index = self.active_action["index"]
				if 0 <= index < len(self.annotations):
					annotation = self.annotations[index]
					self.copy_buffer = {
						"dot_x": annotation["dot_x"],
						"dot_y": annotation["dot_y"],
						"x1": annotation["x1"],
						"y1": annotation["y1"],
						"x2": annotation["x2"],
						"y2": annotation["y2"],
						"name": annotation.get("name", "dot"),
					}

		def on_paste_hotkey(self) -> None:
			self.duplicate_active_annotation()

		def on_right_click(self, event) -> None:
			x, y = self.event_xy(event)
			if not self.is_inside_image(x, y):
				self.active_action = None
				return

			dot_hit = self.find_dot_index_at(x, y)
			if dot_hit is not None:
				annotation = self.annotations[dot_hit]
				self.active_action = {
					"type": "dot",
					"index": dot_hit,
					"start_x": x,
					"start_y": y,
					"moved": False,
					"original": {
						"dot_x": annotation["dot_x"],
						"dot_y": annotation["dot_y"],
						"x1": annotation["x1"],
						"y1": annotation["y1"],
						"x2": annotation["x2"],
						"y2": annotation["y2"],
					},
				}
				return

			corner_hit = self.find_corner_at(x, y)
			if corner_hit is not None:
				annotation_index, corner = corner_hit
				self.active_action = {"type": "corner", "index": annotation_index, "corner": corner}
				return

			side_hit = self.find_side_at(x, y)
			if side_hit is not None:
				annotation_index, side = side_hit
				self.active_action = {"type": "side", "index": annotation_index, "side": side}
				return

			self.add_annotation(x, y)
			self.active_action = None

		def on_right_drag(self, event) -> None:
			if self.active_action is None:
				return

			x, y = self.event_xy(event)
			if self.active_action["type"] == "dot":
				index = self.active_action["index"]
				if index < 0 or index >= len(self.annotations):
					return
				self.move_annotation(
					index,
					x,
					y,
					self.active_action["start_x"],
					self.active_action["start_y"],
					self.active_action["original"],
				)
				dx = abs(x - self.active_action["start_x"])
				dy = abs(y - self.active_action["start_y"])
				if dx >= DRAG_THRESHOLD or dy >= DRAG_THRESHOLD:
					self.active_action["moved"] = True
			elif self.active_action["type"] == "side":
				index = self.active_action["index"]
				side = self.active_action["side"]
				if index < 0 or index >= len(self.annotations):
					return
				self.resize_annotation_side(index, side, x, y)
			elif self.active_action["type"] == "corner":
				index = self.active_action["index"]
				corner = self.active_action["corner"]
				if index < 0 or index >= len(self.annotations):
					return
				self.resize_annotation_corner(index, corner, x, y)

		def on_right_release(self, _event) -> None:
			if self.active_action and self.active_action.get("type") == "dot":
				index = self.active_action["index"]
				if not self.active_action.get("moved", False):
					if 0 <= index < len(self.annotations):
						dot_name = str(self.annotations[index].get("name", "")).strip()
						if dot_name:
							self.deleted_dot_names.add(dot_name)
						del self.annotations[index]
						self.draw_annotations()
				elif 0 <= index < len(self.annotations):
					annotation = self.annotations[index]
					if not str(annotation.get("name", "")).strip():
						self.ensure_annotation_has_name(annotation)
					else:
						annotation["name"] = self.make_unique_name(str(annotation.get("name", "")), ignore_index=index)
					self.ensure_dot_columns(str(annotation.get("name", "")).strip())
					self.update_memory_for_current_frame()
					self.draw_annotations()
			elif self.active_action and self.active_action.get("type") in {"side", "corner"}:
				self.update_memory_for_current_frame()
				self.draw_annotations()
			self.active_action = None

		def on_remove_click(self, event) -> None:
			x, y = self.event_xy(event)
			hit_index = self.find_dot_index_at(x, y)
			if hit_index is None:
				return
			dot_name = str(self.annotations[hit_index].get("name", "")).strip()
			if dot_name:
				self.deleted_dot_names.add(dot_name)
			del self.annotations[hit_index]
			self.draw_annotations()

		def on_slider_press(self) -> None:
			self.is_dragging_slider = True

		def on_slider_release(self) -> None:
			self.is_dragging_slider = False
			self.pending_seek = int(self.timeline.value())
			if self.paused:
				self.show_frame_at(self.pending_seek)
				self.pending_seek = None
			self.playback_frame_accumulator = 0.0
			self.last_update_time = time.perf_counter()

		def on_seek(self, value: str) -> None:
			self.pending_seek = int(float(value))
			self.playback_frame_accumulator = 0.0

		def toggle_tracking(self) -> None:
			self.tracking_enabled = not self.tracking_enabled
			self.track_button.setText("Track: On" if self.tracking_enabled else "Track: Off")

		def on_algorithm_changed(self, algorithm_name: str) -> None:
			if algorithm_name:
				self.selected_tracking_algorithm = algorithm_name

		def get_annotation_snapshot_by_name(self) -> dict[str, dict[str, float | str]]:
			snapshot: dict[str, dict[str, float | str]] = {}
			for annotation in self.annotations:
				name = str(annotation.get("name", "")).strip()
				if not name:
					continue
				snapshot[name] = {
					"name": name,
					"dot_x": float(annotation["dot_x"]),
					"dot_y": float(annotation["dot_y"]),
					"x1": float(annotation["x1"]),
					"y1": float(annotation["y1"]),
					"x2": float(annotation["x2"]),
					"y2": float(annotation["y2"]),
				}
			return snapshot

		def create_tracker(self, algorithm_name: str):
			factory_name = f"Tracker{algorithm_name}_create"
			factory = getattr(cv2, factory_name, None)
			if callable(factory):
				return factory()
			legacy = getattr(cv2, "legacy", None)
			if legacy is not None:
				legacy_factory = getattr(legacy, factory_name, None)
				if callable(legacy_factory):
					return legacy_factory()
			return None

		def clamp_box_to_frame(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
			max_x = float(max(1, self.source_width - 1))
			max_y = float(max(1, self.source_height - 1))
			min_size = 4.0

			width = max(min_size, float(x2 - x1))
			height = max(min_size, float(y2 - y1))
			center_x = (float(x1) + float(x2)) / 2.0
			center_y = (float(y1) + float(y2)) / 2.0

			x1_new = center_x - (width / 2.0)
			x2_new = center_x + (width / 2.0)
			y1_new = center_y - (height / 2.0)
			y2_new = center_y + (height / 2.0)

			if x1_new < 0.0:
				x2_new -= x1_new
				x1_new = 0.0
			if y1_new < 0.0:
				y2_new -= y1_new
				y1_new = 0.0
			if x2_new > max_x:
				delta = x2_new - max_x
				x1_new -= delta
				x2_new = max_x
			if y2_new > max_y:
				delta = y2_new - max_y
				y1_new -= delta
				y2_new = max_y

			x1_new = max(0.0, min(x1_new, max_x))
			y1_new = max(0.0, min(y1_new, max_y))
			x2_new = max(x1_new + 1.0, min(x2_new, max_x))
			y2_new = max(y1_new + 1.0, min(y2_new, max_y))
			return x1_new, y1_new, x2_new, y2_new

		def track_annotation(
			self,
			algorithm_name: str,
			previous_annotation: dict[str, float | str],
			previous_frame,
			current_frame,
		) -> dict[str, float] | None:
			prev_dot_x = float(previous_annotation["dot_x"])
			prev_dot_y = float(previous_annotation["dot_y"])
			prev_x1 = float(previous_annotation["x1"])
			prev_y1 = float(previous_annotation["y1"])
			prev_x2 = float(previous_annotation["x2"])
			prev_y2 = float(previous_annotation["y2"])

			prev_x1, prev_y1, prev_x2, prev_y2 = self.clamp_box_to_frame(prev_x1, prev_y1, prev_x2, prev_y2)
			left_offset = prev_dot_x - prev_x1
			right_offset = prev_x2 - prev_dot_x
			top_offset = prev_dot_y - prev_y1
			bottom_offset = prev_y2 - prev_dot_y

			frame_height, frame_width = current_frame.shape[:2]
			ix1 = max(0, min(int(math.floor(prev_x1)), frame_width - 2))
			iy1 = max(0, min(int(math.floor(prev_y1)), frame_height - 2))
			ix2 = max(ix1 + 2, min(int(math.ceil(prev_x2)), frame_width))
			iy2 = max(iy1 + 2, min(int(math.ceil(prev_y2)), frame_height))

			if ix2 <= ix1 or iy2 <= iy1:
				return None

			if algorithm_name == "KLT":
				previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
				current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
				point_prev = np.array([[[prev_dot_x, prev_dot_y]]], dtype=np.float32)
				point_next, status, _error = cv2.calcOpticalFlowPyrLK(
					previous_gray,
					current_gray,
					point_prev,
					None,
					winSize=(21, 21),
					maxLevel=3,
					criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
				)
				if point_next is None or status is None or int(status[0][0]) != 1:
					return None

				new_dot_x = float(point_next[0][0][0])
				new_dot_y = float(point_next[0][0][1])
				new_x1 = new_dot_x - left_offset
				new_x2 = new_dot_x + right_offset
				new_y1 = new_dot_y - top_offset
				new_y2 = new_dot_y + bottom_offset

			elif algorithm_name == "Template Matching":
				previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
				current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
				template = previous_gray[iy1:iy2, ix1:ix2]
				if template.size == 0:
					return None

				template_h, template_w = template.shape[:2]
				search_margin = int(max(template_w, template_h) * 2)
				sx1 = max(0, ix1 - search_margin)
				sy1 = max(0, iy1 - search_margin)
				sx2 = min(frame_width, ix2 + search_margin)
				sy2 = min(frame_height, iy2 + search_margin)
				search_img = current_gray[sy1:sy2, sx1:sx2]
				if search_img.shape[0] < template_h or search_img.shape[1] < template_w:
					return None

				match_result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
				_min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(match_result)
				best_x = sx1 + max_loc[0]
				best_y = sy1 + max_loc[1]
				new_x1 = float(best_x)
				new_y1 = float(best_y)
				new_x2 = float(best_x + template_w)
				new_y2 = float(best_y + template_h)
				new_dot_x = new_x1 + left_offset
				new_dot_y = new_y1 + top_offset

			elif algorithm_name in {"CSRT", "KCF"}:
				tracker = self.create_tracker(algorithm_name)
				if tracker is None:
					return None
				bbox = (float(ix1), float(iy1), float(ix2 - ix1), float(iy2 - iy1))
				initialized = tracker.init(previous_frame, bbox)
				if initialized is False:
					return None
				ok, tracked_bbox = tracker.update(current_frame)
				if not ok:
					return None

				tx, ty, tw, th = tracked_bbox
				new_x1 = float(tx)
				new_y1 = float(ty)
				new_x2 = float(tx + tw)
				new_y2 = float(ty + th)
				new_dot_x = new_x1 + left_offset
				new_dot_y = new_y1 + top_offset
			else:
				return None

			new_x1, new_y1, new_x2, new_y2 = self.clamp_box_to_frame(new_x1, new_y1, new_x2, new_y2)
			new_dot_x = max(new_x1, min(new_dot_x, new_x2))
			new_dot_y = max(new_y1, min(new_dot_y, new_y2))
			new_dot_x, new_dot_y = self.clamp_to_image_coords(new_dot_x, new_dot_y)

			return {
				"dot_x": float(new_dot_x),
				"dot_y": float(new_dot_y),
				"x1": float(new_x1),
				"y1": float(new_y1),
				"x2": float(new_x2),
				"y2": float(new_y2),
			}

		def run_tracking_for_frame(
			self,
			frame_index: int,
			previous_frame,
			previous_annotations_by_name: dict[str, dict[str, float | str]],
		) -> None:
			if not self.tracking_enabled:
				return
			if previous_frame is None or self.current_frame_bgr is None:
				return
			if not previous_annotations_by_name:
				return

			current_annotations_by_name = {
				str(annotation.get("name", "")).strip(): annotation
				for annotation in self.annotations
				if str(annotation.get("name", "")).strip()
			}

			tracked_any = False
			for annotation_name, previous_annotation in previous_annotations_by_name.items():
				current_annotation = current_annotations_by_name.get(annotation_name)
				if current_annotation is None:
					continue

				tracked = self.track_annotation(
					self.selected_tracking_algorithm,
					previous_annotation,
					previous_frame,
					self.current_frame_bgr,
				)
				if tracked is None:
					continue

				current_annotation["dot_x"] = tracked["dot_x"]
				current_annotation["dot_y"] = tracked["dot_y"]
				current_annotation["x1"] = tracked["x1"]
				current_annotation["y1"] = tracked["y1"]
				current_annotation["x2"] = tracked["x2"]
				current_annotation["y2"] = tracked["y2"]
				tracked_any = True

			if tracked_any:
				self.update_memory_for_current_frame()
				self.draw_annotations()

		def toggle_pause(self) -> None:
			self.paused = not self.paused
			self.playback_frame_accumulator = 0.0
			self.last_update_time = time.perf_counter()
			self.pause_button.setText("Resume" if self.paused else "Pause")

		def on_speed_change(self, value: str) -> None:
			index = int(round(float(value)))
			index = max(0, min(len(self.speed_options) - 1, index))
			self.playback_speed = self.speed_options[index]
			speed_text = f"{self.playback_speed:g}x"
			self.speed_label.setText(speed_text)

		def update_loop(self) -> None:
			try:
				now = time.perf_counter()
				elapsed_seconds = max(0.0, now - self.last_update_time)
				self.last_update_time = now

				if self.pending_seek is not None and not self.is_dragging_slider:
					self.show_frame_at(self.pending_seek)
					self.pending_seek = None
					self.playback_frame_accumulator = 0.0
					self.last_update_time = time.perf_counter()

				if not self.paused and not self.is_dragging_slider:
					self.playback_frame_accumulator += elapsed_seconds * self.fps * self.playback_speed
					frames_to_advance = int(self.playback_frame_accumulator)
					if frames_to_advance > 0:
						previous_frame = self.current_frame_bgr.copy() if self.current_frame_bgr is not None else None
						previous_annotations_by_name = self.get_annotation_snapshot_by_name()
						self.playback_frame_accumulator -= frames_to_advance
						next_index = self.current_frame_index + frames_to_advance
						if next_index >= self.total_frames:
							self.show_frame_at(self.total_frames - 1)
							self.run_tracking_for_frame(self.current_frame_index, previous_frame, previous_annotations_by_name)
							self.paused = True
							self.pause_button.setText("Resume")
							self.playback_frame_accumulator = 0.0
						else:
							self.show_frame_at(next_index)
							self.run_tracking_for_frame(self.current_frame_index, previous_frame, previous_annotations_by_name)
			except Exception as exc:
				QMessageBox.critical(self.root, "Playback Error", str(exc))
				self.on_close()

		def mark_csv_dirty(self) -> None:
			pass

		def on_close(self) -> None:
			try:
				if hasattr(self, "timer") and self.timer.isActive():
					self.timer.stop()
			except Exception:
				pass
			try:
				self.capture.release()
			except Exception:
				pass
			try:
				self.root.close()
			except Exception:
				pass

		def on_close_event(self, event) -> None:
			try:
				if hasattr(self, "timer") and self.timer.isActive():
					self.timer.stop()
			except Exception:
				pass
			try:
				self.capture.release()
			except Exception:
				pass
			event.accept()

	player = VideoPlayerUI()
	player.root.show()
	app.exec()


def main() -> int:
	try:
		args = parse_args()
		video_path = get_video_path(args.video_path)
		csv_path, csv_frames = create_video_timeline_csv(video_path)
		print(f"Total frames: {csv_frames}")
		try:
			run_video_player_ui(video_path, csv_path)
		except UIUnavailableError as ui_error:
			print(f"UI unavailable, falling back to console mode: {ui_error}")
			process_video_console(video_path)
		return 0
	except KeyboardInterrupt:
		print("\nOperation cancelled by user.")
		return 130
	except (FileNotFoundError, ValueError) as exc:
		print(f"Input error: {exc}")
		return 2
	except RuntimeError as exc:
		print(f"Runtime error: {exc}")
		return 3
	except Exception as exc:
		import traceback
		traceback.print_exc()
		print(f"Unexpected error: {exc}")
		return 1


if __name__ == "__main__":
	sys.exit(main())
