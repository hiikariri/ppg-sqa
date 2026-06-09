import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample, find_peaks
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

try:
    from PyQt5 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - fallback for alternate Qt bindings
    from PySide6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


APP_TITLE = "PPG Preprocessing Viewer"
DEFAULT_ORIGINAL_FS = 200
DEFAULT_TARGET_FS = 125


def load_metadata_and_signal(file_path: str):
    metadata = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            match = re.match(r"#\s*([^:]+):\s*(.*)", stripped)
            if match:
                metadata[match.group(1).strip().lower()] = match.group(2).strip()

    dataframe = pd.read_csv(file_path, comment="#")
    if dataframe.empty:
        raise ValueError("No signal rows found in the selected file.")

    columns = {column.strip().lower(): column for column in dataframe.columns}
    if "ir" in columns:
        signal = pd.to_numeric(dataframe[columns["ir"]], errors="coerce").dropna().to_numpy()
    elif dataframe.shape[1] >= 2:
        signal = pd.to_numeric(dataframe.iloc[:, 1], errors="coerce").dropna().to_numpy()
    else:
        raise ValueError("The CSV file must contain an IR column or at least two data columns.")

    if signal.size == 0:
        raise ValueError("No valid IR samples were found in the file.")

    return metadata, signal


def preprocess_signal(signal: np.ndarray, original_fs: int = DEFAULT_ORIGINAL_FS, target_fs: int = DEFAULT_TARGET_FS):
    if signal.size < 2:
        raise ValueError("Signal is too short to preprocess.")

    inverted = -signal.astype(float)
    target_length = max(1, int(round(len(inverted) * target_fs / original_fs)))
    resampled = resample(inverted, target_length) if target_length != len(inverted) else inverted.copy()

    nyquist = target_fs / 2.0
    low = 0.5 / nyquist
    high = 8.0 / nyquist
    filtered = resampled.copy()

    if 0 < low < high < 1 and filtered.size > 15:
        b, a = butter(4, [low, high], btype="band")
        try:
            filtered = filtfilt(b, a, filtered)
        except ValueError:
            pass

    mean = float(np.mean(filtered))
    std = float(np.std(filtered))
    normalized = (filtered - mean) / (std + 1e-8)

    return {
        "normalized": normalized,
        "raw_time": np.arange(len(signal)) / original_fs,
        "processed_time": np.arange(len(normalized)) / target_fs,
    }


def segment_signal(normalized: np.ndarray, target_fs: int = DEFAULT_TARGET_FS, min_hr_bpm: int = 35, max_hr_bpm: int = 220):
    """Simple segmentation by detecting peaks in the normalized PPG signal.

    Returns a dict with peak indices, peak times and list of (start_idx, end_idx) segments.
    """
    result = {"peaks": np.array([], dtype=int), "peak_times": np.array([], dtype=float), "segments": []}
    if normalized is None or normalized.size < 3:
        return result

    fs = int(target_fs)
    # Minimum distance in samples between peaks based on maximum heart rate
    min_distance = max(1, int(round(fs * 60.0 / max_hr_bpm)))

    try:
        peaks, props = find_peaks(normalized, distance=min_distance, prominence=0.3)
    except Exception:
        peaks = np.array([], dtype=int)

    if peaks.size > 0:
        peak_times = peaks / float(fs)
        segments = []
        if peaks.size > 1:
            for i in range(peaks.size - 1):
                segments.append((int(peaks[i]), int(peaks[i + 1])))
        else:
            # single peak -> one tiny segment around it
            p = int(peaks[0])
            segments.append((max(0, p - int(0.5 * fs)), min(len(normalized) - 1, p + int(0.5 * fs))))

        result = {"peaks": peaks, "peak_times": peak_times, "segments": segments}

    return result


def build_template_from_peaks(normalized: np.ndarray, peaks: np.ndarray, fs: int = DEFAULT_TARGET_FS, template_len: int = 128, pre_sec: float = 0.3, post_sec: float = 0.5):
    if peaks is None or peaks.size == 0:
        return None

    pre = int(round(pre_sec * fs))
    post = int(round(post_sec * fs))
    segments = []
    for p in peaks:
        s = int(p) - pre
        e = int(p) + post
        if s >= 0 and e <= len(normalized):
            seg = normalized[s:e]
            try:
                seg_rs = resample(seg, template_len)
                segments.append(seg_rs)
            except Exception:
                continue

    if len(segments) < 3:
        return None

    stack = np.vstack(segments)
    template = np.median(stack, axis=0)
    # z-score normalize template
    template = (template - np.mean(template)) / (np.std(template) + 1e-8)
    return template


def check_correlation(segment: np.ndarray, fs: int, template: np.ndarray = None, min_corr: float = 0.6):
    """Signal quality assessment using beat correlation (from bgl_inference.py).

    Returns (is_good: bool, message: str)
    """
    if segment is None or segment.size < 3:
        return False, "too_short"

    try:
        # detect peaks with minimum distance ~0.3s
        peaks, _ = find_peaks(segment, distance=int(0.3 * fs))
        if len(peaks) < 3:
            return False, "Noisy (too few beats)"

        beats = []
        for i in range(len(peaks) - 1):
            beat = segment[peaks[i]:peaks[i + 1]]
            if len(beat) < 5:
                continue
            x = np.linspace(0, 1, len(beat))
            f = interp1d(x, beat, kind='linear')
            beat_norm = f(np.linspace(0, 1, 100))
            beats.append(beat_norm)

        if len(beats) < 2:
            return False, "Noisy (insufficient beats)"

        beats = np.array(beats)
        template_local = np.mean(beats, axis=0)
        corrs = [pearsonr(b, template_local)[0] for b in beats]
        mean_corr = float(np.mean(corrs))

        return (mean_corr >= min_corr, f"Corr={mean_corr:.2f}")
    except Exception:
        return False, "quality_err"


def list_data_files(folder: str):
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted(path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() == ".csv")


def format_metadata(metadata: dict, file_path: str, raw_signal: np.ndarray):
    lines = [f"<b>File:</b> {Path(file_path).name}", f"<b>Samples:</b> {len(raw_signal):,}"]
    for key in ("name", "iteration", "bgl", "sbp", "dbp", "timestamp", "total samples"):
        if key in metadata:
            lines.append(f"<b>{key.upper()}:</b> {metadata[key]}")
    return "<br>".join(lines)


class PreprocessingViewer(QtWidgets.QMainWindow):
    def __init__(self, default_folder: str):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 920)
        self.default_folder = default_folder
        self.current_folder = default_folder

        self._build_ui()
        self._apply_style()
        self.load_folder(default_folder)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        left_panel = QtWidgets.QFrame()
        left_panel.setObjectName("LeftPanel")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(10)

        folder_row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit(self.default_folder)
        self.folder_button = QtWidgets.QPushButton("Browse Folder")
        self.folder_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.folder_button)

        self.file_list = QtWidgets.QListWidget()
        self.file_list.currentTextChanged.connect(self.on_file_selected)

        self.refresh_button = QtWidgets.QPushButton("Refresh Files")
        self.refresh_button.clicked.connect(self.refresh_files)

        # Segmentation controls
        params_row = QtWidgets.QHBoxLayout()
        params_row.setSpacing(6)
        params_row.addWidget(QtWidgets.QLabel("Seg length (s):"))
        self.segment_len_spin = QtWidgets.QSpinBox()
        self.segment_len_spin.setRange(1, 60)
        self.segment_len_spin.setValue(10)
        params_row.addWidget(self.segment_len_spin)

        params_row.addWidget(QtWidgets.QLabel("Min corr:"))
        self.min_corr_spin = QtWidgets.QDoubleSpinBox()
        self.min_corr_spin.setRange(0.0, 1.0)
        self.min_corr_spin.setSingleStep(0.05)
        self.min_corr_spin.setValue(0.8)
        params_row.addWidget(self.min_corr_spin)

        self.show_accepted_chk = QtWidgets.QCheckBox("Show accepted")
        self.show_accepted_chk.setChecked(True)
        params_row.addWidget(self.show_accepted_chk)

        self.show_rejected_chk = QtWidgets.QCheckBox("Show rejected")
        self.show_rejected_chk.setChecked(True)
        params_row.addWidget(self.show_rejected_chk)

        # Connect to reprocess when parameters change
        self.segment_len_spin.valueChanged.connect(self.on_params_changed)
        self.min_corr_spin.valueChanged.connect(self.on_params_changed)
        self.show_accepted_chk.stateChanged.connect(self.on_params_changed)
        self.show_rejected_chk.stateChanged.connect(self.on_params_changed)

        # Segment navigation
        nav_row = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.next_btn = QtWidgets.QPushButton("Next")
        self.seg_index_label = QtWidgets.QLabel("Segment: - / -")
        self.sqa_label = QtWidgets.QLabel("")
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        nav_row.addWidget(self.seg_index_label)
        nav_row.addWidget(self.sqa_label)
        left_layout.addLayout(nav_row)

        self.prev_btn.clicked.connect(self.on_prev_segment)
        self.next_btn.clicked.connect(self.on_next_segment)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.file_info = QtWidgets.QLabel("Select a CSV file from data_raw.")
        self.file_info.setWordWrap(True)

        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setWordWrap(True)

        left_layout.addLayout(folder_row)
        left_layout.addLayout(params_row)
        left_layout.addWidget(self.refresh_button)
        left_layout.addWidget(QtWidgets.QLabel("Data files"))
        left_layout.addWidget(self.file_list, 1)
        left_layout.addWidget(QtWidgets.QLabel("File details"))
        left_layout.addWidget(self.file_info)
        left_layout.addWidget(self.stats_label)

        right_panel = QtWidgets.QFrame()
        right_panel.setObjectName("RightPanel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        header = QtWidgets.QLabel("Raw signal and preprocessing result")
        header.setObjectName("HeaderLabel")

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.raw_ax = self.figure.add_subplot(311)
        self.processed_ax = self.figure.add_subplot(312)
        self.segments_ax = self.figure.add_subplot(313)

        right_layout.addWidget(header)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)

        root_layout.addWidget(left_panel, 0)
        root_layout.addWidget(right_panel, 1)

        self.statusBar().showMessage("Ready")

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4f5f7;
            }
            QFrame#LeftPanel, QFrame#RightPanel {
                background: #ffffff;
                border: 1px solid #dfe3ea;
                border-radius: 14px;
            }
            QLabel {
                color: #1f2937;
                font-size: 13px;
            }
            QLabel#HeaderLabel {
                font-size: 20px;
                font-weight: 700;
                color: #0f172a;
            }
            QLineEdit, QListWidget {
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px;
                background: #ffffff;
            }
            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 9px 14px;
                background: #2563eb;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #1d4ed8;
            }
            QPushButton:disabled {
                background: #94a3b8;
            }
            """
        )

    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data_raw folder", self.current_folder)
        if folder:
            self.folder_edit.setText(folder)
            self.load_folder(folder)

    def refresh_files(self):
        self.load_folder(self.folder_edit.text().strip())

    def load_folder(self, folder: str):
        folder = folder.strip()
        if not folder:
            return

        self.current_folder = folder
        files = list_data_files(folder)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for file_path in files:
            self.file_list.addItem(file_path.name)
        self.file_list.blockSignals(False)

        if not files:
            self.file_info.setText("No CSV files found in the selected folder.")
            self.stats_label.setText("")
            self.clear_plots()
            self.statusBar().showMessage("No files found")
            return

        self.statusBar().showMessage(f"Loaded {len(files)} CSV files from {folder}")
        self.file_list.setCurrentRow(0)

    def on_file_selected(self, file_name: str):
        if not file_name:
            return

        file_path = Path(self.current_folder) / file_name
        self.process_file(str(file_path))

    def on_params_changed(self):
        # Re-run processing for the currently selected file with new parameters
        item = self.file_list.currentItem()
        if item is None:
            return
        file_path = Path(self.current_folder) / item.text()
        if file_path.exists():
            self.process_file(str(file_path))

    def on_prev_segment(self):
        if not hasattr(self, 'current_segment_index'):
            return
        if self.current_segment_index > 0:
            self.current_segment_index -= 1
            self.update_segment_navigation()

    def on_next_segment(self):
        if not hasattr(self, 'current_segment_index'):
            return
        if hasattr(self, 'current_segments') and self.current_segment_index < len(self.current_segments) - 1:
            self.current_segment_index += 1
            self.update_segment_navigation()

    def update_segment_navigation(self):
        # update labels and redraw plots for the current segment
        total = len(self.current_segments) if hasattr(self, 'current_segments') else 0
        idx = getattr(self, 'current_segment_index', 0)
        self.seg_index_label.setText(f"Segment: {idx+1} / {total}")
        qm = None
        if hasattr(self, 'current_quality_map'):
            qm = self.current_quality_map.get(idx)
        accepted = 'REJECTED'
        if hasattr(self, 'current_valid_flags') and idx < len(self.current_valid_flags) and self.current_valid_flags[idx]:
            accepted = 'ACCEPTED'
        sqa_text = f"{accepted} {qm or ''}".strip()
        self.sqa_label.setText(sqa_text)
        # redraw plots with current segment highlighted
        current = getattr(self, 'current_segment_index', None)
        seg_idxs = list(range(len(self.current_segments))) if hasattr(self, 'current_segments') else []
        self._draw_plots(self.current_file_path, self.current_raw_signal, self.current_result, self.current_segmentation, valid_idxs=self.current_valid_idxs, rejected_idxs=self.current_rejected_idxs, segment_len=self.current_segment_len, quality_map=self.current_quality_map, current_segment_index=current)

    def clear_plots(self):
        self.raw_ax.clear()
        self.processed_ax.clear()
        self.raw_ax.set_title("Raw signal")
        self.processed_ax.set_title("Preprocessed signal")
        try:
            self.segments_ax.clear()
            self.segments_ax.set_title("Segments (SQA: green=accepted, gray=rejected)")
        except Exception:
            pass
        self.canvas.draw_idle()

    def process_file(self, file_path: str):
        try:
            metadata, raw_signal = load_metadata_and_signal(file_path)
            result = preprocess_signal(raw_signal)
            segmentation = segment_signal(result["normalized"])

            # Build a pulse template from detected peaks (if available)
            fs = DEFAULT_TARGET_FS
            template = build_template_from_peaks(result["normalized"], segmentation.get("peaks", np.array([], dtype=int)), fs=fs)

            # Fixed-length segmentation for model input (seconds)
            segment_len_sec = int(self.segment_len_spin.value()) if hasattr(self, 'segment_len_spin') else 10
            segment_len = int(round(segment_len_sec * fs))

            # Use first 60s if available, otherwise use entire signal
            one_min_samples = fs * 60
            data_to_segment = result["normalized"][:one_min_samples] if len(result["normalized"]) >= one_min_samples else result["normalized"]
            n_segments = len(data_to_segment) // segment_len

            valid_segments = []
            segment_info = []
            valid_idxs = []
            rejected_idxs = []
            quality_map = {}

            normalize_segments = True
            verbose = False

            # prepare storage for navigation and current dataset
            self.current_file_path = file_path
            self.current_raw_signal = raw_signal
            self.current_result = result
            self.current_segmentation = segmentation
            self.current_segment_len = segment_len

            self.current_segments = []
            for i in range(n_segments):
                start_idx = i * segment_len
                end_idx = start_idx + segment_len
                segment = data_to_segment[start_idx:end_idx]

                min_corr = float(self.min_corr_spin.value()) if hasattr(self, 'min_corr_spin') else 0.6
                is_good, quality_msg = check_correlation(segment, fs, template=template, min_corr=min_corr)

                # store quality message for this segment
                quality_map[i] = quality_msg

                if is_good:
                    if normalize_segments:
                        mean = float(np.mean(segment))
                        std = float(np.std(segment))
                        segment_normalized = (segment - mean) / (std + 1e-6)
                    else:
                        segment_normalized = segment

                    segment_reshaped = segment_normalized.reshape(-1, 1)
                    valid_segments.append(segment_reshaped)
                    valid_idxs.append(i)
                    self.current_segments.append(segment_normalized)

                    segment_info.append({
                        "segment_idx": i,
                        "start_time": start_idx / float(fs),
                        "end_time": end_idx / float(fs),
                        "quality": quality_msg,
                        "mean": float(np.mean(segment)),
                        "std": float(np.std(segment)),
                    })
                    if verbose:
                        print(f"  Segment {i+1}: GOOD ({quality_msg})")
                else:
                    rejected_idxs.append(i)
                    self.current_segments.append(segment)
                    if verbose:
                        print(f"  Segment {i+1}: REJECTED ({quality_msg})")

            # finalize current state for navigation
            self.current_valid_idxs = valid_idxs
            self.current_rejected_idxs = rejected_idxs
            self.current_valid_flags = [i in valid_idxs for i in range(n_segments)]
            self.current_quality_map = quality_map
            self.current_segment_index = 0

            # enable navigation if there are segments
            self.prev_btn.setEnabled(n_segments > 0)
            self.next_btn.setEnabled(n_segments > 0)

            # Pass segment info and quality map to plotting (show first segment)
            self._draw_plots(file_path, raw_signal, result, segmentation, valid_idxs=valid_idxs, rejected_idxs=rejected_idxs, segment_len=segment_len, quality_map=quality_map, current_segment_index=0)
            self.update_segment_navigation()
            self.file_info.setText(format_metadata(metadata, file_path, raw_signal))

            num_segments = len(segmentation.get("segments", []))
            avg_seg_sec = "-"
            if num_segments > 0:
                lengths = [(end - start) / float(DEFAULT_TARGET_FS) for start, end in segmentation["segments"]]
                avg_seg_sec = f"{(sum(lengths) / len(lengths)):.2f} s"

            self.stats_label.setText(
                f"<b>Raw min/max:</b> {raw_signal.min():.1f} / {raw_signal.max():.1f}<br>"
                f"<b>Processed length:</b> {len(result['normalized']):,} samples<br>"
                f"<b>Detected peaks:</b> {len(segmentation.get('peaks', []))}<br>"
                f"<b>Fixed windows:</b> {n_segments} ({len(valid_idxs)} good / {len(rejected_idxs)} rejected)<br>"
                f"<b>Segments:</b> {num_segments} (avg {avg_seg_sec})"
            )
            self.statusBar().showMessage(f"Processed {Path(file_path).name}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Processing error", str(exc))
            self.statusBar().showMessage("Error while processing file")

    def _draw_plots(self, file_path: str, raw_signal: np.ndarray, result: dict, segmentation: dict, valid_idxs=None, rejected_idxs=None, segment_len: int = None, quality_map: dict = None, current_segment_index: int = None):
        self.raw_ax.clear()
        self.processed_ax.clear()

        self.raw_ax.plot(result["raw_time"], raw_signal, color="#2563eb", linewidth=0.9)
        self.raw_ax.set_title(f"Raw IR Signal")
        self.raw_ax.set_xlabel("Time (s)")
        self.raw_ax.set_ylabel("Amplitude")
        self.raw_ax.grid(True, alpha=0.25)

        self.processed_ax.plot(result["processed_time"], result["normalized"], color="#ea580c", linewidth=0.9)
        self.processed_ax.set_title("Pre-processed Signal")
        self.processed_ax.set_xlabel("Time (s)")
        self.processed_ax.set_ylabel("Normalized Amplitude")
        self.processed_ax.grid(True, alpha=0.25)

        # Draw detected peaks and shaded segments
        peaks = segmentation.get("peaks", [])
        peak_times = segmentation.get("peak_times", [])
        segments = segmentation.get("segments", [])
        fs = DEFAULT_TARGET_FS

        try:
            for start_idx, end_idx in segments:
                start_t = start_idx / float(fs)
                end_t = end_idx / float(fs)
                self.processed_ax.axvspan(start_t, end_t, color="#f97316", alpha=0.04)

            if peak_times is not None and peak_times.size > 0:
                self.processed_ax.vlines(peak_times, ymin=min(result["normalized"]) - 0.1, ymax=max(result["normalized"]) + 0.1, colors="#b91c1c", linestyles="dashed", linewidth=0.7)

            # Overlay fixed-length windows on processed_ax and plot segmented rows in segments_ax
            if segment_len and (valid_idxs or rejected_idxs):
                t_full = result["processed_time"]
                show_accepted = getattr(self, 'show_accepted_chk', None) is None or self.show_accepted_chk.isChecked()
                show_rejected = getattr(self, 'show_rejected_chk', None) is None or self.show_rejected_chk.isChecked()

                # On the processed signal, lightly overlay accepted/rejected windows
                for i in (valid_idxs or []):
                    s = i * segment_len
                    e = s + segment_len
                    if e <= len(result["normalized"]):
                        t_seg = t_full[s:e]
                        seg = result["normalized"][s:e]
                        if show_accepted:
                            self.processed_ax.plot(t_seg, seg, color="#059669", alpha=0.6, linewidth=1.0)

                for i in (rejected_idxs or []):
                    s = i * segment_len
                    e = s + segment_len
                    if e <= len(result["normalized"]):
                        t_seg = t_full[s:e]
                        seg = result["normalized"][s:e]
                        if show_rejected:
                            self.processed_ax.plot(t_seg, seg, color="#6b7280", alpha=0.35, linewidth=0.8)

                # Plot only the selected 10s segment in segments_ax and annotate SQA label
                try:
                    self.segments_ax.clear()
                    cur_idx = current_segment_index if current_segment_index is not None else getattr(self, 'current_segment_index', 0)
                    if cur_idx is None:
                        cur_idx = 0
                    seg = None
                    if hasattr(self, 'current_segments') and cur_idx < len(self.current_segments):
                        seg = self.current_segments[cur_idx]
                    else:
                        s = cur_idx * segment_len
                        e = s + segment_len
                        if e <= len(result["normalized"]):
                            seg = result["normalized"][s:e]

                    if seg is not None and len(seg) > 0:
                        t_seg = np.linspace(0, segment_len / float(DEFAULT_TARGET_FS), num=len(seg))
                        color = "#6b7280" if cur_idx in (rejected_idxs or []) else "#059669" if cur_idx in (valid_idxs or []) else "#9ca3af"
                        self.segments_ax.plot(t_seg, seg, color=color, linewidth=1.8)

                        qm = None
                        if quality_map is not None:
                            qm = quality_map.get(cur_idx, None)
                        label = "REJECTED" if cur_idx in (rejected_idxs or []) else "GOOD"
                        if qm:
                            label = f"{label} ({qm})"

                        ymax = np.max(seg)
                        xpos = 0.95 * t_seg[-1]
                        self.segments_ax.text(xpos, ymax + 0.02 * (abs(ymax) + 1e-8), label, color=color, fontsize=14, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                        self.segments_ax.set_xlabel("Segment time (s)")
                        self.segments_ax.set_title(f"Segment {cur_idx+1} / {len(self.current_segments) if hasattr(self, 'current_segments') else 0}")
                        self.segments_ax.set_ylim(np.min(seg) - 0.05, np.max(seg) + 0.1)
                    else:
                        self.segments_ax.set_title("No segment available")
                except Exception:
                    pass
        except Exception:
            pass

        self.figure.tight_layout()
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Load a CSV from data_raw, preprocess IR, and show both plots in a PyQt GUI.")
    parser.add_argument("--file", help="Optional CSV file to open immediately")
    parser.add_argument("--folder", help="Folder containing raw CSV files", default=None)
    args = parser.parse_args()

    default_folder = args.folder or str(Path(__file__).resolve().parent / "data_raw")

    app = QtWidgets.QApplication(sys.argv)
    viewer = PreprocessingViewer(default_folder)
    viewer.show()

    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            folder = str(file_path.parent)
            viewer.folder_edit.setText(folder)
            viewer.load_folder(folder)
            items = viewer.file_list.findItems(file_path.name, QtCore.Qt.MatchExactly)
            if items:
                viewer.file_list.setCurrentItem(items[0])
        else:
            QtWidgets.QMessageBox.warning(viewer, "File not found", f"The file does not exist:\n{args.file}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()