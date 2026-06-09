"""
PyQt6 GUI: PPG validity test based on the Elgendi (2016) signal-quality indices.

For a selected PPG record from dataset_primer_1 it:
  1. plots the band-pass filtered PPG with the detected systolic peaks,
  2. plots the power spectrum and computes the SNR (cardiac band vs HF noise),
  3. decides validity from beat-template correlation (primary) + pulse-rate
     plausibility, with detector agreement (MSQI) and SNR refining the
     "excellent" tier. The corr / MSQI / SNR thresholds are tunable live, and
     each record can be eye-flagged good/bad (saved to ppg_quality_labels.json).
     The eight Elgendi SQIs are shown as diagnostics but no longer decide it.

"Evaluate all" sweeps the folder, shows the validity distribution and the
beat-correlation-vs-SNR quality space, and lets you step through the records it
marked unfit ("Review unfit").

Run:
    python ppg_sqa_gui.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "PyQt6")   # bind matplotlib's Qt backend to PyQt6

import numpy as np
from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import ppg_datasets as ds
from ppg_sqa import PPGSQAEngine

DEFAULT_DATASET = ds.PRIMER_DIR if Path(ds.PRIMER_DIR).exists() else "."

# Colours per verdict tier
TIER_COLOR = {"excellent": "#1b7837", "acceptable": "#7fbf7b", "unfit": "#b2182b"}

# (label, result value-key, kind, threshold-key) for the decision-criteria bars.
# The threshold key is resolved from the result's active thresholds at draw time
# so the bars reflect any live tuning. kind "higher": pass when value >= thr;
# "range": pass when lo <= value <= hi (threshold-key is a (lo, hi) key pair).
DECISION_CRITERIA = [
    ("Beat corr", "beat_corr", "higher", "BEAT_CORR_THRESH"),
    ("MSQI (%)", "msqi", "higher", "MSQI_THRESH"),
    ("SNR (dB)", "snr_db", "higher", "SNR_FLOOR_DB"),
    ("RR regularity", "reg_frac", "higher", "REG_FRAC_MIN"),
    ("Pulse rate (bpm)", "pulse_bpm", "range", ("HR_MIN_BPM", "HR_MAX_BPM")),
]


def _criterion_status(value, kind, thresh):
    """Return ``(passed, bar_ratio, annotation)`` for one decision criterion.

    ``bar_ratio`` is normalised so the pass line sits at 1.0.
    """
    if not np.isfinite(value):
        return False, 0.0, "n/a"
    if kind == "range":
        lo, hi = thresh
        ok = lo <= value <= hi
        return ok, (1.5 if ok else 0.5), f"{value:.0f}"
    ok = value >= thresh                                  # "higher is better"
    ratio = value / thresh if thresh else (1.5 if value > 0 else 0.0)
    annot = f"{value:.2f}" if abs(value) < 10 else f"{value:.0f}"
    return ok, float(np.clip(ratio, 0.0, 2.0)), annot


# --------------------------------------------------------------------------- #
class PPGSQAGui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG Validity Test — Signal Quality Indices")
        self.resize(1320, 940)
        self.dataset_dir = Path(DEFAULT_DATASET)
        self.records = []          # [(base_path, name), ...]
        self._cur = None           # (rec, res, name)
        self.flags = {}            # {name: "good"|"bad"} eye-labels (ppg_quality_labels.json)
        self._flags_path = None
        self._unfit = []           # [(base, name), ...] from the last "Evaluate all"
        self._unfit_pos = -1

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        # Dataset folder
        controls.addWidget(QtWidgets.QLabel("Dataset folder"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.dataset_dir))
        controls.addWidget(self.dataset_edit, 0, 1, 1, 5)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_dataset)
        controls.addWidget(browse, 0, 6)

        # Record + navigation
        controls.addWidget(QtWidgets.QLabel("Record"), 1, 0)
        self.record_combo = QtWidgets.QComboBox()
        controls.addWidget(self.record_combo, 1, 1)
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(lambda: self.step_record(-1))
        controls.addWidget(self.prev_btn, 1, 2)
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.next_btn.clicked.connect(lambda: self.step_record(1))
        controls.addWidget(self.next_btn, 1, 3)
        self.record_combo.currentIndexChanged.connect(self._update_nav_buttons)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_records)
        controls.addWidget(refresh, 1, 4)

        # Channel + view window
        controls.addWidget(QtWidgets.QLabel("PPG channel"), 2, 0)
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(ds.PPG_CHANNELS.keys())
        controls.addWidget(self.channel_combo, 2, 1)
        controls.addWidget(QtWidgets.QLabel("View start (s)"), 2, 2)
        self.view_start = QtWidgets.QDoubleSpinBox()
        self.view_start.setRange(0.0, 1e6); self.view_start.setSingleStep(5.0)
        controls.addWidget(self.view_start, 2, 3)
        controls.addWidget(QtWidgets.QLabel("View width (s)"), 2, 4)
        self.view_width = QtWidgets.QDoubleSpinBox()
        self.view_width.setRange(2.0, 1e6); self.view_width.setSingleStep(5.0)
        self.view_width.setValue(15.0)
        controls.addWidget(self.view_width, 2, 5)
        self.view_start.valueChanged.connect(self._apply_view)
        self.view_width.valueChanged.connect(self._apply_view)

        # Action buttons
        self.assess_btn = QtWidgets.QPushButton("Assess && Plot")
        self.assess_btn.clicked.connect(self.assess_and_plot)
        controls.addWidget(self.assess_btn, 1, 5, 1, 1)
        self.batch_btn = QtWidgets.QPushButton("Evaluate all")
        self.batch_btn.clicked.connect(self.evaluate_all)
        controls.addWidget(self.batch_btn, 1, 6, 1, 1)

        # Tunable decision thresholds (re-assess the current record on change)
        controls.addWidget(QtWidgets.QLabel("Beat corr ≥"), 3, 0)
        self.corr_spin = QtWidgets.QDoubleSpinBox()
        self.corr_spin.setRange(0.0, 1.0); self.corr_spin.setSingleStep(0.05)
        self.corr_spin.setDecimals(2); self.corr_spin.setValue(PPGSQAEngine.BEAT_CORR_THRESH)
        controls.addWidget(self.corr_spin, 3, 1)
        controls.addWidget(QtWidgets.QLabel("MSQI ≥ (%)"), 3, 2)
        self.msqi_spin = QtWidgets.QDoubleSpinBox()
        self.msqi_spin.setRange(0.0, 100.0); self.msqi_spin.setSingleStep(1.0)
        self.msqi_spin.setDecimals(0); self.msqi_spin.setValue(PPGSQAEngine.MSQI_THRESH)
        controls.addWidget(self.msqi_spin, 3, 3)
        controls.addWidget(QtWidgets.QLabel("SNR ≥ (dB)"), 3, 4)
        self.snr_spin = QtWidgets.QDoubleSpinBox()
        self.snr_spin.setRange(-10.0, 60.0); self.snr_spin.setSingleStep(1.0)
        self.snr_spin.setDecimals(1); self.snr_spin.setValue(PPGSQAEngine.SNR_FLOOR_DB)
        controls.addWidget(self.snr_spin, 3, 5)
        for sp in (self.corr_spin, self.msqi_spin, self.snr_spin):
            sp.valueChanged.connect(self._on_threshold_changed)

        # Eye-flagging (manual ground truth) + unfit review
        controls.addWidget(QtWidgets.QLabel("Eye flag:"), 4, 0)
        self.flag_good_btn = QtWidgets.QPushButton("Good")
        self.flag_good_btn.clicked.connect(lambda: self._set_flag("good"))
        controls.addWidget(self.flag_good_btn, 4, 1)
        self.flag_bad_btn = QtWidgets.QPushButton("Bad")
        self.flag_bad_btn.clicked.connect(lambda: self._set_flag("bad"))
        controls.addWidget(self.flag_bad_btn, 4, 2)
        self.flag_clear_btn = QtWidgets.QPushButton("Clear")
        self.flag_clear_btn.clicked.connect(lambda: self._set_flag(None))
        controls.addWidget(self.flag_clear_btn, 4, 3)
        self.review_unfit_btn = QtWidgets.QPushButton("Review unfit ▶")
        self.review_unfit_btn.clicked.connect(self.review_unfit)
        self.review_unfit_btn.setEnabled(False)
        controls.addWidget(self.review_unfit_btn, 4, 4)
        self.flag_status = QtWidgets.QLabel("—")
        controls.addWidget(self.flag_status, 4, 5)

        # Verdict label
        self.metrics_label = QtWidgets.QLabel("Select a record and click Assess && Plot.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.metrics_label)

        # Figure
        self.figure = Figure(figsize=(12, 8.5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas)

        self._ppg_ax = None
        self._ppg_tmax = 0.0
        self.refresh_records()

    # ----------------------------------------------------------------- #
    def browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(self.dataset_dir))
        if path:
            self.dataset_edit.setText(path)
            self.refresh_records()

    def refresh_records(self):
        self.dataset_dir = Path(self.dataset_edit.text().strip())
        self.record_combo.clear()
        if not self.dataset_dir.exists():
            self.metrics_label.setText(f"Folder not found: {self.dataset_dir}")
            self.records = []
            return
        self.records = ds.discover_ppg(str(self.dataset_dir))
        self._load_flags()
        self.record_combo.addItems([ds.short_label(n) for _, n in self.records])
        if self.records:
            self.metrics_label.setText(
                f"{len(self.records)} PPG records ({len(self.flags)} eye-flagged). "
                f"Pick one and Assess && Plot, or Evaluate all.")
        else:
            self.metrics_label.setText(
                f"No '*{ds.PPG_SUFFIX}' records under {self.dataset_dir}")
        self._update_nav_buttons()

    def step_record(self, delta):
        n = self.record_combo.count()
        if n == 0:
            return
        i = min(n - 1, max(0, self.record_combo.currentIndex() + delta))
        if i != self.record_combo.currentIndex():
            self.record_combo.setCurrentIndex(i)
            self.assess_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n, i = self.record_combo.count(), self.record_combo.currentIndex()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)

    # ----------------------------------------------------------------- #
    def _load_current(self):
        i = self.record_combo.currentIndex()
        if i < 0 or i >= len(self.records):
            return None
        base, _ = self.records[i]
        col = ds.PPG_CHANNELS[self.channel_combo.currentText()]
        return ds.load_primer_ppg(base, channel=col)

    def assess_and_plot(self):
        rec = self._load_current()
        if rec is None:
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return
        try:
            res = PPGSQAEngine(rec.ppg, rec.fs, overrides=self._overrides()).assess()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Assess failed", str(exc))
            return
        self._cur = (rec, res, rec.name)
        self.view_start.setMaximum(max(0.0, rec.duration - self.view_width.value()))
        self._draw()
        self._update_flag_label()

    # ----------------------------------------------------------------- #
    def _draw(self):
        rec, res, name = self._cur
        fs = rec.fs
        t = rec.time
        y = res["filtered"]
        color = TIER_COLOR[res["tier"]]
        sqi = res["sqi"]

        # Verdict + the metrics that drove it; eye-flag + SSQI as a footnote.
        thr = res["thresholds"]
        corr = res.get("beat_corr", float("nan"))
        hr = res.get("pulse_bpm", float("nan"))
        reg = res.get("reg_frac", float("nan"))
        msqi = res.get("msqi", float("nan"))
        corr_s = f"{corr:.2f}" if np.isfinite(corr) else "n/a"
        hr_s = f"{hr:.0f}" if np.isfinite(hr) else "n/a"
        reg_s = f"{reg:.2f}" if np.isfinite(reg) else "n/a"
        flag_s = {None: "unflagged", "good": "GOOD", "bad": "BAD"}[self.flags.get(name)]
        self.metrics_label.setText(
            f"{ds.short_label(name)}  [{rec.channel}]  fs {fs:.0f} Hz  —  {res['label']}\n"
            f"decision: corr {corr_s} (≥{thr['BEAT_CORR_THRESH']:.2f})  ·  "
            f"MSQI {msqi:.0f} (≥{thr['MSQI_THRESH']:.0f})  ·  "
            f"SNR {res['snr_db']:.1f} dB (≥{thr['SNR_FLOOR_DB']:.1f})  ·  "
            f"RR reg {reg_s}  ·  HR {hr_s} bpm  ·  {res.get('n_beats', 0)} beats"
            f"   |   eye-flag: {flag_s}   ·   diag SSQI {sqi['SSQI']:+.3f}")
        self.metrics_label.setStyleSheet(f"font-weight: bold; color: {color};")

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
        ax_ppg = self.figure.add_subplot(gs[0, :])
        ax_psd = self.figure.add_subplot(gs[1, 0])
        ax_sqi = self.figure.add_subplot(gs[1, 1])

        # 1) PPG trace + detected peaks (time axis offset by the trimmed warm-up)
        start = res.get("start_idx", 0)
        ts = t[start:start + y.size]
        ax_ppg.plot(ts, y, lw=0.8, color="tab:blue", label="PPG (0.5–8 Hz)")
        peaks = res.get("peaks")
        if peaks is not None and len(peaks):
            ax_ppg.scatter(ts[peaks], y[peaks], s=28, c="tab:red", marker="v",
                           zorder=3, label=f"systolic peaks ({len(peaks)})")
        ax_ppg.set_title(f"{ds.short_label(name)} — {res['label']}", color=color)
        ax_ppg.set_xlabel("Time (s)"); ax_ppg.set_ylabel("PPG (filtered)")
        ax_ppg.grid(alpha=0.3); ax_ppg.legend(loc="upper right", fontsize=8)
        self._ppg_ax = ax_ppg
        self._ppg_tmax = float(ts[-1]) if ts.size else 0.0
        self._apply_view(redraw=False)

        # 2) power spectrum with cardiac & noise bands, SNR annotation
        f, pxx = res["psd_f"], res["psd_pxx"]
        ax_psd.semilogy(f, pxx + 1e-20, lw=0.8, color="0.3")
        sb, nb = res["sig_band"], res["noise_band"]
        ax_psd.axvspan(sb[0], sb[1], color="tab:green", alpha=0.12,
                       label=f"cardiac {sb[0]:g}–{sb[1]:g} Hz")
        ax_psd.axvspan(nb[0], nb[1], color="tab:red", alpha=0.10,
                       label=f"noise {nb[0]:g} Hz–Nyq")
        if np.isfinite(res["pulse_hz"]):
            ax_psd.axvline(res["pulse_hz"], ls="--", color="tab:blue", lw=1.0,
                           label=f"pulse {res['pulse_hz']:.2f} Hz ({res['pulse_hz']*60:.0f} bpm)")
        ax_psd.set_title(f"Spectrum — SNR = {res['snr_db']:.1f} dB")
        ax_psd.set_xlabel("Frequency (Hz)"); ax_psd.set_ylabel("PSD")
        ax_psd.set_xlim(0, 0.5 * fs)
        ax_psd.grid(alpha=0.3, which="both"); ax_psd.legend(loc="upper right", fontsize=7)

        # 3) decision criteria vs their (current, possibly tuned) thresholds
        names = [d[0] for d in DECISION_CRITERIA]
        ratios, bar_colors, annots = [], [], []
        for label, vkey, kind, tkey in DECISION_CRITERIA:
            thresh = (thr[tkey[0]], thr[tkey[1]]) if kind == "range" else thr[tkey]
            ok, ratio, annot = _criterion_status(res.get(vkey, float("nan")), kind, thresh)
            ratios.append(ratio)
            bar_colors.append(TIER_COLOR["excellent"] if ok else TIER_COLOR["unfit"])
            annots.append(annot)
        ypos = np.arange(len(names))
        ax_sqi.barh(ypos, ratios, color=bar_colors, alpha=0.85)
        ax_sqi.axvline(1.0, ls="--", color="0.4", lw=1.0, label="pass threshold")
        ax_sqi.set_yticks(ypos); ax_sqi.set_yticklabels(names, fontsize=8)
        ax_sqi.invert_yaxis()
        for yi, rr, lab in zip(ypos, ratios, annots):
            ax_sqi.text(rr + 0.03, yi, lab, va="center", fontsize=8)
        ax_sqi.set_xlim(0, 2.2)
        ax_sqi.set_title("Decision criteria (value ÷ threshold; beat corr = primary)")
        ax_sqi.set_xlabel("normalised score (≥1 = pass)")
        ax_sqi.legend(loc="lower right", fontsize=7)

        self.canvas.draw_idle()

    def _apply_view(self, redraw=True):
        if self._ppg_ax is None:
            return
        start = max(0.0, self.view_start.value())
        end = min(start + self.view_width.value(), self._ppg_tmax)
        if start >= end:
            start, end = 0.0, self._ppg_tmax
        self._ppg_ax.set_xlim(start, end)
        if redraw:
            self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def evaluate_all(self):
        if not self.records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return
        col = ds.PPG_CHANNELS[self.channel_combo.currentText()]

        progress = QtWidgets.QProgressDialog(
            "Assessing all PPG records...", "Cancel", 0, len(self.records), self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Evaluate all")
        progress.show()

        tiers = {"excellent": 0, "acceptable": 0, "unfit": 0}
        pts = []                 # (beat_corr, SNR, tier)
        unfit = []               # [(base, name), ...] for "Review unfit"
        overrides = self._overrides()
        for k, (base, name) in enumerate(self.records):
            if progress.wasCanceled():
                break
            progress.setValue(k); QtWidgets.QApplication.processEvents()
            try:
                rec = ds.load_primer_ppg(base, channel=col)
                res = PPGSQAEngine(rec.ppg, rec.fs, overrides=overrides).assess()
            except Exception:
                continue
            tiers[res["tier"]] += 1
            if res["tier"] == "unfit":
                unfit.append((base, name))
            corr, snr = res.get("beat_corr", float("nan")), res["snr_db"]
            if np.isfinite(corr) and np.isfinite(snr):
                pts.append((corr, snr, res["tier"]))
        progress.close()
        self._unfit, self._unfit_pos = unfit, -1
        self.review_unfit_btn.setEnabled(bool(unfit))
        self.review_unfit_btn.setText(
            f"Review unfit ({len(unfit)}) ▶" if unfit else "Review unfit ▶")

        total = sum(tiers.values())
        if total == 0:
            QtWidgets.QMessageBox.warning(self, "No results", "No records assessed.")
            return
        valid = tiers["excellent"] + tiers["acceptable"]
        self.metrics_label.setStyleSheet("font-weight: bold;")
        self.metrics_label.setText(
            f"[{self.channel_combo.currentText()}] {total} records | "
            f"valid {valid} ({100*valid/total:.0f}%): excellent {tiers['excellent']}, "
            f"acceptable {tiers['acceptable']} | invalid (unfit) {tiers['unfit']}")

        self.figure.clear()
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2)

        labels = ["Excellent", "Acceptable", "Unfit\n(invalid)"]
        counts = [tiers["excellent"], tiers["acceptable"], tiers["unfit"]]
        bar_colors = [TIER_COLOR["excellent"], TIER_COLOR["acceptable"], TIER_COLOR["unfit"]]
        ax1.bar(labels, counts, color=bar_colors, alpha=0.9)
        for i, c in enumerate(counts):
            ax1.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
        ax1.set_ylabel("records"); ax1.set_title(f"Validity distribution (n={total})")
        ax1.grid(alpha=0.3, axis="y")

        if pts:
            corr = np.array([p[0] for p in pts])
            snr = np.array([p[1] for p in pts])
            cols = [TIER_COLOR[p[2]] for p in pts]
            ax2.scatter(corr, snr, c=cols, s=28, alpha=0.8, edgecolors="0.3", lw=0.3)
            ax2.axvline(PPGSQAEngine.BEAT_CORR_THRESH, ls="--", color="0.5", lw=0.9,
                        label=f"corr threshold ({PPGSQAEngine.BEAT_CORR_THRESH:.2f})")
            ax2.legend(loc="lower right", fontsize=7)
        ax2.set_xlabel("Beat-template correlation — primary index")
        ax2.set_ylabel("SNR (dB)")
        ax2.set_title("Quality space (colour = validity tier)")
        ax2.grid(alpha=0.3)

        self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def _overrides(self):
        """Current tuning-spinbox values as PPGSQAEngine threshold overrides."""
        return {"BEAT_CORR_THRESH": self.corr_spin.value(),
                "MSQI_THRESH": self.msqi_spin.value(),
                "SNR_FLOOR_DB": self.snr_spin.value()}

    def _on_threshold_changed(self, *_):
        if self._cur is not None:        # re-assess the loaded record with new thresholds
            self.assess_and_plot()

    def review_unfit(self):
        """Step through the records the last 'Evaluate all' marked unfit."""
        if not self._unfit:
            QtWidgets.QMessageBox.information(
                self, "Review unfit", "Run 'Evaluate all' first (no unfit records).")
            return
        self._unfit_pos = (self._unfit_pos + 1) % len(self._unfit)
        _, name = self._unfit[self._unfit_pos]
        idx = next((i for i, (_, n) in enumerate(self.records) if n == name), -1)
        if idx >= 0:
            self.record_combo.blockSignals(True)
            self.record_combo.setCurrentIndex(idx)
            self.record_combo.blockSignals(False)
        self.assess_and_plot()
        self.review_unfit_btn.setText(f"Unfit {self._unfit_pos + 1}/{len(self._unfit)} ▶")

    # ── Eye-flagging (manual ground truth) ──────────────────────────────────
    def _load_flags(self):
        self._flags_path = self.dataset_dir / "ppg_quality_labels.json"
        self.flags = {}
        if self._flags_path.exists():
            try:
                with open(self._flags_path, encoding="utf-8") as fh:
                    self.flags = json.load(fh)
            except (ValueError, OSError):
                self.flags = {}

    def _save_flags(self):
        if self._flags_path is None:
            return
        try:
            with open(self._flags_path, "w", encoding="utf-8") as fh:
                json.dump(self.flags, fh, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))

    def _set_flag(self, value):
        if self._cur is None:
            return
        name = self._cur[2]
        if value is None:
            self.flags.pop(name, None)
        else:
            self.flags[name] = value
        self._save_flags()
        self._update_flag_label()

    def _update_flag_label(self):
        if self._cur is None:
            self.flag_status.setText("—")
            return
        flag = self.flags.get(self._cur[2])
        text = {None: "unflagged", "good": "⚑ GOOD", "bad": "⚑ BAD"}[flag]
        col = {None: "#555", "good": TIER_COLOR["excellent"],
               "bad": TIER_COLOR["unfit"]}[flag]
        self.flag_status.setText(text)
        self.flag_status.setStyleSheet(f"font-weight: bold; color: {col};")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PPGSQAGui()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
