"""
PPG Signal Quality Assessment (SQA) from the eight signal-quality indices of

    Elgendi, M. "Optimal Signal Quality Index for Photoplethysmogram Signals."
    Bioengineering 2016, 3(4), 21.

The paper found the **skewness index (SSQI)** to be the single optimal SQI for
separating excellent PPG from acceptable/unfit, with a fixed threshold of zero
(Figure 4). This engine still computes all eight indices and a spectral SNR as
diagnostics, but does **not** decide on skewness: its sign is confounded by
sensor orientation (a clean but inverted pulse scores negative). The validity
verdict instead follows an Orphanidou-2015-style rule -- repeatable beat
morphology (template correlation) at a plausible, regular pulse rate, with SNR
as a backstop. See :meth:`PPGSQAEngine.classify`.

Indices (Section 2.3 of the paper)
----------------------------------
    PSQI  perfusion           (y_max - y_min) / |mean(raw)| x 100   gold standard
    SSQI  skewness            skew(y)                               OPTIMAL, thr = 0
    KSQI  kurtosis            Pearson kurtosis of y
    ESQI  entropy             -sum p^2 ln p^2   (energy-normalised)
    ZSQI  zero crossing       fraction of y < 0, x 100              lower better
    NSQI  signal-to-noise     var(|y|) / var(y)                     paper's SNR index
    MSQI  detector matching   |Bing ∩ Billauer| / |Bing| x 100      higher better
    RSQI  relative power      P(1-2.25 Hz) / P(0-8 Hz)              higher better

``y`` is the band-pass filtered PPG (0.5-8 Hz); the raw signal keeps its DC term
only where the paper's formula calls for it (PSQI mean). A practical spectral
SNR (dB) around the dominant pulse frequency is reported alongside NSQI.

Usage
-----
    from ppg_sqa import PPGSQAEngine
    res = PPGSQAEngine(ppg, fs).assess()
    res["valid"]      # True / False
    res["tier"]       # "excellent" | "acceptable" | "unfit"
    res["beat_corr"]  # mean beat-to-template correlation (decision: primary)
    res["pulse_bpm"]  # detected pulse rate (bpm)
    res["snr_db"]     # spectral SNR in dB (decision: backstop)
    res["sqi"]        # dict of the eight Elgendi indices (diagnostics only)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import kurtosis, skew

__all__ = ["PPGSQAEngine"]

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class PPGSQAEngine:
    """Photoplethysmogram signal-quality assessment engine."""

    # ── Preprocessing ──────────────────────────────────────────────────────
    BANDPASS = (0.5, 8.0)     # Hz, cardiac analysis band (clipped below Nyquist)

    # ── Decision thresholds (the GUI can override these per-instance) ───────
    # The verdict follows an Orphanidou-2015-style rule -- repeatable beat
    # morphology at a plausible, regular rate -- with detector agreement (MSQI)
    # and spectral SNR refining the "excellent" tier:
    #   primary    -- beat-to-template correlation (is the pulse shape repeatable?)
    #   supporting -- plausible, regular pulse rate; MSQI and SNR above a floor.
    # Skewness (SSQI) is NOT used: its sign is confounded by sensor orientation,
    # so a clean-but-inverted pulse scores negative. The eight Elgendi SQIs are
    # still computed and reported as diagnostics.
    BEAT_CORR_THRESH = 0.80   # min mean beat-vs-template r to be HR-derivable (valid)
    BEAT_CORR_EXCELLENT = 0.90  # stronger r for the "excellent" tier
    MSQI_THRESH = 74.0        # min detector-agreement % for the "excellent" tier
    SNR_FLOOR_DB = 3.0        # min spectral SNR (dB) for the "excellent" tier
    HR_MIN_BPM, HR_MAX_BPM = 40.0, 180.0   # plausible resting/active pulse range
    REG_FRAC_MIN = 0.66       # min fraction of RR intervals within +/-30% of median
    MIN_BEATS = 4             # need at least this many beats to judge morphology
    BEAT_PRE_S, BEAT_POST_S = 0.25, 0.45   # peak-aligned beat window (s before/after)
    TEMPLATE_LEN = 100        # samples each beat window is resampled to

    EDGE_TRIM_S = 0.5         # s discarded each end: sensor warm-up + filtfilt edges

    def __init__(self, ppg, fs, *, overrides=None):
        self.fs = float(fs)
        # Per-instance threshold overrides (e.g. from the GUI tuning controls)
        # shadow the class defaults that classify() reads.
        for name, value in (overrides or {}).items():
            setattr(self, name, value)
        raw_full = np.asarray(ppg, dtype=float)
        filt_full = self._filter(raw_full)
        # Drop the sensor-settling onset (and the symmetric tail) so a warm-up
        # transient cannot inflate skewness / spawn a phantom peak; genuine
        # mid-recording motion artifacts are left intact for the SQIs to catch.
        trim = int(self.EDGE_TRIM_S * self.fs)
        if trim > 0 and raw_full.size > 4 * trim:
            self.start_idx = trim
            self.raw = raw_full[trim:raw_full.size - trim]
            self.filtered = filt_full[trim:filt_full.size - trim]
        else:
            self.start_idx = 0
            self.raw = raw_full
            self.filtered = filt_full

    # ── Preprocessing ──────────────────────────────────────────────────────
    def _filter(self, sig):
        lo, hi = self.BANDPASS
        hi = min(hi, 0.45 * self.fs)
        if sig.size < 30 or lo <= 0 or hi <= lo:
            return sig - np.mean(sig)
        try:
            b, a = butter(2, [lo / (0.5 * self.fs), hi / (0.5 * self.fs)], btype="band")
            return filtfilt(b, a, sig)
        except Exception:
            return sig - np.mean(sig)

    # ── Systolic-wave detectors (for MSQI) ─────────────────────────────────
    def _min_distance(self):
        return max(1, int(0.30 * self.fs))      # >=300 ms apart (<=200 bpm)

    def billauer_peaks(self):
        """Local-maxima detector: peaks >=300 ms apart, prominent vs the noise."""
        y = self.filtered
        if y.size < 3:
            return np.array([], dtype=int)
        prom = 0.3 * np.std(y)
        peaks, _ = find_peaks(y, distance=self._min_distance(), prominence=prom)
        return peaks

    def bing_peaks(self):
        """First-derivative detector: systolic peaks following max-slope upstrokes."""
        y = self.filtered
        if y.size < 3:
            return np.array([], dtype=int)
        dy = np.gradient(y)
        slope_energy = np.clip(dy, 0, None) ** 2          # rising edges only
        thr = np.mean(slope_energy)
        upstrokes, _ = find_peaks(slope_energy, distance=self._min_distance(),
                                  height=thr)
        search = max(1, int(0.25 * self.fs))
        peaks = []
        for u in upstrokes:                                # peak just after upstroke
            seg = y[u:min(u + search, y.size)]
            if seg.size:
                peaks.append(u + int(np.argmax(seg)))
        return np.array(sorted(set(peaks)), dtype=int)

    # ── The eight SQIs ─────────────────────────────────────────────────────
    def perfusion(self):
        """PSQI: pulsatile range over DC level (gold-standard perfusion index)."""
        y = self.filtered
        dc = abs(float(np.mean(self.raw)))
        if dc == 0:
            return 0.0
        return float((y.max() - y.min()) / dc * 100.0)

    def skewness(self):
        """SSQI: skewness of the AC signal (the paper's optimal index)."""
        return float(skew(self.filtered))

    def kurtosis_(self):
        """KSQI: Pearson kurtosis of the AC signal (~3 = Gaussian)."""
        return float(kurtosis(self.filtered, fisher=False))

    def entropy(self):
        """ESQI: energy-distribution entropy, -sum p^2 ln p^2 with p energy-norm."""
        y = self.filtered
        e = float(np.sum(y ** 2))
        if e == 0:
            return 0.0
        p2 = (y ** 2) / e                                  # normalised energy, sums to 1
        nz = p2 > 0
        return float(-np.sum(p2[nz] * np.log(p2[nz])))

    def zero_crossing(self):
        """ZSQI: fraction of filtered samples below zero, as a percent (Eq. 6)."""
        y = self.filtered
        return float(np.mean(y < 0) * 100.0) if y.size else 0.0

    def nsqi(self):
        """NSQI: the paper's SNR index, var(|y|) / var(y) (Eq. 7)."""
        y = self.filtered
        v = float(np.var(y))
        if v == 0:
            return 0.0
        return float(np.var(np.abs(y)) / v)

    def relative_power(self):
        """RSQI: cardiac-band (1-2.25 Hz) power over total 0-8 Hz power (Eq. 9)."""
        f, pxx = self._psd()
        total = _trapz(pxx[(f >= 0) & (f <= 8)], f[(f >= 0) & (f <= 8)])
        if total <= 0:
            return 0.0
        band = (f >= 1.0) & (f <= 2.25)
        num = _trapz(pxx[band], f[band])
        return float(num / total)

    def msqi(self):
        """MSQI: agreement between the Bing and Billauer systolic detectors."""
        bing = self.bing_peaks()
        billauer = self.billauer_peaks()
        if bing.size == 0:
            return 0.0
        tol = 0.15 * self.fs                               # 150 ms match window
        matched = sum(1 for p in bing
                      if billauer.size and np.min(np.abs(billauer - p)) <= tol)
        return float(matched / bing.size * 100.0)

    # ── Spectra ────────────────────────────────────────────────────────────
    def _psd(self):
        """PSD of the band-pass filtered signal (used for RSQI, 0-8 Hz)."""
        y = self.filtered
        nper = int(min(y.size, max(256, 4 * self.fs)))
        f, pxx = welch(y, fs=self.fs, nperseg=nper)
        return f, pxx

    def _raw_psd(self):
        """Full-band PSD of the DC-removed raw signal (up to Nyquist).

        Keeps the high-frequency content the band-pass filter would discard, so
        the noise floor used by :meth:`spectral_snr` is genuine sensor noise.
        """
        x = self.raw - np.mean(self.raw)
        nper = int(min(x.size, max(256, 4 * self.fs)))
        f, pxx = welch(x, fs=self.fs, nperseg=nper)
        return f, pxx

    # ── Spectral SNR (dB) ──────────────────────────────────────────────────
    SIG_BAND = (0.5, 8.0)     # Hz, cardiac (pulse + harmonics)
    NOISE_BAND = (8.0, None)  # Hz, high-frequency noise (None -> Nyquist)

    def spectral_snr(self):
        """SNR (dB): mean spectral density in the cardiac band vs the HF noise band.

        Comparing average PSD per Hz (not integrated power) keeps the figure
        bounded and bandwidth-fair: clean PPG concentrates energy in 0.5-8 Hz
        while sensor/motion noise raises the 8 Hz-Nyquist floor. The dominant
        pulse frequency (PSD peak in 0.5-4 Hz) is returned for display.
        Returns ``(snr_db, f0_hz)``.
        """
        f, pxx = self._raw_psd()
        nyq = 0.5 * self.fs
        sig = (f >= self.SIG_BAND[0]) & (f <= self.SIG_BAND[1])
        noise = (f > self.NOISE_BAND[0]) & (f <= nyq)
        if not sig.any():
            return float("nan"), float("nan")
        search_f = f[(f >= 0.5) & (f <= 4.0)]
        search_p = pxx[(f >= 0.5) & (f <= 4.0)]
        f0 = float(search_f[int(np.argmax(search_p))]) if search_f.size else float("nan")
        p_sig = float(np.mean(pxx[sig]))
        p_noise = float(np.mean(pxx[noise])) if noise.any() else 0.0
        if p_noise <= 0 or p_sig <= 0:
            return float("nan"), f0
        return float(10.0 * np.log10(p_sig / p_noise)), f0

    # ── Pulse morphology + rate (the decision indices) ──────────────────────
    def pulse_rate(self):
        """Pulse rate and rhythm regularity from the systolic peaks.

        Returns ``(hr_bpm, reg_frac, n_beats)`` where ``reg_frac`` is the fraction
        of RR intervals within +/-30% of the median (a robust regularity score;
        1.0 == perfectly regular, and unlike max/min it is not wrecked by a single
        missed or doubled beat). ``hr_bpm`` / ``reg_frac`` are ``nan`` when there
        are too few beats.
        """
        pk = self.billauer_peaks()
        if pk.size < 2:
            return float("nan"), float("nan"), int(pk.size)
        rr = np.diff(pk) / self.fs
        rr = rr[(rr > 0.25) & (rr < 2.0)]            # drop implausible intervals
        if rr.size == 0:
            return float("nan"), float("nan"), int(pk.size)
        med = float(np.median(rr))
        hr = 60.0 / med
        reg_frac = float(np.mean(np.abs(rr - med) <= 0.3 * med))
        return hr, reg_frac, int(pk.size)

    def beat_template_correlation(self):
        """Mean correlation of each beat against the average beat template.

        Each beat is a fixed window centred on its systolic peak (``BEAT_PRE_S``
        before to ``BEAT_POST_S`` after), resampled to :data:`TEMPLATE_LEN`
        samples; the template is their mean and the score is the mean Pearson r of
        every beat against it (cf. ``build_template_from_peaks`` in
        ``ppg_current_sqa.py``). Peak-aligning the windows lets a repeatable pulse
        shape score high regardless of beat-to-beat interval changes. Returns
        ``nan`` when there are too few usable beats.
        """
        y = self.filtered
        pk = self.billauer_peaks()
        if pk.size < self.MIN_BEATS:
            return float("nan")
        pre = int(self.BEAT_PRE_S * self.fs)
        post = int(self.BEAT_POST_S * self.fs)
        grid = np.linspace(0.0, 1.0, self.TEMPLATE_LEN)
        beats = []
        for p in pk:
            if p - pre < 0 or p + post > y.size:
                continue                              # window runs off the edge
            w = y[p - pre:p + post]
            beats.append(np.interp(grid, np.linspace(0.0, 1.0, w.size), w))
        if len(beats) < 2:
            return float("nan")
        beats = np.asarray(beats)
        template = beats.mean(axis=0)
        if np.std(template) == 0:
            return float("nan")
        corrs = [float(np.corrcoef(b, template)[0, 1])
                 for b in beats if np.std(b) > 0]
        return float(np.mean(corrs)) if corrs else float("nan")

    # ── Full assessment ────────────────────────────────────────────────────
    def compute_sqi(self):
        return {
            "PSQI": self.perfusion(),
            "SSQI": self.skewness(),
            "KSQI": self.kurtosis_(),
            "ESQI": self.entropy(),
            "ZSQI": self.zero_crossing(),
            "NSQI": self.nsqi(),
            "MSQI": self.msqi(),
            "RSQI": self.relative_power(),
        }

    def classify(self, m):
        """Validity tier from pulse morphology + rate, refined by MSQI and SNR.

        ``m`` carries ``beat_corr``, ``pulse_bpm``, ``reg_frac``, ``n_beats``,
        ``msqi`` and ``snr_db``. A signal is HR-derivable (acceptable) when its
        beats are both *repeatable* (correlation) and at a *plausible, regular*
        rate; strong correlation plus detector agreement (MSQI) and SNR above
        their floors promote it to excellent. Returns ``(valid, tier, reason)``.
        """
        corr_ok = np.isfinite(m["beat_corr"]) and m["beat_corr"] >= self.BEAT_CORR_THRESH
        corr_strong = np.isfinite(m["beat_corr"]) and m["beat_corr"] >= self.BEAT_CORR_EXCELLENT
        hr_ok = (m["n_beats"] >= self.MIN_BEATS
                 and np.isfinite(m["pulse_bpm"])
                 and self.HR_MIN_BPM <= m["pulse_bpm"] <= self.HR_MAX_BPM
                 and np.isfinite(m["reg_frac"]) and m["reg_frac"] >= self.REG_FRAC_MIN)
        msqi_ok = np.isfinite(m["msqi"]) and m["msqi"] >= self.MSQI_THRESH
        snr_ok = np.isfinite(m["snr_db"]) and m["snr_db"] >= self.SNR_FLOOR_DB

        if corr_ok and corr_strong and hr_ok and msqi_ok and snr_ok:
            return True, "excellent", "highly repeatable beats, plausible rate, MSQI & SNR pass"
        if corr_ok and hr_ok:
            return True, "acceptable", "HR-derivable (repeatable beats at a plausible rate)"
        reasons = []
        if not corr_ok:
            reasons.append("inconsistent beat shape")
        if not hr_ok:
            reasons.append("implausible / irregular rate")
        return False, "unfit", ", ".join(reasons) or "fails quality criteria"

    def assess(self):
        """Run the decision and return a result dict.

        The eight Elgendi SQIs are still computed (``r['sqi']``) as diagnostics,
        but the verdict comes from :meth:`classify` on the morphology/rate/SNR
        metrics, not from skewness. A dead, flat, or saturated capture needs no
        special gate: it yields no detectable beats (failing rate plausibility)
        or distorted ones (failing the correlation), so it lands in ``unfit``.
        """
        f, pxx = self._raw_psd()
        snr_db, f0 = self.spectral_snr()
        r = {
            "fs": self.fs, "filtered": self.filtered, "start_idx": self.start_idx,
            "psd_f": f, "psd_pxx": pxx, "snr_db": snr_db, "pulse_hz": f0,
            "sig_band": self.SIG_BAND, "noise_band": (self.NOISE_BAND[0], 0.5 * self.fs),
            "peaks": self.billauer_peaks(),
            "thresholds": {
                "BEAT_CORR_THRESH": self.BEAT_CORR_THRESH,
                "MSQI_THRESH": self.MSQI_THRESH, "SNR_FLOOR_DB": self.SNR_FLOOR_DB,
                "REG_FRAC_MIN": self.REG_FRAC_MIN,
                "HR_MIN_BPM": self.HR_MIN_BPM, "HR_MAX_BPM": self.HR_MAX_BPM,
            },
        }

        sqi = self.compute_sqi()
        hr, reg_frac, n_beats = self.pulse_rate()
        beat_corr = self.beat_template_correlation()
        metrics = {"beat_corr": beat_corr, "pulse_bpm": hr, "reg_frac": reg_frac,
                   "n_beats": n_beats, "msqi": sqi["MSQI"], "snr_db": snr_db}
        valid, tier, reason = self.classify(metrics)
        verdict = {"excellent": "Valid – Excellent quality",
                   "acceptable": "Valid – Acceptable (HR-quality)",
                   "unfit": "Invalid – Unfit for diagnosis"}[tier]
        corr_str = f"{beat_corr:.2f}" if np.isfinite(beat_corr) else "n/a"
        hr_str = f"{hr:.0f}" if np.isfinite(hr) else "n/a"
        r.update(sqi=sqi, **metrics, valid=valid, tier=tier, reason=reason,
                 label=f"{verdict}  (corr {corr_str}, HR {hr_str} bpm, "
                       f"SNR {snr_db:.1f} dB)")
        return r
