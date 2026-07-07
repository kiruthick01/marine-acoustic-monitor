"""
Signal conditioning pipeline.

Implements Stage 0 of the ML pipeline described in docs/ml-pipeline.md: a
pre-processing pass that runs on raw captured audio before Stage 1 feature
extraction (simulation/pipeline/feature_extraction.py) ever sees it. Two
steps, applied in order:

1. bandpass_filter() -- isolate the frequency band that actually carries
   signal of interest (biological calls, vessel tonal peaks), cutting very
   low-frequency flow/handling noise and very high-frequency electronic
   noise that contribute nothing but variance to the downstream features.
2. spectral_denoise() -- basic spectral-subtraction denoising against an
   estimated noise floor, to reduce broadband ambient noise energy before
   features (especially RMS energy and spectral flatness, per
   feature_extraction.py) are computed from the signal.

condition_signal() runs both in sequence and is the single entry point
wired into simulation/scripts/run_simulation.py, immediately before
extract_acoustic_features().

This operates on the same synthetic data generators as feature_extraction.py
(see simulation/data_generator/synthetic_audio.py) -- see DECISIONS.md,
project status is planning/no hardware yet, so "noise" here is the
synthetic pink ambient background rather than a real hydrophone's self-noise
or flow noise, but the conditioning steps are written to be equally valid
once real recordings exist.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal

# STFT frame size shared by estimate_noise_floor() and spectral_denoise() so
# their frequency bins line up for subtraction without any resampling step.
# 1024 samples is a conventional choice for speech/bioacoustic spectral
# subtraction -- long enough for reasonable frequency resolution (~21.5 Hz
# bins at 22050 Hz) while short enough to track the noise floor drifting
# over the course of a multi-second capture window.
STFT_NPERSEG = 1024

# Duration of the "quietest segment" used to estimate the noise floor and
# the pre/post SNR diagnostic. 1 second is short enough to reliably land
# inside a gap between events in a multi-second capture window, long enough
# to average out per-frame variance in the noise estimate.
NOISE_SEGMENT_DURATION_S = 1.0


def _quietest_segment_indices(
    audio: np.ndarray, sample_rate: int, segment_duration_s: float = NOISE_SEGMENT_DURATION_S
) -> Tuple[int, int]:
    """
    Locate the lowest-energy contiguous segment of `audio`, used as a stand-
    in for a "noise-only" reference since the synthetic/real audio has no
    separate noise-only channel to sample from directly.

    Slides a `segment_duration_s`-long window across `audio` in quarter-
    window hops and returns the (start, end) sample indices of the window
    with the lowest mean-square energy. Falls back to the whole array if
    `audio` is shorter than one segment.
    """
    segment_len = min(len(audio), max(int(segment_duration_s * sample_rate), 1))
    if segment_len >= len(audio):
        return 0, len(audio)

    hop = max(segment_len // 4, 1)
    best_start = 0
    best_energy = np.inf
    for start in range(0, len(audio) - segment_len + 1, hop):
        segment = audio[start : start + segment_len]
        energy = float(np.mean(segment.astype(np.float64) ** 2))
        if energy < best_energy:
            best_energy = energy
            best_start = start

    return best_start, best_start + segment_len


def bandpass_filter(
    audio: np.ndarray, sample_rate: int, low_hz: float = 20, high_hz: float = 4000, order: int = 4
) -> np.ndarray:
    """
    Butterworth bandpass filter isolating the frequency range relevant to
    biological calls and vessel noise tonal peaks.

    Vessel tonal peaks (shaft/blade-rate harmonics) concentrate around
    60-120 Hz (see synthetic_audio.py's vessel event generator, and the
    citation in its docstring); biological calls typically extend higher.
    The default 20-4000 Hz band retains both while cutting very
    low-frequency flow/handling noise below 20 Hz and very high-frequency
    electronic/self-noise above 4 kHz that carry no signal of interest but
    do add energy the downstream RMS/spectral-flatness features would
    otherwise pick up as variance.

    Note: this project's synthetic biological whistle (synthetic_audio.py)
    sweeps 4-12 kHz, i.e. mostly *above* this default's 4000 Hz upper edge.
    That's intentional here as the general-purpose default matching the
    real-world citation above, not a claim that it's tuned to this
    particular synthetic asset -- callers working specifically with the
    synthetic whistle generator should pass a higher `high_hz` (e.g. 12000)
    if they need to preserve it through conditioning.

    Uses `scipy.signal.sosfiltfilt` (second-order-sections, zero-phase) so
    the filter introduces no phase delay/smearing into the conditioned
    audio -- important since feature extraction downstream is sensitive to
    the audio's time-domain envelope (RMS std, ZCR), not just its spectrum.

    Args:
        audio: 1D audio array for one capture window.
        sample_rate: samples per second (Hz) of `audio`.
        low_hz: lower cutoff of the passband, in Hz.
        high_hz: upper cutoff of the passband, in Hz.
        order: Butterworth filter order. 4 is a moderate default -- steep
            enough to meaningfully attenuate out-of-band energy, low enough
            to avoid the numerical instability higher-order SOS bandpass
            designs can hit near the Nyquist frequency.

    Returns:
        Filtered 1D audio array, same length and dtype (float32) as input.
    """
    audio = np.asarray(audio, dtype=np.float32)
    nyquist = sample_rate / 2.0

    # Clamp high_hz just under Nyquist -- a bandpass edge placed exactly at
    # (or above) Nyquist is invalid and scipy.signal.butter raises for it.
    high_hz = min(high_hz, nyquist * 0.99)

    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, audio)

    return filtered.astype(np.float32)


def estimate_noise_floor(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Estimate a baseline noise magnitude spectrum from the quietest portion
    of `audio`, for use as the subtrahend in spectral_denoise().

    Finds the lowest-energy ~1-second segment (see
    _quietest_segment_indices()) and returns that segment's mean STFT
    magnitude spectrum -- i.e. what the "noise floor" looks like frequency-
    bin-by-frequency-bin, averaged over the segment's frames to smooth out
    single-frame estimation noise.

    This assumes the quietest segment is representative of the noise
    present throughout the whole window (no event happening, just ambient
    background) -- reasonable for the sparse, short synthetic/real events
    this pipeline targets (a call or vessel passage occupies a small
    fraction of a multi-second capture window), but it will overestimate
    the noise floor if the whole window is busy with signal, and
    underestimate it if the "quiet" segment happens to catch an unusually
    calm moment relative to the window's actual average noise level.

    Args:
        audio: 1D audio array for one capture window.
        sample_rate: samples per second (Hz) of `audio`.

    Returns:
        1D array of per-frequency-bin magnitude estimates, length
        matching the frequency axis of an STFT computed with
        nperseg=STFT_NPERSEG (or len(audio) if shorter).
    """
    audio = np.asarray(audio, dtype=np.float32)
    start, end = _quietest_segment_indices(audio, sample_rate)
    quiet_segment = audio[start:end]

    nperseg = min(STFT_NPERSEG, len(quiet_segment))
    _, _, quiet_stft = signal.stft(quiet_segment, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)

    return np.mean(np.abs(quiet_stft), axis=1)


def spectral_denoise(audio: np.ndarray, sample_rate: int, noise_floor: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Basic spectral-subtraction denoising: STFT the signal, subtract the
    estimated noise floor's magnitude from each frame's magnitude
    (clamping at zero), reconstruct via inverse STFT using the original
    (unmodified) phase.

    This is a baseline denoising method, not state-of-the-art. Known
    limitation: hard-flooring negative post-subtraction magnitude at zero
    produces "musical noise" -- isolated, randomly-appearing tonal
    artifacts, because individual time-frequency bins that dip briefly
    below the estimated noise level get zeroed discontinuously from frame
    to frame, and the remaining sparse random bins reconstruct as
    audible warbling tones. More robust approaches (Wiener filtering,
    oversubtraction with a spectral floor, minimum-statistics noise
    tracking) exist and would be the next step if conditioning quality
    turns out to matter more than this simple baseline provides -- see
    docs/ml-pipeline.md.

    Args:
        audio: 1D audio array for one capture window.
        sample_rate: samples per second (Hz) of `audio`.
        noise_floor: optional pre-computed noise magnitude spectrum (as
            returned by estimate_noise_floor()). If omitted, it's estimated
            from `audio` itself via estimate_noise_floor().

    Returns:
        Denoised 1D audio array, same length and dtype (float32) as input.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if noise_floor is None:
        noise_floor = estimate_noise_floor(audio, sample_rate)

    nperseg = min(STFT_NPERSEG, len(audio))
    noverlap = nperseg // 2
    _, _, audio_stft = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(audio_stft)
    phase = np.angle(audio_stft)

    # noise_floor's bin count tracks STFT_NPERSEG under normal use, matching
    # magnitude's bin count here; if a caller passes a noise_floor computed
    # with different framing (e.g. from a different-length audio array),
    # interpolate it onto this STFT's frequency axis rather than erroring,
    # since an approximate noise estimate is still better than none.
    if noise_floor.shape[0] != magnitude.shape[0]:
        noise_floor = np.interp(
            np.linspace(0, 1, magnitude.shape[0]), np.linspace(0, 1, noise_floor.shape[0]), noise_floor
        )

    subtracted_magnitude = np.maximum(magnitude - noise_floor[:, np.newaxis], 0.0)
    denoised_stft = subtracted_magnitude * np.exp(1j * phase)

    _, denoised_audio = signal.istft(denoised_stft, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    # istft's output length can differ slightly from the input's due to
    # frame/hop boundary effects; trim or zero-pad back to the original
    # length so callers (bandpass_filter, feature extraction) can keep
    # assuming a fixed-length window.
    if len(denoised_audio) > len(audio):
        denoised_audio = denoised_audio[: len(audio)]
    elif len(denoised_audio) < len(audio):
        denoised_audio = np.pad(denoised_audio, (0, len(audio) - len(denoised_audio)))

    return denoised_audio.astype(np.float32)


def _estimate_snr_db(audio: np.ndarray, sample_rate: int) -> float:
    """
    Rough SNR estimate in dB: ratio of the whole window's RMS level to the
    quietest segment's RMS level (used as a noise-only proxy), in dB.

    This is a coarse diagnostic, not a rigorous SNR measurement -- it
    conflates "signal" with "whole-window energy including the noise
    itself", so it's really (signal+noise)-to-noise rather than pure
    signal-to-noise. Good enough as a before/after conditioning sanity
    check (condition_signal()'s diagnostic dict), not intended for
    anything more precise.
    """
    audio = np.asarray(audio, dtype=np.float64)
    start, end = _quietest_segment_indices(audio, sample_rate)

    noise_rms = float(np.sqrt(np.mean(audio[start:end] ** 2)))
    signal_rms = float(np.sqrt(np.mean(audio**2)))

    eps = 1e-12
    return float(20 * np.log10((signal_rms + eps) / (noise_rms + eps)))


def condition_signal(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Full Stage 0 signal conditioning: bandpass_filter() then
    spectral_denoise(), applied in that order so denoising's noise-floor
    estimate is computed on the already band-limited signal (matching what
    Stage 1 feature extraction will actually see).

    bandpass_filter() is called with high_hz=12000 rather than its own
    20-4000 Hz general-purpose default: this pipeline's audio is produced
    exclusively by simulation/data_generator/synthetic_audio.py, whose
    biological whistle sweeps 4-12 kHz (see bandpass_filter()'s docstring)
    -- the 4000 Hz default would filter that call almost entirely out
    before it ever reached feature extraction, which is exactly the
    "callers working specifically with the synthetic whistle generator"
    case that docstring calls out.

    Args:
        audio: 1D audio array for one capture window (as produced by
            simulation/data_generator/synthetic_audio.py).
        sample_rate: samples per second (Hz) of `audio`.

    Returns:
        Tuple of:
        - conditioned audio: 1D float32 array, same length as input, after
          bandpass filtering and spectral denoising.
        - diagnostics: dict with `snr_before_db` and `snr_after_db`, a
          rough before/after SNR estimate (see _estimate_snr_db()) letting
          callers sanity-check that conditioning is actually improving the
          signal rather than degrading it on a given window.
    """
    audio = np.asarray(audio, dtype=np.float32)
    snr_before_db = _estimate_snr_db(audio, sample_rate)

    filtered = bandpass_filter(audio, sample_rate, high_hz=12000)
    conditioned = spectral_denoise(filtered, sample_rate)

    snr_after_db = _estimate_snr_db(conditioned, sample_rate)

    diagnostics = {
        "snr_before_db": snr_before_db,
        "snr_after_db": snr_after_db,
    }
    return conditioned, diagnostics


if __name__ == "__main__":
    from simulation.data_generator.synthetic_audio import generate_duty_cycle_sample

    # One capture window per anomaly type, at the same 5s duration
    # run_simulation.py actually uses (the 30s vessel-only demo this used
    # to run masked the biological-call regression: vessel's tonal energy
    # sits well under 4000 Hz so it survived the old default fine, but
    # only the wider 5-12 kHz duration_s=5 case here exposed it).
    for anomaly in [None, "vessel", "biological"]:
        audio, audio_meta = generate_duty_cycle_sample(duration_s=5, sample_rate=22050, inject_anomaly=anomaly)
        conditioned_audio, diagnostics = condition_signal(audio, sample_rate=22050)

        print(f"-- anomaly={anomaly} --")
        print("Audio ground truth:", audio_meta)
        print(f"Conditioned audio shape: {conditioned_audio.shape}")
        print(f"Diagnostics: {diagnostics}")
        print()
