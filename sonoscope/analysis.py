"""Décomposition fréquentielle d'un signal audio : FFT, spectrogramme, pics dominants."""

import numpy as np
from scipy.signal import find_peaks, spectrogram


def compute_fft(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (fréquences, magnitude) du spectre d'amplitude d'un signal mono."""
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    magnitude = np.abs(fft_vals) / n
    return freqs, magnitude


def find_dominant_frequencies(
    freqs: np.ndarray, magnitude: np.ndarray, num_peaks: int = 5, min_freq: float = 20.0
) -> list[tuple[float, float]]:
    """Renvoie les `num_peaks` pics de fréquence les plus marqués, triés par amplitude décroissante."""
    mask = freqs >= min_freq
    freqs, magnitude = freqs[mask], magnitude[mask]
    if magnitude.size == 0 or magnitude.max() == 0:
        return []

    peak_idx, _ = find_peaks(magnitude, height=magnitude.max() * 0.05)
    top = sorted(peak_idx, key=lambda i: magnitude[i], reverse=True)[:num_peaks]
    return sorted(((freqs[i], magnitude[i]) for i in top), key=lambda p: p[1], reverse=True)


def compute_spectrogram(
    signal: np.ndarray, sample_rate: int, nperseg: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renvoie (fréquences, temps, magnitude en dB) pour affichage en spectrogramme."""
    nperseg = min(nperseg, len(signal))
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db
