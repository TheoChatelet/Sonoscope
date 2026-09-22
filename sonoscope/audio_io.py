"""Capture, chargement et sauvegarde de sons."""

from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def record_from_mic(duration_sec: float, sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    """Enregistre `duration_sec` secondes depuis le micro par défaut et renvoie (signal mono, sample_rate)."""
    import sounddevice as sd

    audio = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten(), sample_rate


def load_audio_file(file_like_or_path) -> tuple[np.ndarray, int]:
    """Charge un fichier audio (wav/flac/ogg/mp3) et renvoie (signal mono, sample_rate)."""
    try:
        data, sr = sf.read(file_like_or_path, always_2d=False)
    except Exception:
        # Formats non couverts par libsndfile (ex. certains mp3) : repli sur librosa/audioread.
        import librosa

        if hasattr(file_like_or_path, "seek"):
            file_like_or_path.seek(0)
        data, sr = librosa.load(file_like_or_path, sr=None, mono=False)
        data = data.T if data.ndim > 1 else data

    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def save_recording(data: np.ndarray, sample_rate: int, product_name: str, base_dir: str = "data/recordings") -> Path:
    """Sauvegarde un enregistrement en .wav sous data/recordings/<produit>/<horodatage>.wav."""
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in product_name.strip()) or "produit"
    folder = Path(base_dir) / safe_name
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{timestamp}.wav"
    sf.write(path, data, sample_rate)
    return path
