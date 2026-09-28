"""Capture, chargement et sauvegarde de sons."""

import json
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
    except Exception:  # noqa: BLE001 - libsndfile lève des types d'erreur variables selon le format/la version
        # Formats non couverts par libsndfile (ex. certains mp3) : repli sur librosa/audioread.
        import librosa

        if hasattr(file_like_or_path, "seek"):
            file_like_or_path.seek(0)
        data, sr = librosa.load(file_like_or_path, sr=None, mono=False)
        data = data.T if data.ndim > 1 else data

    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def _safe_folder_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "produit"


def _product_folder_name(nom: str, numero: str) -> str:
    """Le dossier d'un produit est dérivé de nom + numéro : deux produits peuvent partager un nom
    tant que leur numéro diffère, sans se marcher dessus."""
    key = f"{nom}_{numero}" if numero else nom
    return _safe_folder_name(key)


def product_exists(nom: str, numero: str, base_dir: str = "data/recordings") -> bool:
    """Vrai si un produit avec ce nom (insensible à la casse) et ce numéro exact est déjà enregistré."""
    nom_norm = nom.strip().lower()
    numero_norm = numero.strip()
    return any(
        product.get("nom", "").strip().lower() == nom_norm and product.get("numero", "").strip() == numero_norm
        for product in list_registered_products(base_dir)
    )


def register_product(
    nom: str,
    numero: str = "",
    specificite: str = "",
    type_test: str = "",
    date_produit: str = "",
    base_dir: str = "data/recordings",
) -> Path:
    """Enregistre un nouveau produit (identifié par nom + numéro) et renvoie son dossier.

    Écrase les informations existantes si le couple nom+numéro (et donc le dossier) existe déjà :
    utilisez update_product() pour modifier un produit sans perdre ses infos, et product_exists()
    pour vérifier l'absence de doublon avant d'appeler cette fonction.
    """
    folder = Path(base_dir) / _product_folder_name(nom, numero)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "nom": nom.strip(),
        "numero": numero.strip(),
        "specificite": specificite.strip(),
        "type_test": type_test,
        "date_produit": date_produit.strip(),
        "date_creation": datetime.now().strftime("%Y-%m-%d %H:%M"),  # noqa: DTZ005 - horodatage local d'affichage
    }
    (folder / "product.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def get_product(folder_name: str, base_dir: str = "data/recordings") -> dict | None:
    """Charge les métadonnées d'un produit enregistré à partir de son nom de dossier."""
    product_path = Path(base_dir) / folder_name / "product.json"
    if not product_path.is_file():
        return None
    metadata = json.loads(product_path.read_text(encoding="utf-8"))
    metadata["folder"] = folder_name
    return metadata


def update_product(
    folder_name: str,
    numero: str = "",
    specificite: str = "",
    type_test: str = "",
    date_produit: str = "",
    base_dir: str = "data/recordings",
) -> Path:
    """Met à jour les infos d'un produit déjà enregistré. Le nom et le dossier ne changent pas."""
    folder = Path(base_dir) / folder_name
    if not folder.is_dir():
        msg = f"Produit introuvable : {folder_name}"
        raise FileNotFoundError(msg)

    existing = get_product(folder_name, base_dir) or {}
    metadata = {
        "nom": existing.get("nom", folder_name),
        "numero": numero.strip(),
        "specificite": specificite.strip(),
        "type_test": type_test,
        "date_produit": date_produit.strip(),
        "date_creation": existing.get("date_creation", ""),
    }
    (folder / "product.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def list_registered_products(base_dir: str = "data/recordings") -> list[dict]:
    """Renvoie les produits enregistrés (avec leur nombre de tests).

    Un dossier sans product.json (créé avant l'ajout de cette fonctionnalité) est tout de même
    listé, avec son nom de dossier en guise de nom et des champs numéro/spécificité vides.
    """
    root = Path(base_dir)
    if not root.is_dir():
        return []

    products = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        product_path = folder / "product.json"
        if product_path.is_file():
            metadata = json.loads(product_path.read_text(encoding="utf-8"))
        else:
            metadata = {
                "nom": folder.name,
                "numero": "",
                "specificite": "",
                "type_test": "",
                "date_produit": "",
                "date_creation": "",
            }
        metadata["folder"] = folder.name
        metadata["nb_tests"] = sum(1 for _ in folder.glob("*.wav"))
        products.append(metadata)
    return products


def save_recording(
    data: np.ndarray,
    sample_rate: int,
    folder_name: str,
    base_dir: str = "data/recordings",
    cas: str = "",
    remarques: str = "",
    dominant_frequencies: list[tuple[float, float]] | None = None,
) -> Path:
    """Sauvegarde un enregistrement (.wav) et ses métadonnées (.json) sous data/recordings/<produit>/<horodatage>.*.

    `folder_name` est le nom de dossier du produit (celui renvoyé par register_product ou trouvé
    dans list_registered_products), pas forcément son nom affiché : plusieurs produits peuvent
    partager un nom tant que leur numéro diffère.
    """
    folder = Path(base_dir) / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005 - horodatage local pour un nom de fichier
    wav_path = folder / f"{timestamp}.wav"
    sf.write(wav_path, data, sample_rate)

    product = get_product(folder_name, base_dir)
    metadata = {
        "produit": (product or {}).get("nom", folder_name),
        "cas": cas,
        "remarques": remarques,
        "duree_sec": round(len(data) / sample_rate, 2),
        "sample_rate": sample_rate,
        "frequences_dominantes": [[round(freq, 1), round(mag, 6)] for freq, mag in (dominant_frequencies or [])],
    }
    wav_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav_path


def list_products(base_dir: str = "data/recordings") -> list[str]:
    """Renvoie les noms de produits pour lesquels des enregistrements existent."""
    root = Path(base_dir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_recordings(folder_name: str, base_dir: str = "data/recordings") -> list[dict]:
    """Renvoie les enregistrements sauvegardés d'un produit (par nom de dossier), triés par date.

    Chaque enregistrement contient au minimum : path, horodatage, cas, remarques, duree_sec,
    sample_rate, frequences_dominantes. Les .wav sauvegardés avant l'ajout des métadonnées (pas
    de .json associé) sont tout de même listés, avec des métadonnées minimales.
    """
    folder = Path(base_dir) / folder_name
    if not folder.is_dir():
        return []

    recordings = []
    for wav_path in sorted(folder.glob("*.wav")):
        json_path = wav_path.with_suffix(".json")
        if json_path.is_file():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            info = sf.info(wav_path)
            metadata = {
                "produit": folder_name,
                "cas": "",
                "remarques": "",
                "duree_sec": round(info.frames / info.samplerate, 2),
                "sample_rate": info.samplerate,
                "frequences_dominantes": [],
            }
        metadata["horodatage"] = wav_path.stem
        metadata["path"] = wav_path
        recordings.append(metadata)
    return recordings
