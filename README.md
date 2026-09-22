# Sonoscope

Logiciel pour enregistrer le son d'un produit (électroménager, moteur...) et le décomposer : forme d'onde, spectre de fréquences (FFT), spectrogramme et fréquences dominantes.

## Objectif

Étudier le bruit émis par un produit pour repérer, par exemple, un bruit aigu inhabituel apparaissant au fil du temps (usure, dérive de fréquence par rapport à un enregistrement de référence).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur Linux, l'enregistrement micro (`sounddevice`) nécessite la bibliothèque système `portaudio` :

```bash
sudo apt install libportaudio2
```

## Lancer l'application

```bash
python app.py
```

Interface de bureau Tkinter (pas de serveur web, pas de popup pare-feu/réseau privé).

## Fonctionnalités (v1)

- Enregistrement depuis le micro ou import d'un fichier audio (wav, flac, ogg, mp3)
- Visualisation : forme d'onde, spectre FFT, spectrogramme
- Détection des fréquences dominantes
- Sauvegarde des enregistrements par produit (`data/recordings/<produit>/<horodatage>.wav`), en vue d'un suivi dans le temps

## Pistes pour la suite

- Comparaison automatique entre un enregistrement "neuf" et un enregistrement "actuel" (écart de fréquence)
- Historique et suivi de la dérive fréquentielle d'un produit dans le temps
- Détection d'anomalie automatique (alerte sur bruit aigu hors plage normale)
