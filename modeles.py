# -*- coding: utf-8 -*-
"""
modeles.py
Utilitaires pour Whisper (base/small/medium en français) + pyannote.audio.
- Entrées supportées : MP3 et WAV (+ URL traité côté app).
- Conversion vers WAV 16 kHz mono via ffmpeg si nécessaire.
- Toutes les fonctions lèvent des erreurs explicitement rédigées en français.
"""

import os
import subprocess
from typing import Tuple, List, Dict, Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def verifier_ffmpeg_disponible() -> None:
    """Vérifie que ffmpeg est accessible dans l'environnement."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        raise RuntimeError("ffmpeg est introuvable. Installez-le ou ajoutez-le au PATH.")


def _est_wav_16k_mono(chemin_wav: str) -> bool:
    """Vérifie qu'un WAV est bien 16 kHz mono."""
    try:
        info = sf.info(chemin_wav)
        return info.samplerate == 16000 and info.channels == 1 and str(info.format).upper().startswith("WAV")
    except Exception:
        return False


def convertir_vers_wav(chemin_entree: str, taux_hz: int = 16000) -> str:
    """
    Convertit un MP3/WAV vers WAV mono {taux_hz}.
    - Si le WAV est déjà mono 16 kHz, renvoie le fichier tel quel.
    - Sinon, ré-échantillonne via ffmpeg.
    """
    if not chemin_entree or not os.path.exists(chemin_entree):
        raise RuntimeError("Fichier introuvable sur le disque.")
    ext = os.path.splitext(chemin_entree)[1].lower()
    if ext not in [".mp3", ".wav"]:
        raise RuntimeError("Format non supporté. Fournissez un fichier .mp3 ou .wav.")

    if ext == ".wav" and _est_wav_16k_mono(chemin_entree):
        return chemin_entree

    verifier_ffmpeg_disponible()
    base, _ = os.path.splitext(chemin_entree)
    chemin_sortie = f"{base}.conv.wav"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", chemin_entree, "-ar", str(taux_hz), "-ac", "1",
        chemin_sortie, "-y",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Échec de conversion via ffmpeg : {e}")
    return chemin_sortie


def lire_wave(chemin_wav: str) -> Tuple[np.ndarray, int]:
    """Lit un WAV en float32 mono. Renvoie (signal, sample_rate)."""
    try:
        data, sr = sf.read(chemin_wav, dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Lecture WAV impossible : {e}")
    if data.ndim > 1:
        data = data.mean(axis=1).astype("float32")
    return data, int(sr)


def charger_modele_whisper(nom_modele: str, appareil: str = "cpu", compute_type: str | None = None):
    """Charge un modèle Faster-Whisper ('base' | 'small' | 'medium')."""
    try:
        ct = compute_type
        if ct is None:
            ct = "int8" if appareil == "cpu" else "float16"
        return WhisperModel(nom_modele, device=appareil, compute_type=ct)
    except Exception as e:
        raise RuntimeError(f"Chargement du modèle Whisper '{nom_modele}' impossible : {e}")


def transcrire_par_fenetres(
    chemin_wav: str,
    modele_whisper,
    langue: str = "fr",
    fenetre_s: float = 120.0,
    chevauchement_s: float = 5.0,
    progress_cb=None,
) -> tuple[List[Dict], str, float]:
    """
    Transcrit l'audio par fenêtres successives pour limiter l’empreinte mémoire.
    Renvoie (segments, texte_total, duree_totale_s).
    """
    try:
        import librosa
    except Exception as e:
        raise RuntimeError(f"Le paquet 'librosa' est manquant : {e}")

    try:
        y, sr = librosa.load(chemin_wav, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Chargement audio pour transcription impossible : {e}")

    duree = float(len(y)) / float(sr) if sr > 0 else 0.0
    segments: List[Dict] = []
    textes: List[str] = []

    t0 = 0.0
    while t0 < duree:
        t1 = min(duree, t0 + fenetre_s)
        i0, i1 = int(t0 * sr), int(t1 * sr)
        morceau = y[i0:i1].astype(np.float32)

        try:
            segments_iter, _ = modele_whisper.transcribe(
                morceau,
                language=langue,
                task="transcribe",
                condition_on_previous_text=False,
                vad_filter=False,
                beam_size=5,
            )
        except Exception as e:
            raise RuntimeError(f"Erreur Whisper pendant la transcription : {e}")

        for s in segments_iter:
            s0 = float(getattr(s, "start", 0.0) or 0.0) + t0
            s1 = float(getattr(s, "end", 0.0) or 0.0) + t0
            txt = (getattr(s, "text", "") or "").strip()
            segments.append({"start": s0, "end": s1, "text": txt})
            if txt:
                textes.append(txt)

        if progress_cb is not None:
            ratio = 1.0 if duree == 0 else (t1 / duree)
            progress_cb(min(max(ratio, 0.0), 1.0))

        if t1 >= duree:
            break
        t0 = t1 - chevauchement_s

    segments.sort(key=lambda s: s["start"])
    return segments, " ".join(textes).strip(), duree


def verifier_acces_hf(jeton_hf: str):
    """Vérifie l’accès Hugging Face (pyannote)."""
    from huggingface_hub import login as hf_login, hf_hub_download
    from huggingface_hub.utils import (
        GatedRepoError, RepositoryNotFoundError, LocalEntryNotFoundError,
        OfflineModeIsEnabled, HFValidationError,
    )
    from requests.exceptions import HTTPError

    if not jeton_hf or not jeton_hf.strip():
        raise RuntimeError("Aucun jeton Hugging Face fourni. Saisissez un jeton valide.")

    try:
        hf_login(token=jeton_hf, add_to_git_credential=False)
        hf_hub_download("pyannote/speaker-diarization-3.1", "README.md", local_files_only=False)
    except GatedRepoError:
        raise RuntimeError("Accès refusé : acceptez les conditions du modèle pyannote/speaker-diarization-3.1.")
    except RepositoryNotFoundError:
        raise RuntimeError("Repo pyannote/speaker-diarization-3.1 introuvable.")
    except LocalEntryNotFoundError:
        raise RuntimeError("Cache Hugging Face inaccessible.")
    except OfflineModeIsEnabled:
        raise RuntimeError("Mode hors-ligne Hugging Face activé.")
    except HFValidationError as e:
        raise RuntimeError(f"Jeton invalide : {e}")
    except HTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", "?")
        if code == 401:
            raise RuntimeError("401 : jeton invalide/expiré.")
        if code == 403:
            raise RuntimeError("403 : conditions non acceptées.")
        raise RuntimeError(f"Erreur HTTP Hugging Face ({code}).")


def charger_pipeline_diarisation(jeton_hf: str):
    """Charge le pipeline pyannote/speaker-diarization-3.1."""
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError(f"Le paquet 'pyannote.audio' est manquant ou invalide : {e}")

    verifier_acces_hf(jeton_hf)
    try:
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as e:
        raise RuntimeError(f"Chargement du pipeline pyannote impossible : {e}")


def diariser_par_fenetres(
    chemin_wav: str,
    pipeline,
    fenetre_s: float = 60.0,
    chevauchement_s: float = 10.0,
    progress_cb=None,
    num_speakers: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[Dict]:
    """
    Diarise par fenêtres avec fusion et options UI :
      - num_speakers : si fourni (>0), on le contraint dans pyannote.
      - threshold : exposé pour compatibilité UI (non utilisé par pyannote).
    Renvoie une liste de dicts {'start','end','speaker'} triée.
    """
    try:
        import librosa
        import torch
    except Exception as e:
        raise RuntimeError(f"Dépendance manquante pour la diarisation : {e}")

    try:
        y, sr = librosa.load(chemin_wav, sr=16000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Chargement audio pour diarisation impossible : {e}")

    duree = float(len(y)) / float(sr) if sr > 0 else 0.0
    segs: List[Dict] = []
    t0 = 0.0
    while t0 < duree:
        t1 = min(duree, t0 + fenetre_s)
        i0, i1 = int(t0 * sr), int(t1 * sr)
        morceau = {"waveform": torch.from_numpy(y[i0:i1]).unsqueeze(0), "sample_rate": sr}
        try:
            if num_speakers and num_speakers > 0:
                diar = pipeline(morceau, num_speakers=int(num_speakers))
            else:
                diar = pipeline(morceau)
        except Exception as e:
            raise RuntimeError(f"Erreur pyannote pendant l'inférence : {e}")

        for (seg, _, spk) in diar.itertracks(yield_label=True):
            segs.append({"start": float(seg.start) + t0, "end": float(seg.end) + t0, "speaker": str(spk)})

        if progress_cb is not None:
            ratio = 1.0 if duree == 0 else (t1 / duree)
            progress_cb(min(max(ratio, 0.0), 1.0))

        if t1 >= duree:
            break
        t0 = t1 - chevauchement_s

    segs.sort(key=lambda x: x["start"])
    fusion: List[Dict] = []
    for s in segs:
        if not fusion:
            fusion.append(s)
            continue
        d = fusion[-1]
        if s["speaker"] == d["speaker"] and s["start"] <= d["end"] + 0.2:
            d["end"] = max(d["end"], s["end"])
        else:
            fusion.append(s)
    return fusion

