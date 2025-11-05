# -*- coding: utf-8 -*-
import io
import os
import tempfile
from typing import List, Dict

import numpy as np
import streamlit as st
import soundfile as sf

# =====================================================================
# Configuration de la page
# =====================================================================
st.set_page_config(page_title="Transcription + Diarisation (Whisper + pyannote)", layout="centered")

# =====================================================================
# Fonctions utilitaires
# =====================================================================
def formater_hms(t: float) -> str:
    """Formate un temps en secondes au format HH:MM:SS."""
    t = float(max(0.0, t))
    s = int(round(t))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def enregistrer_temporaire(fichier) -> str:
    """Enregistre un fichier Streamlit en fichier temporaire et retourne son chemin."""
    suffixe = os.path.splitext(getattr(fichier, "name", ""))[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffixe) as tmp:
        tmp.write(fichier.read())
        return tmp.name

def charger_audio_mono16k(chemin: str, sr_cible: int = 16000):
    """Charge un audio en mono/sr_cible pour pyannote."""
    import librosa
    y, sr = librosa.load(chemin, sr=sr_cible, mono=True)
    return y.astype(np.float32), int(sr)

def fusionner_intervalles(segs: List[Dict], tol: float = 0.2) -> List[Dict]:
    """Fusionne les segments adjacents du même locuteur avec éventuel recouvrement."""
    segs = sorted(segs, key=lambda s: s["start"])
    out = []
    for s in segs:
        if not out:
            out.append(dict(s))
            continue
        d = out[-1]
        if s["speaker"] == d["speaker"] and s["start"] <= d["end"] + tol:
            d["end"] = max(d["end"], s["end"])
        else:
            out.append(dict(s))
    return out

def attribuer_locuteurs(segments_whisper: List[Dict], segments_diar: List[Dict]) -> List[Dict]:
    """Associe à chaque segment Whisper le locuteur dominant selon le recouvrement temporel."""
    def recouv(a0, a1, b0, b1):
        d = max(a0, b0)
        f = min(a1, b1)
        return max(0.0, f - d)
    out = []
    for s in segments_whisper:
        s0, s1 = float(s["start"]), float(s["end"])
        best = "SPEAKER_00"
        score = 0.0
        for d in segments_diar:
            sc = recouv(s0, s1, d["start"], d["end"])
            if sc > score:
                score, best = sc, d["speaker"]
        out.append({"debut": s0, "fin": s1, "locuteur": best, "texte": (s.get("text") or "").strip()})
    return out

def generer_document(segments_attr: List[Dict]) -> str:
    """Génère un document diarisé à partir de segments avec locuteurs."""
    if not segments_attr:
        return ""
    lignes = []
    cur = {
        "locuteur": segments_attr[0]["locuteur"],
        "debut": segments_attr[0]["debut"],
        "texte": segments_attr[0]["texte"],
    }
    for s in segments_attr[1:]:
        if s["locuteur"] == cur["locuteur"]:
            t = s["texte"].strip()
            if t:
                cur["texte"] = (cur["texte"] + " " + t).strip()
        else:
            lignes.append(f"[{formater_hms(cur['debut'])}] {cur['locuteur']} : {cur['texte']}")
            cur = {"locuteur": s["locuteur"], "debut": s["debut"], "texte": s["texte"]}
    lignes.append(f"[{formater_hms(cur['debut'])}] {cur['locuteur']} : {cur['texte']}")
    return "\n\n".join(lignes)

# =====================================================================
# Transcription Whisper (modèles base/small/medium) en français
# =====================================================================
def transcrire_whisper(chemin_audio: str, modele: str, progress_placeholder) -> (List[Dict], str):
    """
    Transcrit l'audio avec openai-whisper (modèles CPU: base/small/medium).
    La langue est forcée en français.
    Retourne (segments, texte_global).
    """
    import whisper

    # Indication de progression – début transcription
    bar = progress_placeholder.progress(0.0, text="Transcription 0%")

    # Chargement du modèle
    # Les modèles "base", "small", "medium" sont multilingues.
    # On force language='fr' pour la transcription en français.
    model = whisper.load_model(modele)

    # Paramètres pour stabilité en FR
    options = dict(
        language="fr",
        task="transcribe",
        condition_on_previous_text=False,
        fp16=False,
        verbose=False,
    )

    # Exécution
    result = model.transcribe(chemin_audio, **options)

    # Extraction segments
    segments = []
    textes = []
    for i, seg in enumerate(result.get("segments", [])):
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", 0.0))
        txt = (seg.get("text") or "").strip()
        segments.append({"start": s0, "end": s1, "text": txt})
        if txt:
            textes.append(txt)
        # Progression approximative
        if len(result.get("segments", [])) > 0:
            bar.progress(min(1.0, (i + 1) / len(result["segments"])), text=f"Transcription {int((i + 1) * 100 / len(result['segments']))}%")

    bar.progress(1.0, text="Transcription 100%")
    texte_global = " ".join(textes).strip()
    return segments, texte_global

# =====================================================================
# Diarisation pyannote 3.1 (jeton via variable d'environnement)
# =====================================================================
def diariser_audio(chemin_audio: str, nb_locuteurs: int, seuil: float, progress_placeholder) -> List[Dict]:
    """
    Réalise la diarisation avec pyannote/audio 3.1.
    Le jeton HF doit être dans la variable d'environnement HUGGING_FACE_HUB_TOKEN.
    On traite l'audio par fenêtres pour de longs fichiers, puis on fusionne.
    """
    import torch
    from huggingface_hub import login as hf_login
    from pyannote.audio import Pipeline as PipelineDiarisation

    # Auth HF (si secret présent)
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if hf_token:
        try:
            hf_login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass  # on ignore si déjà loggé

    # Chargement pipeline sans argument token (incompatible v3.1 si fourni)
    pipeline = PipelineDiarisation.from_pretrained("pyannote/speaker-diarization-3.1")

    # Chargement audio mono/16k
    y, sr = charger_audio_mono16k(chemin_audio, sr_cible=16000)
    duree = len(y) / sr if sr > 0 else 0.0

    # Barre de progression
    bar = progress_placeholder.progress(0.0, text="Diarisation 0%")

    # Découpage par fenêtres glissantes
    fenetre_s, chev_s = 60.0, 10.0
    t0 = 0.0
    bruts = []

    while t0 < duree:
        t1 = min(duree, t0 + fenetre_s)
        i0, i1 = int(t0 * sr), int(t1 * sr)
        morceau = {"waveform": torch.from_numpy(y[i0:i1]).unsqueeze(0), "sample_rate": sr}

        diar = pipeline(morceau, num_speakers=None if nb_locuteurs <= 0 else nb_locuteurs, threshold=None if nb_locuteurs > 0 else float(seuil))

        for (seg, _, spk) in diar.itertracks(yield_label=True):
            bruts.append({"start": float(seg.start) + t0, "end": float(seg.end) + t0, "speaker": str(spk)})

        ratio = 1.0 if duree == 0 else t1 / duree
        bar.progress(ratio, text=f"Diarisation {int(ratio * 100):d}%")

        if t1 >= duree:
            break
        t0 = t1 - chev_s

    bar.progress(1.0, text="Diarisation 100%")

    bruts.sort(key=lambda s: s["start"])
    fusion = fusionner_intervalles(bruts, tol=0.2)
    return fusion

# =====================================================================
# Interface Streamlit
# =====================================================================
def interface():
    # Titre et lien
    st.title("Transcription + Diarisation (Whisper + pyannote)")
    st.markdown('[www.codeandcortex.fr](https://www.codeandcortex.fr)')

    # Phrase explicative simple (FR)
    st.markdown(
        "Cette application permet de transcrire un fichier audio en français avec Whisper "
        "(modèles base, small, medium) et d’identifier les locuteurs avec pyannote.audio. "
        "Vous pouvez indiquer le nombre de locuteurs (ou laisser 0 pour détection automatique) "
        "et ajuster le seuil de regroupement si nécessaire. Les résultats sont téléchargeables en texte."
    )

    # Choix modèle Whisper
    modele = st.selectbox(
        "Modèle Whisper",
        options=["base", "small", "medium"],
        index=1,
        help="Choisir la taille du modèle de transcription (langue forcée en français)."
    )

    # Paramètres diarisation
    col1, col2 = st.columns(2)
    with col1:
        nb_locuteurs_str = st.text_input(
            "Nombre de locuteurs",
            value="0",
            help="Mettre 0 si inconnu (clustering automatique)."
        )
    with col2:
        seuil_str = st.text_input(
            "Seuil de clustering",
            value="0.5",
            help="Utilisé si le nombre de locuteurs vaut 0. Plus grand → moins de locuteurs."
        )

    # Import fichier
    fichier = st.file_uploader(
        "Importer un fichier audio",
        type=["wav", "mp3"],
        accept_multiple_files=False,
        help="Formats pris en charge : WAV, MP3."
    )

    # Bouton lancer
    lancer = st.button("Lancer le traitement")

    # Emplacements UI
    ph_trans = st.empty()
    ph_diar = st.empty()
    ph_fin = st.empty()

    # État pour téléchargements persistants
    if "doc_bytes" not in st.session_state:
        st.session_state.doc_bytes = None
    if "trans_bytes" not in st.session_state:
        st.session_state.trans_bytes = None

    # Boutons de téléchargement (affichés si dispo)
    if st.session_state.trans_bytes:
        st.download_button(
            "Télécharger la transcription (.txt)",
            data=st.session_state.trans_bytes,
            file_name="transcription.txt",
            mime="text/plain",
            key="dl_trans"
        )
    if st.session_state.doc_bytes:
        st.download_button(
            "Télécharger le document diarisé (.txt)",
            data=st.session_state.doc_bytes,
            file_name="document_diarise.txt",
            mime="text/plain",
            key="dl_doc"
        )

    # Traitement
    if lancer:
        if fichier is None:
            st.error("Veuillez d’abord importer un fichier audio.")
            return

        # Parsing des champs numériques
        try:
            nb_locuteurs = int(nb_locuteurs_str)
        except Exception:
            st.error("Le nombre de locuteurs doit être un entier (0 autorisé).")
            return
        try:
            seuil = float(seuil_str)
            if nb_locuteurs <= 0 and not (0.0 < seuil < 10.0):
                st.error("Le seuil de clustering doit être compris entre 0 et 10 lorsque le nombre de locuteurs vaut 0.")
                return
        except Exception:
            st.error("Le seuil de clustering doit être un nombre réel.")
            return

        # Enregistrement en temporaire
        try:
            chemin = enregistrer_temporaire(fichier)
        except Exception as e:
            st.error(f"Erreur d’enregistrement du fichier : {e}")
            return

        # Transcription
        try:
            seg_w, texte_global = transcrire_whisper(chemin, modele, progress_placeholder=ph_trans)
        except Exception as e:
            st.error(f"Erreur pendant la transcription : {e}")
            return

        # Diarisation
        try:
            seg_d = diariser_audio(chemin, nb_locuteurs=nb_locuteurs, seuil=seuil, progress_placeholder=ph_diar)
        except Exception as e:
            st.error(f"Erreur pendant la diarisation : {e}")
            return

        # Attribution locuteurs + Document
        try:
            attrib = attribuer_locuteurs(seg_w, seg_d)
            document = generer_document(attrib)
        except Exception as e:
            st.error(f"Erreur pendant la génération du document : {e}")
            return

        # Préparation téléchargements
        try:
            st.session_state.trans_bytes = (texte_global or "").encode("utf-8")
            st.session_state.doc_bytes = (document or "").encode("utf-8")
        except Exception as e:
            st.error(f"Erreur d’encodage pour le téléchargement : {e}")
            return

        ph_fin.success("Traitement terminé. Les fichiers sont prêts au téléchargement ci-dessus.")

        # Réafficher les boutons à la fin si première exécution
        st.download_button(
            "Télécharger la transcription (.txt)",
            data=st.session_state.trans_bytes,
            file_name="transcription.txt",
            mime="text/plain",
            key="dl_trans_end"
        )
        st.download_button(
            "Télécharger le document diarisé (.txt)",
            data=st.session_state.doc_bytes,
            file_name="document_diarise.txt",
            mime="text/plain",
            key="dl_doc_end"
        )

if __name__ == "__main__":
    interface()
