# -*- coding: utf-8 -*-
import io
import os
import tempfile
from typing import List, Dict

import numpy as np
import streamlit as st
import soundfile as sf
from faster_whisper import WhisperModel

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
# Transcription Faster-Whisper (modèle configurable) en français
# =====================================================================
@st.cache_resource(show_spinner=False)
def _charger_modele_whisper(modele: str) -> WhisperModel:
    """Charge et met en cache un modèle Faster-Whisper côté serveur."""
    modele = (modele or "").strip() or "distil-large-v2"
    compute_type = "int8"  # adapté au CPU sur les runtimes Streamlit
    return WhisperModel(modele, device="cpu", compute_type=compute_type)


def transcrire_whisper(chemin_audio: str, modele: str, progress_placeholder) -> (List[Dict], str):
    """
    Transcrit l'audio avec Faster-Whisper (modèle configurable depuis l'interface).
    La langue est forcée en français.
    Retourne (segments, texte_global).
    """

    # Indication de progression – début transcription
    bar = progress_placeholder.progress(0.0, text="Transcription 0%")

    # Chargement du modèle (en cache Streamlit)
    model = _charger_modele_whisper(modele)

    # Exécution via Faster-Whisper
    segments_iter, info = model.transcribe(
        chemin_audio,
        language="fr",
        task="transcribe",
        condition_on_previous_text=False,
        vad_filter=False,
        beam_size=5,
    )

    duree = float(getattr(info, "duration", 0.0) or 0.0)
    segments = []
    textes = []

    for seg in segments_iter:
        s0 = float(getattr(seg, "start", 0.0) or 0.0)
        s1 = float(getattr(seg, "end", 0.0) or 0.0)
        txt = (getattr(seg, "text", "") or "").strip()
        segments.append({"start": s0, "end": s1, "text": txt})
        if txt:
            textes.append(txt)

        if duree > 0.0:
            ratio = min(max(s1 / duree, 0.0), 1.0)
            bar.progress(ratio, text=f"Transcription {int(ratio * 100):d}%")

    bar.progress(1.0, text="Transcription 100%")
    texte_global = " ".join(textes).strip()
    return segments, texte_global

# =====================================================================
# Diarisation pyannote 3.1 (jeton via variable d'environnement)
# =====================================================================
def _recuperer_token_hf() -> str:
    """Récupère le jeton Hugging Face depuis l'UI, st.secrets ou l'environnement."""

    # Valeur saisie par l'utilisateur dans l'interface
    try:
        token_ui = st.session_state.get("hf_token_user", "")
    except Exception:
        token_ui = ""
    if token_ui:
        return str(token_ui).strip()

    # Priorité aux secrets Streamlit Cloud
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    candidates = [
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN",
        "HF_API_TOKEN",
    ]

    for cle in candidates:
        if isinstance(secrets, dict) and secrets.get(cle):
            return str(secrets.get(cle, "")).strip()
        try:
            valeur = secrets[cle]  # type: ignore[index]
        except Exception:
            valeur = ""
        if valeur:
            return str(valeur).strip()

    # Fallback : variables d'environnement (utile en local)
    for cle in candidates:
        valeur = os.environ.get(cle, "")
        if valeur:
            return valeur.strip()

    return ""


@st.cache_resource(show_spinner=False)
def _charger_pipeline_diarisation(token: str):
    """Charge le pipeline pyannote en s'authentifiant une seule fois."""

    if not token:
        raise RuntimeError(
            "Aucun jeton Hugging Face trouvé. Ajoutez-le dans st.secrets"
            " (clé HUGGING_FACE_HUB_TOKEN), saisissez-le dans l'interface"
            " ou définissez la variable d'environnement HUGGING_FACE_HUB_TOKEN."
        )

    from huggingface_hub import login as hf_login
    from pyannote.audio import Pipeline as PipelineDiarisation

    # Authentification Hugging Face – silencieuse si déjà connectée
    try:
        hf_login(token=token, add_to_git_credential=False)
    except Exception:
        pass

    return PipelineDiarisation.from_pretrained("pyannote/speaker-diarization-3.1")


def diariser_audio(chemin_audio: str, nb_locuteurs: int, seuil: float, progress_placeholder) -> List[Dict]:
    """
    Réalise la diarisation avec pyannote/audio 3.1.
    Le jeton HF peut être saisi dans l'interface ou fourni via les secrets/variables d'environnement.
    On traite l'audio par fenêtres pour de longs fichiers, puis on fusionne.
    """
    import torch

    hf_token = _recuperer_token_hf()
    pipeline = _charger_pipeline_diarisation(hf_token)

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

        # pyannote>=3.1 attend une valeur numérique ou une fonction de décision pour
        # ``threshold``. Lui passer ``None`` (cas ``nb_locuteurs > 0``) provoque
        # une erreur ``'NoneType' object is not callable`` lorsque le pipeline
        # tente d'utiliser ce seuil. On n'ajoute donc le paramètre ``threshold``
        # que lorsque l'on souhaite laisser pyannote déterminer le nombre de
        # locuteurs automatiquement.
        if nb_locuteurs > 0:
            diar = pipeline(morceau, num_speakers=nb_locuteurs)
        else:
            diar = pipeline(morceau, num_speakers=None, threshold=float(seuil))

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
        "Cette application permet de transcrire un fichier audio en français avec Faster-Whisper "
        "(modèle configurable) et d’identifier les locuteurs avec pyannote.audio. "
        "Vous pouvez indiquer le nombre de locuteurs (ou laisser 0 pour détection automatique) "
        "et ajuster le seuil de regroupement si nécessaire. Les résultats sont téléchargeables en texte."
    )

    # Choix modèle Faster-Whisper
    if "modele_whisper" not in st.session_state:
        st.session_state.modele_whisper = "distil-large-v2"
    modele = st.text_input(
        "Modèle Faster-Whisper",
        value=st.session_state.modele_whisper,
        help=(
            "Indiquez l'identifiant du modèle Faster-Whisper à utiliser (ex. `distil-large-v2`, "
            "`Systran/faster-whisper-small`)."
        ),
    ).strip()
    if modele and modele != st.session_state.modele_whisper:
        st.session_state.modele_whisper = modele
    modele_effectif = modele or st.session_state.get("modele_whisper", "")

    # Saisie du jeton pyannote / Hugging Face
    token_defaut = st.session_state.get("hf_token_user", "")
    token_saisi = st.text_input(
        "Jeton Hugging Face pour pyannote",
        value=token_defaut,
        type="password",
        help=(
            "Obligatoire sur Streamlit Cloud si le jeton n'est pas défini dans *Settings → Secrets*. "
            "Le jeton reste stocké dans la session utilisateur uniquement."
        ),
    ).strip()
    if token_saisi != token_defaut:
        st.session_state.hf_token_user = token_saisi

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

    # Vérification du jeton Hugging Face côté Streamlit Cloud
    if not _recuperer_token_hf():
        st.warning(
            "Aucun jeton Hugging Face détecté. Ajoutez-le dans *Settings → Secrets* (clé `HUGGING_FACE_HUB_TOKEN`), "
            "dans la variable d’environnement correspondante ou saisissez-le ci-dessus."
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
            seg_w, texte_global = transcrire_whisper(chemin, modele_effectif, progress_placeholder=ph_trans)
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
