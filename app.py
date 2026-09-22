"""Sonoscope : enregistrer un son (produit) et le décomposer en fréquences."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sonoscope import analysis, audio_io

st.set_page_config(page_title="Sonoscope", layout="wide")
st.title("Sonoscope")
st.caption("Enregistrer un son de produit et l'analyser : forme d'onde, spectre, fréquences dominantes.")

tab_record, tab_import = st.tabs(["Enregistrer (micro)", "Importer un fichier"])

with tab_record:
    duration = st.slider("Durée d'enregistrement (secondes)", 1, 30, 5)
    if st.button("Enregistrer", type="primary"):
        try:
            with st.spinner(f"Enregistrement en cours ({duration}s)..."):
                data, sr = audio_io.record_from_mic(duration)
            st.session_state["audio"] = (data, sr)
            st.success("Enregistrement terminé.")
        except Exception as exc:  # noqa: BLE001 - erreur micro remontée telle quelle à l'utilisateur
            st.error(f"Impossible d'accéder au micro : {exc}")

with tab_import:
    uploaded = st.file_uploader("Fichier audio", type=["wav", "flac", "ogg", "mp3"])
    if uploaded is not None:
        data, sr = audio_io.load_audio_file(uploaded)
        st.session_state["audio"] = (data, sr)
        st.success(f"Fichier chargé : {uploaded.name} ({sr} Hz, {len(data) / sr:.2f} s)")

if "audio" in st.session_state:
    data, sr = st.session_state["audio"]

    st.divider()
    st.subheader("Analyse")

    col_name, col_save = st.columns([3, 1])
    with col_name:
        product_name = st.text_input("Nom du produit (pour la sauvegarde)", value="")
    with col_save:
        st.write("")
        st.write("")
        if st.button("Sauvegarder l'enregistrement"):
            path = audio_io.save_recording(data, sr, product_name or "produit")
            st.info(f"Sauvegardé : {path}")

    st.audio(data, sample_rate=sr)

    t = np.linspace(0, len(data) / sr, len(data))
    fig_wave = go.Figure(go.Scatter(x=t, y=data, mode="lines", line={"width": 1}))
    fig_wave.update_layout(
        title="Forme d'onde", xaxis_title="Temps (s)", yaxis_title="Amplitude", height=300, margin={"t": 40}
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    freqs, magnitude = analysis.compute_fft(data, sr)
    fig_fft = go.Figure(go.Scatter(x=freqs, y=magnitude, mode="lines"))
    fig_fft.update_layout(
        title="Spectre de fréquence (FFT)",
        xaxis_title="Fréquence (Hz)",
        yaxis_title="Amplitude",
        height=350,
        margin={"t": 40},
    )
    st.plotly_chart(fig_fft, use_container_width=True)

    peaks = analysis.find_dominant_frequencies(freqs, magnitude)
    if peaks:
        st.write("Fréquences dominantes détectées :")
        st.table(
            [{"Fréquence (Hz)": round(f, 1), "Amplitude": round(m, 5)} for f, m in peaks]
        )
    else:
        st.write("Aucun pic de fréquence marqué détecté.")

    f_spec, t_spec, Sxx_db = analysis.compute_spectrogram(data, sr)
    fig_spec = go.Figure(go.Heatmap(z=Sxx_db, x=t_spec, y=f_spec, colorscale="Viridis"))
    fig_spec.update_layout(
        title="Spectrogramme",
        xaxis_title="Temps (s)",
        yaxis_title="Fréquence (Hz)",
        height=400,
        margin={"t": 40},
    )
    st.plotly_chart(fig_spec, use_container_width=True)
else:
    st.info("Enregistrez un son via le micro ou importez un fichier pour lancer l'analyse.")
