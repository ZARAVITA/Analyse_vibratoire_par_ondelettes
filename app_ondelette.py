# -*- coding: utf-8 -*-
"""
Analyse vibratoire par transformée en ondelettes - Remplace la FFT classique
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
import pywt
import requests
from io import BytesIO

# Titre de l'application
st.markdown("""
Cette application effectue l'analyse vibratoire de signaux en utilisant la transformée en ondelettes (Wavelet Transform). Elle permet :
1. **Importation de données** : Chargez un fichier CSV contenant les colonnes "time" et "amplitude".
2. **Visualisation des données** : Affichez les premières lignes du dataset et le signal original.
3. **Traitement BLSD (Bearing Low Speed Detection) du signal** :
   - Filtre passe-haut
   - Redressement du signal
   - Filtre passe-bas
4. **Affichage des résultats** : Visualisez le signal après traitement et son analyse par ondelettes.

Ce projet a été réalisé par **M. A Angelico** et **ZARAVITA** dans le cadre de l'analyse vibratoire.
""")

# Chargement des données des roulements depuis GitHub
@st.cache_data
def load_bearing_data():
    url = "https://github.com/ZARAVITA/analyse_vibratoire_app/raw/main/Bearing%20data%20Base.xlsx"
    try:
        response = requests.get(url)
        response.raise_for_status()
        bearing_data = pd.read_excel(BytesIO(response.content))
        # Nettoyage des données
        bearing_data = bearing_data.dropna(subset=['Manufacturer'])
        bearing_data['Manufacturer'] = bearing_data['Manufacturer'].astype(str).str.strip()
        for col in ['FTF', 'BSF', 'BPFO', 'BPFI']:
            bearing_data[col] = pd.to_numeric(bearing_data[col], errors='coerce')
        bearing_data = bearing_data.dropna(subset=['FTF','BSF','BPFO', 'BPFI'])
        return bearing_data
    except:
        # Données par défaut si le chargement échoue
        default_data = {
            'Manufacturer': ['AMI', 'AMI', 'DODGE', 'DODGE', 'FAFNIR', 'FAFNIR', 'KOYO', 'KOYO', 
                            'SEALMASTER', 'SKF', 'SKF', 'SNR', 'SNR', 'TORRINGTON', 'TORRINGTON'],
            'Name': ['201', '202', 'P2B5__USAF115TTAH (B)', 'P2B5__USAF115TTAH (C)', '206NPP', 
                    '206NPPA1849', '7304B (B)', '7304B (C)', '204', '214 (A)', '205 (A)', 
                    '6316ZZ (B)', 'NU324', '23172B', '23172BW33C08BR'],
            'Number of Rollers': [8, 8, 18, 17, 9, 9, 9, 9, 21, 15, 13, 8, 13, 22, 22],
            'FTF': [0.383, 0.383, 0.42, 0.57, 0.39, 0.39, 0.38, 0.38, 0.4404, 0.41, 0.42, 
                   0.38, 0.4, 0.44, 0.44],
            'BSF': [2.025, 2.025, 3.22, 6.49, 2.31, 2.31, 1.79, 1.79, 7.296, 2.7, 2.36, 
                   2.07, 2.42, 4.16, 4.16],
            'BPFO': [3.066, 3.066, 7.65, 7.24, 3.56, 3.56, 3.47, 3.46, 9.2496, 6.15, 5.47, 
                    3.08, 5.21, 9.71, 9.71],
            'BPFI': [4.934, 4.934, 10.34, 9.75, 5.43, 5.43, 5.53, 5.53, 11.7504, 8.84, 7.52, 
                    4.91, 7.78, 12.28, 12.28]
        }
        return pd.DataFrame(default_data)

bearing_data = load_bearing_data()

# Sidebar - Sélection du roulement
st.sidebar.header("Paramètres du roulement")

# Sélection du fabricant
manufacturers = sorted(bearing_data['Manufacturer'].unique())
selected_manufacturer = st.sidebar.selectbox("Fabricant", manufacturers)

# Filtrage des modèles
models = bearing_data[bearing_data['Manufacturer'] == selected_manufacturer]['Name'].unique()
selected_model = st.sidebar.selectbox("Modèle", models)

# Nombre de rouleaux
selected_roller_count = bearing_data[
    (bearing_data['Manufacturer'] == selected_manufacturer) & 
    (bearing_data['Name'] == selected_model)
]['Number of Rollers'].values[0]

st.sidebar.info(f"Nombre de rouleaux: {selected_roller_count}")

# Vitesse de rotation en Hz
rotation_speed_hz = st.sidebar.number_input("Vitesse de rotation (Hz)", 
                                         min_value=0.1, 
                                         max_value=1000.0, 
                                         value=16.67,
                                         step=0.1,
                                         format="%.2f")

# Calcul des fréquences caractéristiques
selected_bearing = bearing_data[
    (bearing_data['Manufacturer'] == selected_manufacturer) & 
    (bearing_data['Name'] == selected_model)
].iloc[0]

ftf_freq = selected_bearing['FTF'] * rotation_speed_hz
bsf_freq = selected_bearing['BSF'] * rotation_speed_hz
bpfo_freq = selected_bearing['BPFO'] * rotation_speed_hz
bpfi_freq = selected_bearing['BPFI'] * rotation_speed_hz

# Affichage des fréquences calculées
st.sidebar.subheader("Fréquences caractéristiques")
st.sidebar.write(f"- FTF: {ftf_freq:.2f} Hz")
st.sidebar.write(f"- BSF: {bsf_freq:.2f} Hz")
st.sidebar.write(f"- BPFO: {bpfo_freq:.2f} Hz")
st.sidebar.write(f"- BPFI: {bpfi_freq:.2f} Hz")

# Options d'affichage des fréquences
st.sidebar.subheader("Options d'affichage")
show_ftf = st.sidebar.checkbox("Afficher FTF", False)
show_bsf = st.sidebar.checkbox("Afficher BSF", False)
show_bpfo = st.sidebar.checkbox("Afficher BPFO", False)
show_bpfi = st.sidebar.checkbox("Afficher BPFI", False)
show_harmonics = st.sidebar.checkbox("Afficher les harmoniques", False)
harmonics_count = st.sidebar.slider("Nombre d'harmoniques à afficher", 1, 5, 3) if show_harmonics else 0

# Paramètres des ondelettes
st.sidebar.subheader("Paramètres des ondelettes")
wavelet_type = st.sidebar.selectbox(
    "Type d'ondelette",
    ['cmor', 'morl', 'cgau', 'gaus', 'mexh', 'fbsp'],
    index=1
)

if wavelet_type == 'fbsp':
    fbsp_params = st.sidebar.slider(
        "Paramètres FBSP (m, fbsp_b, fbsp_c)",
        min_value=1,
        max_value=5,
        value=(1, 1, 1)
    )
else:
    fbsp_params = None

scale_min = st.sidebar.number_input("Échelle minimale", min_value=1, value=1)
scale_max = st.sidebar.number_input("Échelle maximale", min_value=2, value=128)
scale_step = st.sidebar.number_input("Pas d'échelle", min_value=1, value=2)

# Upload du fichier CSV
uploaded_file = st.file_uploader("Importez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lecture du fichier CSV
    data = pd.read_csv(uploaded_file, sep=";", skiprows=1)
    time = data.iloc[:, 0].values / 1000  # Conversion en secondes
    amplitude = data.iloc[:, 1].values

    # Aperçu du dataset
    if st.checkbox("Afficher les 5 premières lignes du dataset"):
        st.write(data.head())

    # Fréquence d'échantillonnage
    dt = np.diff(time)
    fs = 1 / np.mean(dt)
    st.write(f"Fréquence d'échantillonnage : {fs:.0f} Hz")

    # Fonction pour calculer la transformée en ondelettes continue (CWT)
    def calculate_cwt(signal, scales, wavelet, sampling_period=1.0/fs):
        if wavelet == 'fbsp' and fbsp_params is not None:
            wavelet = f"{wavelet}{fbsp_params[0]}-{fbsp_params[1]}-{fbsp_params[2]}"
        
        coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period)
        return coeffs, freqs

    # Variables globales pour stocker les points sélectionnés
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = {'t1': None, 't2': None, 'fig': None}

    # Fonction pour créer un graphique avec sélection de points
    def create_signal_figure(x, y, title, x_title, y_title):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Signal',
            hovertemplate='Time: %{x:.3f} s<br>Amplitude: %{y:.1f}<extra></extra>'
        ))
        
        if st.session_state.selected_points['fig'] == title:
            t1 = st.session_state.selected_points['t1']
            t2 = st.session_state.selected_points['t2']
            
            if t1 is not None:
                idx1 = np.abs(x - t1).argmin()
                y1 = y[idx1]
                
                fig.add_trace(go.Scatter(
                    x=[t1],
                    y=[y1],
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text='T1',
                    textposition='top center',
                    name='Point 1',
                    hoverinfo='none'
                ))
                
                fig.add_shape(
                    type='line',
                    x0=t1, y0=min(y), x1=t1, y1=max(y),
                    line=dict(color='red', width=2, dash='dot')
                )
                
            if t2 is not None:
                idx2 = np.abs(x - t2).argmin()
                y2 = y[idx2]
                
                fig.add_trace(go.Scatter(
                    x=[t2],
                    y=[y2],
                    mode='markers+text',
                    marker=dict(size=12, color='blue'),
                    text='T2',
                    textposition='top center',
                    name='Point 2',
                    hoverinfo='none'
                ))
                
                fig.add_shape(
                    type='line',
                    x0=t2, y0=min(y), x1=t2, y1=max(y),
                    line=dict(color='blue', width=2, dash='dot')
                )
            
            if t1 is not None and t2 is not None:
                fig.add_shape(
                    type='line',
                    x0=t1, y0=(y1+y2)/2, x1=t2, y1=(y1+y2)/2,
                    line=dict(color='green', width=2, dash='dash'),
                    name='Intervalle'
                )
                
                delta_t = abs(t2 - t1)
                if delta_t > 0:
                    frequency = 1 / delta_t
                    fig.add_annotation(
                        x=(t1 + t2)/2,
                        y=(y1 + y2)/2,
                        text=f'F = {frequency:.2f} Hz',
                        showarrow=True,
                        arrowhead=1,
                        font=dict(size=14, color='green'),
                        bgcolor='white',
                        bordercolor='green',
                        borderwidth=1
                    )
        
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            hovermode='x unified',
            clickmode='event+select',
            dragmode='zoom'
        )
        
        return fig

    # Affichage du signal original avec sélection de points
    if st.checkbox("Afficher le signal original (time vs amplitude)"):
        fig_original = create_signal_figure(
            time, amplitude, 
            'Signal Original', 
            'Time (s)', 'Amplitude'
        )
        
        st.plotly_chart(
            fig_original, 
            config={'displayModeBar': True, 'scrollZoom': True},
            key='original_signal'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            t1 = st.number_input("Sélectionnez T1 (s)", min_value=float(time.min()), 
                                max_value=float(time.max()), value=0.0, step=0.001)
        with col2:
            t2 = st.number_input("Sélectionnez T2 (s)", min_value=float(time.min()), 
                                max_value=float(time.max()), value=0.0, step=0.001)
        
        if t1 != 0 or t2 != 0:
            st.session_state.selected_points = {'t1': t1, 't2': t2, 'fig': 'Signal Original'}
            if t1 != 0 and t2 != 0 and t1 != t2:
                delta_t = abs(t2 - t1)
                frequency = 1 / delta_t
                st.success(f"Fréquence calculée: {frequency:.2f} Hz")

    # Filtre passe-haut
    freq_coupure_haut = st.sidebar.slider("Fréquence de coupure passe-haut (Hz)", 1, 5000, 500)
    def filtre_pass_haut(data, freq_coupure, freq_echantillonnage):
        freq_nyquist = 0.5 * freq_echantillonnage
        freq_normalisee = freq_coupure / freq_nyquist
        b, a = butter(4, freq_normalisee, btype='high', analog=False)
        return filtfilt(b, a, data)
    
    filtre_haut = filtre_pass_haut(amplitude, freq_coupure_haut, fs)
    
    # Redressement
    signal_redresse = np.abs(filtre_haut)
    
    # Filtre passe-bas
    freq_coupure_bas = st.sidebar.slider("Fréquence de coupure passe-bas (Hz)", 1, 1000, 200)
    def filtre_passe_bas(data, freq_coupure, freq_echantillonnage):
        freq_nyquist = 0.5 * freq_echantillonnage
        freq_normalisee = freq_coupure / freq_nyquist
        b, a = butter(4, freq_normalisee, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    signal_filtre = filtre_passe_bas(signal_redresse, freq_coupure_bas, fs)
    
    # Affichage du signal après traitement
    if st.checkbox("Afficher le signal après traitement BLSD(Bearing Low Speed Detection)"):
        fig_treated = create_signal_figure(
            time, signal_filtre, 
            'Signal après traitement (passe-haut, redressement, passe-bas)', 
            'Time (s)', 'Amplitude'
        )
        
        st.plotly_chart(
            fig_treated, 
            config={'displayModeBar': True, 'scrollZoom': True},
            key='treated_signal'
        )
        
        col1, col2 = st.columns(2)
        with col1:
            t1_treated = st.number_input("Sélectionnez T1 (s) - Signal traité", 
                                       min_value=float(time.min()), 
                                       max_value=float(time.max()), 
                                       value=0.0, step=0.001)
        with col2:
            t2_treated = st.number_input("Sélectionnez T2 (s) - Signal traité", 
                                       min_value=float(time.min()), 
                                       max_value=float(time.max()), 
                                       value=0.0, step=0.001)
        
        if t1_treated != 0 or t2_treated != 0:
            st.session_state.selected_points = {'t1': t1_treated, 't2': t2_treated, 'fig': 'Signal traité'}
            if t1_treated != 0 and t2_treated != 0 and t1_treated != t2_treated:
                delta_t = abs(t2_treated - t1_treated)
                frequency = 1 / delta_t
                st.success(f"Fréquence calculée: {frequency:.2f} Hz")

    # Analyse par ondelettes
    if st.checkbox("Effectuer l'analyse par ondelettes"):
        st.subheader("Transformée en ondelettes continues (CWT)")
        
        # Calcul des échelles pour la CWT
        scales = np.arange(scale_min, scale_max, scale_step)
        
        # Calcul de la CWT
        coeffs, freqs = calculate_cwt(signal_filtre, scales, wavelet_type)
        
        # Affichage du scalogramme
        fig = go.Figure()
        
        # Création du heatmap pour le scalogramme
        fig.add_trace(go.Heatmap(
            z=np.abs(coeffs),
            x=time,
            y=freqs,
            colorscale='Jet',
            hoverongaps=False,
            hovertemplate='Time: %{x:.3f} s<br>Fréquence: %{y:.1f} Hz<br>Amplitude: %{z:.2f}<extra></extra>'
        ))
        
        # Ajout des fréquences caractéristiques si demandé
        freq_colors = {
            'FTF': 'violet',
            'BSF': 'green',
            'BPFO': 'blue',
            'BPFI': 'red'
        }
        
        summary_data = []
        
        for freq_type, show, freq in [('FTF', show_ftf, ftf_freq),
                                     ('BSF', show_bsf, bsf_freq),
                                     ('BPFO', show_bpfo, bpfo_freq),
                                     ('BPFI', show_bpfi, bpfi_freq)]:
            if show:
                # Trouver l'index de fréquence le plus proche
                idx = np.abs(freqs - freq).argmin()
                measured_freq = freqs[idx]
                
                # Ajout de la ligne horizontale pour la fréquence
                fig.add_shape(
                    type='line',
                    x0=time[0], y0=measured_freq,
                    x1=time[-1], y1=measured_freq,
                    line=dict(color=freq_colors[freq_type], width=2, dash='dot')
                )
                
                # Ajout d'une annotation
                fig.add_annotation(
                    x=time[-1],
                    y=measured_freq,
                    text=freq_type,
                    showarrow=False,
                    font=dict(color=freq_colors[freq_type], size=12),
                    bgcolor="white",
                    bordercolor=freq_colors[freq_type],
                    borderwidth=1
                )
                
                # Stockage des données pour le tableau
                freq_row = {'Type': freq_type, 'Fréquence (Hz)': f'{measured_freq:.2f}'}
                
                # Ajout des harmoniques si demandé
                if show_harmonics:
                    for i in range(1, harmonics_count + 1):
                        harmonic_freq = i * freq
                        idx_harm = np.abs(freqs - harmonic_freq).argmin()
                        harmonic_freq_measured = freqs[idx_harm]
                        
                        fig.add_shape(
                            type='line',
                            x0=time[0], y0=harmonic_freq_measured,
                            x1=time[-1], y1=harmonic_freq_measured,
                            line=dict(color=freq_colors[freq_type], width=1, dash='dot')
                        
                        fig.add_annotation(
                            x=time[-1],
                            y=harmonic_freq_measured,
                            text=f'{i}×',
                            showarrow=False,
                            font=dict(color=freq_colors[freq_type], size=10),
                            bgcolor="white",
                            bordercolor=freq_colors[freq_type],
                            borderwidth=1
                        )
                        
                        freq_row[f'Harmonique {i} (Hz)'] = f'{harmonic_freq_measured:.2f}'
                
                summary_data.append(freq_row)
        
        # Mise en page du graphique
        fig.update_layout(
            title='Scalogramme - Transformée en ondelettes',
            xaxis_title='Temps (s)',
            yaxis_title='Fréquence (Hz)',
            yaxis_type='log' if st.checkbox('Échelle logarithmique en fréquence') else 'linear',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage du tableau récapitulatif
        if summary_data:
            st.subheader("Tableau récapitulatif des fréquences")
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # Option de téléchargement
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le tableau récapitulatif",
                data=csv,
                file_name='frequences_caracteristiques.csv',
                mime='text/csv'
            )
        
        # Affichage des coefficients d'ondelettes pour une fréquence spécifique
        st.subheader("Analyse à fréquence spécifique")
        selected_freq = st.slider(
            "Sélectionnez une fréquence pour l'analyse (Hz)",
            min_value=float(freqs.min()),
            max_value=float(freqs.max()),
            value=float(ftf_freq) if show_ftf else float(freqs.mean())
        )
        
        # Trouver l'index le plus proche de la fréquence sélectionnée
        idx_freq = np.abs(freqs - selected_freq).argmin()
        coeffs_at_freq = coeffs[idx_freq, :]
        
        # Graphique des coefficients à cette fréquence
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=time,
            y=np.abs(coeffs_at_freq),
            mode='lines',
            name=f'Fréquence {freqs[idx_freq]:.1f} Hz',
            hovertemplate='Time: %{x:.3f} s<br>Amplitude: %{y:.2f}<extra></extra>'
        ))
        
        fig_freq.update_layout(
            title=f'Coefficients d\'ondelettes à {freqs[idx_freq]:.1f} Hz',
            xaxis_title='Temps (s)',
            yaxis_title='Amplitude',
            height=400
        )
        
        st.plotly_chart(fig_freq)
        
        # Calcul de l'enveloppe du signal à cette fréquence
        envelope = np.abs(coeffs_at_freq)
        
        # FFT de l'enveloppe pour détecter les fréquences de modulation
        n = len(envelope)
        fft_envelope = np.fft.fft(envelope)
        freqs_envelope = np.fft.fftfreq(n, d=1/fs)[:n//2]
        fft_magnitude = 2.0/n * np.abs(fft_envelope[0:n//2])
        
        # Graphique de la FFT de l'enveloppe
        fig_envelope_fft = go.Figure()
        fig_envelope_fft.add_trace(go.Scatter(
            x=freqs_envelope,
            y=fft_magnitude,
            mode='lines',
            name='FFT de l\'enveloppe',
            hovertemplate='Fréquence: %{x:.1f} Hz<br>Amplitude: %{y:.2f}<extra></extra>'
        ))
        
        fig_envelope_fft.update_layout(
            title='FFT de l\'enveloppe du signal à la fréquence sélectionnée',
            xaxis_title='Fréquence (Hz)',
            yaxis_title='Amplitude',
            height=400
        )
        
        st.plotly_chart(fig_envelope_fft)
else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")
