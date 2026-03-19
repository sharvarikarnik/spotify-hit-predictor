"""
app.py — Spotify Hit Predictor Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hit or Flop? · Spotify Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — refined Spotify aesthetic ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #080C0A;
    color: #E8EDE9;
    font-size: 15px;
    line-height: 1.7;
}
.stApp { background: #080C0A; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 3.5rem 5rem 3.5rem;
    max-width: 1280px;
}

/* ── Hero section ── */
.hero {
    background: linear-gradient(160deg, #080C0A 0%, #0C1510 60%, #091209 100%);
    border-bottom: 1px solid #1C2B1F;
    padding: 4rem 0 3rem 0;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; left: -80px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(29,185,84,0.06) 0%, transparent 65%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -100px; right: -60px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(29,185,84,0.03) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 4.2rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -2px;
    background: linear-gradient(100deg, #1DB954 0%, #4ade80 45%, #d1fae5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.15rem;
    color: #6B8A74;
    margin-top: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.01em;
    line-height: 1.8;
}
.hero-badge {
    display: inline-block;
    background: rgba(29,185,84,0.08);
    border: 1px solid rgba(29,185,84,0.2);
    color: #4ade80;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1C2B1F;
}

/* ── Metric cards ── */
.metric-card {
    background: #0D1610;
    border: 1px solid #1C2B1F;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    transition: border-color 0.25s, background 0.25s;
}
.metric-card:hover {
    border-color: rgba(29,185,84,0.4);
    background: #0F1C13;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #1DB954;
    line-height: 1;
    letter-spacing: -1px;
}
.metric-label {
    font-size: 0.75rem;
    color: #4A6B54;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* ── Result card ── */
.result-hit {
    background: linear-gradient(145deg, #0D1F14 0%, #0F2517 100%);
    border: 1px solid rgba(29,185,84,0.35);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.result-flop {
    background: linear-gradient(145deg, #1A1215 0%, #1E1318 100%);
    border: 1px solid rgba(220,100,100,0.25);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.result-score {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -3px;
}
.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 0.7rem;
    letter-spacing: 0.06em;
}
.result-sub {
    font-size: 0.85rem;
    color: #4A6B54;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Sliders ── */
.stSlider > div > div > div > div {
    background: #1DB954 !important;
}
.stSlider label {
    color: #8AAE96 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em;
    font-weight: 500 !important;
}
.stSlider [data-testid="stTickBar"] { display: none; }

/* ── Selectbox ── */
.stSelectbox label {
    color: #8AAE96 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.stSelectbox > div > div {
    background: #0D1610 !important;
    border: 1px solid #1C2B1F !important;
    color: #E8EDE9 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}

/* ── Divider ── */
hr { border-color: #1C2B1F !important; margin: 2.5rem 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1C2B1F;
    gap: 0.5rem;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4A6B54 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.9rem 1.6rem !important;
    border-radius: 0 !important;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #1DB954 !important;
    border-bottom: 2px solid #1DB954 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #4ade80 !important;
}

/* ── Info boxes ── */
.insight-box {
    background: #0D1610;
    border: 1px solid #1C2B1F;
    border-left: 3px solid #1DB954;
    padding: 1.1rem 1.5rem;
    margin: 0.8rem 0;
    border-radius: 0 10px 10px 0;
    font-size: 0.92rem;
    color: #8AAE96;
    line-height: 1.75;
}
.insight-box strong { color: #E8EDE9; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'outputs/models/xgboost_hit_predictor.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    path = 'data/master_dataset.csv'
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

bundle = load_model()
df     = load_data()

FEATURES = [
    'danceability','energy','valence','tempo','loudness',
    'speechiness','acousticness','instrumentalness','liveness',
    'duration_ms','chorus_hit','sections','key','mode',
    'time_signature','decade',
]

# ── Plotly dark theme ──────────────────────────────────────────────────────
PLOT_BG   = '#080C0A'
PAPER_BG  = '#080C0A'
GRID_COL  = '#1C2B1F'
TEXT_COL  = '#4A6B54'
GREEN     = '#1DB954'
ORANGE    = '#4ade80'

def dark_layout(fig, title=None, height=380):
    fig.update_layout(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family='Plus Jakarta Sans', color=TEXT_COL, size=12),
        title=dict(text=title, font=dict(family='Syne', size=13,
                   color='#e8e6e0'), x=0, pad=dict(l=0, t=0)) if title else {},
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        height=height,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                     tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                     tickfont=dict(size=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">Machine Learning &nbsp;·&nbsp; Audio Analysis &nbsp;·&nbsp; Billboard Hot 100</div>
    <h1 class="hero-title">Hit or Flop?</h1>
    <p class="hero-sub">
        Six decades of Billboard data &nbsp;·&nbsp; XGBoost ML model &nbsp;·&nbsp; 16 Spotify audio features<br>
        Adjust the sliders to predict whether a song will chart — before it drops.
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Predict a Song",
    "📊  Data Insights",
    "🤖  Model Performance",
    "📅  Decade Explorer",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — PREDICTOR
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    if bundle is None:
        st.error("⚠️  Model not found. Run `03_modelling.ipynb` first to train and save the model.")
    else:
        model = bundle['model']
        meta  = bundle.get('metadata', {})

        col_controls, col_spacer, col_result = st.columns([2, 0.15, 1.4])

        with col_controls:
            st.markdown('<div class="section-label">Audio Features</div>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                danceability = st.slider("Danceability",     0.0, 1.0, 0.65, 0.01)
                energy       = st.slider("Energy",           0.0, 1.0, 0.72, 0.01)
                valence      = st.slider("Valence (mood)",   0.0, 1.0, 0.55, 0.01)
                acousticness = st.slider("Acousticness",     0.0, 1.0, 0.10, 0.01)
                speechiness  = st.slider("Speechiness",      0.0, 1.0, 0.05, 0.01)
                liveness     = st.slider("Liveness",         0.0, 1.0, 0.12, 0.01)
                instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)

            with c2:
                tempo        = st.slider("Tempo (BPM)",      60,  200, 120, 1)
                loudness     = st.slider("Loudness (dB)",    -40, 0,   -6,  1)
                duration_sec = st.slider("Duration (secs)",  60,  360, 210, 5)
                chorus_hit   = st.slider("Chorus hit (sec)", 0,   90,  22,  1)
                sections     = st.slider("Sections",         3,   20,  9,   1)
                key          = st.selectbox("Key", options=list(range(12)),
                                format_func=lambda x: ['C','C#','D','D#','E','F',
                                                        'F#','G','G#','A','A#','B'][x])
                mode         = st.selectbox("Mode", [1, 0],
                                format_func=lambda x: 'Major' if x == 1 else 'Minor')

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Song Context</div>',
                        unsafe_allow_html=True)

            c3, c4 = st.columns(2)
            with c3:
                decade = st.selectbox("Decade",
                    [1960,1970,1980,1990,2000,2010],
                    index=5,
                    format_func=lambda x: f"{x}s")
            with c4:
                time_sig = st.selectbox("Time Signature",
                    [3, 4, 5, 6, 7], index=1,
                    format_func=lambda x: f"{x}/4")

            st.markdown("""
            <div style="margin-top:1.5rem; padding:0.85rem 1.3rem;
                 background:rgba(29,185,84,0.06);
                 border:1px solid rgba(29,185,84,0.18);
                 border-radius:100px; font-size:0.82rem; color:#4ade80;
                 letter-spacing:0.06em; text-align:center;
                 font-weight:500;">
                ⚡&nbsp;&nbsp;Prediction updates live as you move the sliders
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            st.markdown('<div class="section-label">Prediction</div>',
                        unsafe_allow_html=True)

            # Build input
            input_data = pd.DataFrame([{
                'danceability':     danceability,
                'energy':           energy,
                'valence':          valence,
                'tempo':            float(tempo),
                'loudness':         float(loudness),
                'speechiness':      speechiness,
                'acousticness':     acousticness,
                'instrumentalness': instrumentalness,
                'liveness':         liveness,
                'duration_ms':      duration_sec * 1000,
                'chorus_hit':       float(chorus_hit),
                'sections':         float(sections),
                'key':              float(key),
                'mode':             float(mode),
                'time_signature':   float(time_sig),
                'decade':           float(decade),
            }])

            hit_prob   = model.predict_proba(input_data)[0][1]
            is_hit     = hit_prob >= 0.5
            score_pct  = int(hit_prob * 100)

            card_class = "result-hit" if is_hit else "result-flop"
            score_color = "#1DB954" if is_hit else "#c45a5a"
            verdict     = "LIKELY HIT 🎉" if is_hit else "LIKELY FLOP 😬"

            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-score" style="color:{score_color}">{score_pct}%</div>
                <div class="result-verdict" style="color:{score_color}">{verdict}</div>
                <div class="result-sub">hit probability score</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge chart ────────────────────────────────────────────────
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hit_prob * 100,
                number={'suffix': '%', 'font': {'size': 28,
                        'family': 'Syne', 'color': score_color}},
                gauge={
                    'axis': {'range': [0, 100],
                             'tickcolor': GRID_COL,
                             'tickfont': {'size': 9, 'color': TEXT_COL}},
                    'bar':  {'color': score_color, 'thickness': 0.3},
                    'bgcolor': '#0D1610',
                    'bordercolor': '#1C2B1F',
                    'steps': [
                        {'range': [0, 30],  'color': '#090E0B'},
                        {'range': [30, 50], 'color': '#0B120D'},
                        {'range': [50, 70], 'color': '#0C1810'},
                        {'range': [70, 100],'color': '#0E1F13'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 2},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor=PAPER_BG,
                font=dict(family='DM Sans', color=TEXT_COL),
                margin=dict(l=10, r=10, t=10, b=10),
                height=220,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Radar chart of this song's features ────────────────────────
            st.markdown('<div class="section-label" style="margin-top:1rem">Feature Profile</div>',
                        unsafe_allow_html=True)

            radar_feats  = ['danceability','energy','valence',
                            'acousticness','liveness','speechiness']
            radar_vals   = [danceability, energy, valence,
                            acousticness, liveness, speechiness]

            # Compare to average hit
            avg_hit_vals = []
            if df is not None:
                hits_df = df[df['is_hit'] == 1]
                avg_hit_vals = [hits_df[f].mean() for f in radar_feats]
            else:
                avg_hit_vals = [0.65, 0.73, 0.55, 0.18, 0.19, 0.07]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_feats + [radar_feats[0]],
                fill='toself',
                fillcolor=f'rgba(29,185,84,0.12)',
                line=dict(color=GREEN, width=2),
                name='Your song',
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_hit_vals + [avg_hit_vals[0]],
                theta=radar_feats + [radar_feats[0]],
                fill='toself',
                fillcolor='rgba(255,107,53,0.06)',
                line=dict(color=ORANGE, width=1.5, dash='dot'),
                name='Avg hit',
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='#111118',
                    radialaxis=dict(visible=True, range=[0,1],
                                   gridcolor=GRID_COL,
                                   tickfont=dict(size=8, color=TEXT_COL)),
                    angularaxis=dict(gridcolor=GRID_COL,
                                    tickfont=dict(size=10, color='#aaa')),
                ),
                paper_bgcolor=PAPER_BG,
                font=dict(family='DM Sans', color=TEXT_COL),
                showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                            orientation='h', y=-0.1),
                margin=dict(l=20, r=20, t=10, b=30),
                height=310,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA INSIGHTS
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.error("⚠️  Dataset not found. Make sure `data/master_dataset.csv` exists.")
    else:
        hits_df  = df[df['is_hit'] == 1]
        flops_df = df[df['is_hit'] == 0]

        # ── KPI row ────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        for col, val, label in [
            (k1, f"{len(df):,}",               "Total Songs"),
            (k2, f"{df['is_hit'].sum():,}",     "Billboard Hits"),
            (k3, f"{df['decade'].nunique()}",   "Decades Covered"),
            (k4, f"{len(FEATURES)}",            "Audio Features"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature means comparison ───────────────────────────────────────
        st.markdown('<div class="section-label">Feature Comparison — Hits vs Flops</div>',
                    unsafe_allow_html=True)

        norm_feats = ['danceability','energy','valence',
                      'acousticness','speechiness','instrumentalness','liveness']
        hit_means  = hits_df[norm_feats].mean()
        flop_means = flops_df[norm_feats].mean()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Hits', x=norm_feats, y=hit_means,
            marker_color=GREEN, marker_line_width=0,
        ))
        fig_bar.add_trace(go.Bar(
            name='Flops', x=norm_feats, y=flop_means,
            marker_color='#1C3022', marker_line_width=0,
        ))
        fig_bar.update_layout(
            barmode='group', bargap=0.25, bargroupgap=0.08,
        )
        dark_layout(fig_bar, height=340)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Insights ──────────────────────────────────────────────────────
        for insight in [
            f"<strong>Hits are {((hit_means['danceability']/flop_means['danceability'])-1)*100:.0f}% more danceable</strong> than flops on average.",
            f"<strong>Acousticness is {((flop_means['acousticness']/hit_means['acousticness'])-1)*100:.0f}% higher in flops</strong> — electronic and produced tracks dominate the charts.",
            f"<strong>Instrumentalness nearly zero in hits</strong> — chart songs almost always have vocals.",
        ]:
            st.markdown(f'<div class="insight-box">{insight}</div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Scatter: Energy vs Danceability ──────────────────────────────
        st.markdown('<div class="section-label">Energy × Danceability — The Hit Zone</div>',
                    unsafe_allow_html=True)

        sample = df.sample(min(4000, len(df)), random_state=42)
        fig_scatter = go.Figure()
        for label, subset, color, opacity in [
            ('Flops', sample[sample['is_hit']==0], '#1C3022', 0.4),
            ('Hits',  sample[sample['is_hit']==1], GREEN,     0.55),
        ]:
            fig_scatter.add_trace(go.Scatter(
                x=subset['energy'], y=subset['danceability'],
                mode='markers',
                name=label,
                marker=dict(color=color, size=4, opacity=opacity),
            ))
        # Hit zone ellipse annotation
        fig_scatter.add_shape(type='circle',
            x0=0.5, y0=0.48, x1=0.95, y1=0.88,
            line=dict(color=GREEN, width=1.5, dash='dot'),
        )
        fig_scatter.add_annotation(
            x=0.72, y=0.92, text="Hit sweet spot",
            showarrow=False,
            font=dict(color=GREEN, size=11, family='Syne'),
        )
        fig_scatter.update_xaxes(title='Energy')
        fig_scatter.update_yaxes(title='Danceability')
        dark_layout(fig_scatter, height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    if bundle is None:
        st.error("⚠️  Model not found. Run the modelling notebook first.")
    else:
        meta  = bundle.get('metadata', {})
        model = bundle['model']

        # ── Model KPIs ────────────────────────────────────────────────────
        st.markdown('<div class="section-label">XGBoost Model · Performance</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col, val, label in [
            (m1, f"{meta.get('accuracy',0)*100:.1f}%",  "Accuracy"),
            (m2, f"{meta.get('roc_auc',0):.3f}",        "ROC-AUC"),
            (m3, f"{meta.get('cv_auc',0):.3f}",         "CV AUC (5-fold)"),
            (m4, f"{meta.get('trained_on',0):,}",       "Training Songs"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature importance ────────────────────────────────────────────
        st.markdown('<div class="section-label">Feature Importance — What Predicts a Hit?</div>',
                    unsafe_allow_html=True)

        imp_df = pd.DataFrame({
            'feature':    FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        colors = [GREEN if v > imp_df['importance'].median()
                  else '#1C3022' for v in imp_df['importance']]

        fig_imp = go.Figure(go.Bar(
            x=imp_df['importance'],
            y=imp_df['feature'],
            orientation='h',
            marker_color=colors,
            marker_line_width=0,
        ))
        fig_imp.update_xaxes(title='Importance Score')
        dark_layout(fig_imp, height=460)
        st.plotly_chart(fig_imp, use_container_width=True)

        # ── Top feature insights ──────────────────────────────────────────
        top3 = imp_df.tail(3)['feature'].tolist()[::-1]
        for i, feat in enumerate(top3, 1):
            st.markdown(
                f'<div class="insight-box"><strong>#{i} {feat.title()}</strong> — '
                f'one of the strongest predictors of chart success in the model.</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model comparison bar ──────────────────────────────────────────
        st.markdown('<div class="section-label">Model Comparison</div>',
                    unsafe_allow_html=True)

        comp_df = pd.DataFrame({
            'Model':    ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'ROC-AUC':  [0.72, 0.81, meta.get('roc_auc', 0.85)],
            'Accuracy': [0.68, 0.76, meta.get('accuracy', 0.80)],
        })

        fig_comp = go.Figure()
        for metric, color in [('ROC-AUC', GREEN), ('Accuracy', ORANGE)]:
            fig_comp.add_trace(go.Bar(
                name=metric,
                x=comp_df['Model'],
                y=comp_df[metric],
                marker_color=color,
                marker_line_width=0,
                opacity=0.9,
            ))
        fig_comp.update_layout(barmode='group', bargap=0.3, bargroupgap=0.1)
        dark_layout(fig_comp, height=320)
        st.plotly_chart(fig_comp, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — DECADE EXPLORER
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)

    if df is None:
        st.error("⚠️  Dataset not found.")
    else:
        hits_df = df[df['is_hit'] == 1]
        decade_labels = {
            1960: "1960s", 1970: "1970s", 1980: "1980s",
            1990: "1990s", 2000: "2000s", 2010: "2010s",
        }

        st.markdown('<div class="section-label">How the Hit Formula Evolved by Decade</div>',
                    unsafe_allow_html=True)

        trend_feats = ['danceability','energy','valence',
                       'acousticness','loudness','speechiness']
        decade_means = hits_df.groupby('decade')[trend_feats].mean()

        fig_trends = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f.replace('_',' ').title() for f in trend_feats],
            vertical_spacing=0.18,
            horizontal_spacing=0.08,
        )

        decade_x = [str(d)+'s' for d in decade_means.index]
        colors_line = [GREEN, '#4fc3f7', ORANGE, '#ce93d8', '#ffb74d', '#ef9a9a']

        for i, (feat, col) in enumerate(zip(trend_feats, colors_line)):
            row, c = divmod(i, 3)
            vals = decade_means[feat]
            fig_trends.add_trace(go.Scatter(
                x=decade_x, y=vals,
                mode='lines+markers',
                line=dict(color=col, width=2.5),
                marker=dict(size=7, color=col,
                            line=dict(color=PLOT_BG, width=2)),
                fill='tozeroy',
                fillcolor=f'rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.06)',
                showlegend=False,
            ), row=row+1, col=c+1)

        fig_trends.update_layout(
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(family='DM Sans', color=TEXT_COL, size=10),
            margin=dict(l=0, r=0, t=40, b=0),
            height=460,
        )
        for ann in fig_trends['layout']['annotations']:
            ann['font'] = dict(family='DM Sans', size=11, color='#aaa')
        fig_trends.update_xaxes(gridcolor=GRID_COL, tickfont=dict(size=9))
        fig_trends.update_yaxes(gridcolor=GRID_COL, tickfont=dict(size=9))
        st.plotly_chart(fig_trends, use_container_width=True)

        # ── Key decade findings ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Key Findings</div>',
                    unsafe_allow_html=True)

        findings = [
            ("🕺 Danceability",  "Rose steadily from the 1960s through the 2000s as disco, hip-hop, and EDM took over the charts."),
            ("⚡ Energy",         "Spiked dramatically in the 1980s with rock and synth-pop, and has stayed high ever since."),
            ("😊 Valence",        "Hit songs were happier in the 1960s–80s. Modern hits trend darker and more emotionally complex."),
            ("🎸 Acousticness",   "Collapsed after the 1970s as electric instruments and digital production became the standard."),
            ("🔊 Loudness",       "The 'loudness wars' are real — hits got progressively louder every decade until the 2010s."),
            ("🎤 Speechiness",    "Spiked in the 2010s as hip-hop and rap dominated the Billboard charts."),
        ]

        cols = st.columns(2)
        for i, (title, text) in enumerate(findings):
            with cols[i % 2]:
                st.markdown(
                    f'<div class="insight-box"><strong>{title}</strong><br>{text}</div>',
                    unsafe_allow_html=True
                )

        # ── Song duration over decades ────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Song Duration Over the Decades</div>',
                    unsafe_allow_html=True)

        df['duration_min'] = df['duration_ms'] / 60000
        dur_data = df.groupby(['decade','is_hit'])['duration_min'].median().reset_index()

        fig_dur = go.Figure()
        for label, val, color, dash in [('Hits', 1, GREEN, 'solid'),
                                          ('Flops', 0, '#2A4A34', 'dot')]:
            sub = dur_data[dur_data['is_hit'] == val]
            fig_dur.add_trace(go.Scatter(
                x=[str(d)+'s' for d in sub['decade']],
                y=sub['duration_min'],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2.5, dash=dash),
                marker=dict(size=8, color=color,
                            line=dict(color=PLOT_BG, width=2)),
            ))
        fig_dur.update_yaxes(title='Median duration (minutes)')
        dark_layout(fig_dur, height=320)
        st.plotly_chart(fig_dur, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1C2B1F; padding-top:2rem; text-align:center;
     font-size:0.75rem; color:#2A4A34; letter-spacing:0.1em; font-weight:500;">
    SPOTIFY HIT PREDICTOR &nbsp;·&nbsp; XGBOOST &nbsp;·&nbsp; 41,106 SONGS &nbsp;·&nbsp; 1960–2019 &nbsp;·&nbsp; STREAMLIT
</div>
""", unsafe_allow_html=True)
