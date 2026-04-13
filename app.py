"""
╔══════════════════════════════════════════════════════════════════╗
║          CardioSense — Heart Disease Detection AI  v3.0          ║
║   Smart Inputs · XAI · Actionable Insights · PDF Report          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import time
import base64
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioSense AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:           #06080f;
    --bg2:          #0b0e1a;
    --glass:        rgba(255,255,255,0.04);
    --glass2:       rgba(255,255,255,0.07);
    --glass3:       rgba(255,255,255,0.11);
    --border:       rgba(255,255,255,0.07);
    --border2:      rgba(255,255,255,0.13);
    --blue:         #1d6fff;
    --blue2:        #4d9fff;
    --cyan:         #00d4ff;
    --red:          #ff3b5c;
    --red-glow:     rgba(255,59,92,0.28);
    --green:        #00e5a0;
    --green-glow:   rgba(0,229,160,0.22);
    --amber:        #ffb547;
    --amber-glow:   rgba(255,181,71,0.22);
    --purple:       #9b7fff;
    --txt:          #e8eaf6;
    --muted:        #606880;
    --muted2:       #8890b0;
    --font-display: 'Syne', sans-serif;
    --font-body:    'Space Grotesk', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
}

html, body, [class*="css"], .stApp {
    background-color: var(--bg) !important;
    color: var(--txt) !important;
    font-family: var(--font-body) !important;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 8%  0%,   rgba(29,111,255,0.09)  0%, transparent 60%),
        radial-gradient(ellipse 55% 45% at 92% 100%,  rgba(255,59,92,0.07)   0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 50% 50%,   rgba(155,127,255,0.04) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
    animation: orb-drift 18s ease-in-out infinite;
}
@keyframes orb-drift {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.8; transform: scale(1.04); }
}
.stApp::after {
    content: ''; position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 1;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(6,8,15,0.98) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(24px) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

.sidebar-logo {
    display: flex; align-items: center; gap: 12px;
    padding: 1.4rem 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem; position: relative;
}
.sidebar-logo::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,59,92,0.3), transparent);
}
.logo-icon-wrap {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, #ff3b5c, #c0235e);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    animation: heartbeat 1.5s ease-in-out infinite;
    box-shadow: 0 0 24px rgba(255,59,92,0.45);
    flex-shrink: 0;
}
@keyframes heartbeat {
    0%,100% { transform: scale(1);    }
    14%      { transform: scale(1.22); }
    28%      { transform: scale(1);    }
    42%      { transform: scale(1.14); }
    70%      { transform: scale(1);    }
}
.logo-name { font-family: var(--font-display) !important; font-size: 1.25rem; font-weight: 700; color: var(--txt); line-height: 1.1; letter-spacing: -0.01em; }
.logo-name em { color: var(--red); font-style: normal; }
.logo-sub { font-size: 0.6rem; color: var(--muted); letter-spacing: 0.16em; text-transform: uppercase; font-weight: 600; margin-top: 1px; }

.section-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 20px; padding: 5px 13px;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--purple); margin: 1rem 0 0.6rem;
}

/* Normal hint text under sliders */
.hint-text {
    font-size: 0.68rem; color: var(--muted);
    margin-top: -6px; margin-bottom: 6px;
    padding-left: 2px; line-height: 1.5;
}
.hint-ok   { color: #00c985; }
.hint-warn { color: var(--amber); }
.hint-bad  { color: var(--red); }

[data-testid="stSlider"] { padding-bottom: 0.1rem !important; }

[data-testid="stSelectbox"] > div > div {
    background: var(--glass) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--txt) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(29,111,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(29,111,255,0.08) !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--red) 0%, #b81f53 100%) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-family: var(--font-body) !important;
    font-weight: 600 !important; font-size: 0.92rem !important;
    letter-spacing: 0.06em; padding: 0.75rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px var(--red-glow) !important;
    width: 100% !important; position: relative; overflow: hidden;
}
[data-testid="stButton"] > button::before {
    content: ''; position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    transition: left 0.5s;
}
[data-testid="stButton"] > button:hover::before { left: 100%; }
[data-testid="stButton"] > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 36px var(--red-glow) !important; }
[data-testid="stButton"] > button:active { transform: translateY(0px) !important; }

/* ── ECG STRIP ── */
.ecg-strip {
    height: 3px;
    background: linear-gradient(90deg,
        transparent 0%, transparent 28%, var(--red) 28%, var(--red) 29%,
        transparent 29%, transparent 35%, var(--red) 35%, var(--red) 36.5%,
        #ffffff 36.5%, #ffffff 38%, var(--red) 38%, var(--red) 40%,
        transparent 40%, transparent 100%);
    background-size: 280% 100%;
    animation: ecg-scroll 2.8s linear infinite;
    opacity: 0.65;
}
@keyframes ecg-scroll { from { background-position: 200% 0; } to { background-position: -100% 0; } }

/* ── HERO ── */
.hero { padding: 2.8rem 0 2rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2em; text-transform: uppercase; color: var(--red); margin-bottom: 0.7rem;
}
.hero-eyebrow::before { content: ''; display: inline-block; width: 22px; height: 1.5px; background: var(--red); border-radius: 1px; }
.hero-title { font-family: var(--font-display) !important; font-size: 3.2rem; font-weight: 800; line-height: 1.05; color: var(--txt); margin: 0; letter-spacing: -0.02em; }
.hero-title .accent { background: linear-gradient(135deg, var(--blue2), var(--cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-subtitle { font-size: 0.95rem; color: var(--muted2); margin-top: 0.8rem; font-weight: 400; line-height: 1.65; max-width: 620px; }
.ecg-line {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, transparent 25%, var(--red) 25%, var(--red) 26%, transparent 26%, transparent 33%, var(--red) 33%, var(--red) 34.5%, #fff 34.5%, #fff 36%, var(--red) 36%, var(--red) 38%, transparent 38%);
    margin: 1rem 0 1.4rem; opacity: 0.45;
    animation: ecg-slide 3.2s linear infinite; background-size: 220% 100%; border-radius: 1px;
}
@keyframes ecg-slide { 0% { background-position: 120% 0; } 100% { background-position: -120% 0; } }

/* ── STAT CARDS ── */
.stat-card {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.4rem 1.2rem; text-align: center;
    position: relative; overflow: hidden; transition: border-color 0.3s, transform 0.3s;
}
.stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); }
.stat-card:hover { border-color: var(--border2); transform: translateY(-3px); }
.stat-value { font-family: var(--font-display) !important; font-size: 2.2rem; font-weight: 700; color: var(--txt); line-height: 1; }
.stat-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; margin-top: 0.4rem; }

/* ── GLASS CARD ── */
.glass-card {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.6rem; backdrop-filter: blur(14px);
    position: relative; overflow: hidden; transition: border-color 0.3s, transform 0.3s;
}
.glass-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); }
.glass-card:hover { border-color: var(--border2); transform: translateY(-2px); }
.hud-card { position: relative; }
.hud-card::after { content: ''; position: absolute; top: 0; left: 0; width: 14px; height: 14px; border-top: 1.5px solid rgba(29,111,255,0.5); border-left: 1.5px solid rgba(29,111,255,0.5); border-radius: 2px 0 0 0; pointer-events: none; }

/* ── VERDICT CARDS ── */
.verdict-card {
    border-radius: 20px; padding: 2rem; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 1rem; transition: all 0.4s ease;
}
.verdict-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent); }
.verdict-high {
    background: linear-gradient(140deg, rgba(255,59,92,0.13), rgba(255,59,92,0.04));
    border: 1.5px solid rgba(255,59,92,0.55);
    box-shadow: 0 0 48px rgba(255,59,92,0.12), inset 0 0 40px rgba(255,59,92,0.04);
    animation: pulse-red 3s ease-in-out infinite;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 48px rgba(255,59,92,0.12), inset 0 0 40px rgba(255,59,92,0.04); }
    50%      { box-shadow: 0 0 64px rgba(255,59,92,0.22), inset 0 0 40px rgba(255,59,92,0.06); }
}
.verdict-low {
    background: linear-gradient(140deg, rgba(0,229,160,0.10), rgba(0,229,160,0.03));
    border: 1.5px solid rgba(0,229,160,0.45);
    box-shadow: 0 0 40px rgba(0,229,160,0.10), inset 0 0 40px rgba(0,229,160,0.03);
}
.verdict-icon { font-size: 3rem; margin-bottom: 0.6rem; display: block; animation: pop-in 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.3); }
@keyframes pop-in { from { transform: scale(0.3) rotate(-10deg); opacity: 0; } to { transform: scale(1) rotate(0deg); opacity: 1; } }
.verdict-label { font-family: var(--font-display) !important; font-size: 1.9rem; font-weight: 800; letter-spacing: 0.04em; }
.verdict-high .verdict-label { color: var(--red); }
.verdict-low  .verdict-label { color: var(--green); }
.verdict-conf { font-size: 0.82rem; color: var(--muted2); margin-top: 0.4rem; font-family: var(--font-mono); }

/* Confidence sub-label */
.confidence-label {
    font-size: 0.78rem; color: var(--muted2); text-align: center;
    margin-top: 6px; font-family: var(--font-mono);
}
.confidence-label strong { color: var(--txt); }

/* ── SYSTEM BADGES ── */
.system-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem; }
.badge-ml     { background: rgba(77,159,255,0.12); color: var(--blue2); border: 1px solid rgba(77,159,255,0.28); }
.badge-expert { background: rgba(155,127,255,0.12); color: var(--purple); border: 1px solid rgba(155,127,255,0.28); }

/* ── EXPERT RULE ITEMS ── */
.rule-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 0.6rem 0.9rem; border-radius: 10px; margin: 5px 0;
    font-size: 0.81rem; line-height: 1.45;
    transition: background 0.2s; border-left: 3px solid transparent;
}
.rule-item:hover { background: rgba(255,255,255,0.04); }
.rule-flag-item { background: rgba(255,59,92,0.06); border-left-color: var(--red); }
.rule-ok-item   { background: rgba(0,229,160,0.04); border-left-color: var(--green); }
.rule-icon { font-size: 0.88rem; flex-shrink: 0; margin-top: 2px; }
.rule-main { font-weight: 500; color: var(--txt); }
.rule-explain { font-size: 0.74rem; color: var(--muted2); margin-top: 2px; line-height: 1.4; }

/* ── VERDICT BANNER ── */
.verdict-banner {
    border-radius: 14px; padding: 1.2rem 2rem; text-align: center;
    font-weight: 600; font-size: 1rem; letter-spacing: 0.03em; margin-top: 0.5rem;
    position: relative; overflow: hidden;
}
.verdict-banner::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent); }
.banner-agree-high { background: rgba(255,59,92,0.10); border: 1.5px solid rgba(255,59,92,0.45); color: var(--red); }
.banner-agree-low  { background: rgba(0,229,160,0.08); border: 1.5px solid rgba(0,229,160,0.4);  color: var(--green); }
.banner-disagree   { background: rgba(255,181,71,0.08); border: 1.5px solid rgba(255,181,71,0.4); color: var(--amber); }

/* ── ACTIONABLE INSIGHTS ── */
.insights-section { margin: 1.5rem 0; }
.insights-header {
    font-family: var(--font-display) !important; font-size: 1.3rem; font-weight: 700;
    color: var(--txt); margin-bottom: 1rem;
    display: flex; align-items: center; gap: 10px;
}
.insight-card {
    display: flex; align-items: flex-start; gap: 14px;
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.2rem;
    margin: 8px 0; transition: border-color 0.25s, transform 0.25s;
}
.insight-card:hover { border-color: var(--border2); transform: translateX(4px); }
.insight-card.urgent { border-left: 3px solid var(--red);   background: rgba(255,59,92,0.05); }
.insight-card.warning { border-left: 3px solid var(--amber); background: rgba(255,181,71,0.04); }
.insight-card.ok      { border-left: 3px solid var(--green); background: rgba(0,229,160,0.04); }
.insight-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 2px; }
.insight-title { font-weight: 600; font-size: 0.88rem; color: var(--txt); margin-bottom: 3px; }
.insight-desc  { font-size: 0.78rem; color: var(--muted2); line-height: 1.5; }

/* ── ANALYTICS ── */
.analytics-header { font-family: var(--font-display) !important; font-size: 1.7rem; font-weight: 700; color: var(--txt) !important; margin-bottom: 0.2rem; }
.analytics-sub { font-size: 0.85rem; color: var(--muted2); margin-bottom: 1.5rem; line-height: 1.5; }

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(255,181,71,0.05); border: 1px solid rgba(255,181,71,0.2);
    border-radius: 10px; padding: 0.9rem 1.3rem;
    font-size: 0.76rem; color: var(--amber);
    text-align: center; margin-top: 1.5rem; line-height: 1.6;
}

/* ── HOW-IT-WORKS ── */
.how-card {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 16px; padding: 2rem 1.6rem; text-align: center;
    position: relative; overflow: hidden; transition: border-color 0.3s, transform 0.3s;
}
.how-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); }
.how-card:hover { border-color: var(--border2); transform: translateY(-3px); }
.how-num  { font-family: var(--font-display) !important; font-size: 3.2rem; font-weight: 800; opacity: 0.22; line-height: 1; margin-bottom: 1rem; }
.how-title { font-weight: 600; font-size: 0.95rem; color: var(--txt); margin-bottom: 0.6rem; }
.how-desc  { font-size: 0.8rem; color: var(--muted2); line-height: 1.65; }

/* ── PDF / DOWNLOAD BUTTON ── */
.pdf-btn-wrap { display: flex; justify-content: flex-end; margin: 1rem 0 0.5rem; }
.pdf-section {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.4rem 1.6rem;
    margin-top: 1.2rem; display: flex; align-items: center; justify-content: space-between;
}
.pdf-info { flex: 1; }
.pdf-title { font-family: var(--font-display) !important; font-size: 1rem; font-weight: 700; color: var(--txt); margin-bottom: 4px; }
.pdf-sub   { font-size: 0.78rem; color: var(--muted2); }

/* ── MISC ── */
[data-testid="stTabs"] [data-testid="stTab"] { font-family: var(--font-body) !important; font-weight: 500 !important; font-size: 0.85rem !important; }
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] { color: var(--txt) !important; border-bottom-color: var(--blue2) !important; }
.stAlert { background: var(--glass) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
hr { border-color: var(--border) !important; }
[data-testid="stPlotlyChart"] { background: transparent !important; }
h1, h2, h3 { font-family: var(--font-display) !important; color: var(--txt) !important; font-weight: 700 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 12px !important; overflow: hidden; }
.sidebar-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.68rem; color: var(--muted); text-align: center; line-height: 1.7; }
.sidebar-footer .dot { color: var(--red); }
.pulse-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 2s ease-in-out infinite; vertical-align: middle; margin-right: 4px; }
@keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(0.8); } }

/* Loading animation */
.loading-bar {
    height: 3px; border-radius: 2px; margin: 8px 0;
    background: linear-gradient(90deg, var(--blue), var(--cyan), var(--blue));
    background-size: 200% 100%;
    animation: loading-sweep 1.4s linear infinite;
}
@keyframes loading-sweep { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

/* Expander styling */
[data-testid="stExpander"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    color: var(--muted2) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CHART THEME
# ══════════════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#606880", size=12),
    margin=dict(t=24, b=24, l=20, r=20),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(color="#e8eaf6", size=11),
    ),
)
GRID_STYLE = dict(
    gridcolor="rgba(255,255,255,0.05)",
    zerolinecolor="rgba(255,255,255,0.08)",
    tickfont=dict(color="#606880"),
)

# Normal reference lines for benchmarking charts
NORMAL_REFS = {
    "Resting_Blood_Pressure": 120,
    "Cholesterol":            200,
    "Maximum_Heart_Rate":     150,
    "ST_Depression":          1.0,
}


# ══════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    for path in ["cleaned_data.csv", "data/cleaned_data.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("cleaned_data.csv not found.")


@st.cache_resource
def load_model():
    for path in ["heart_disease_model.pkl", "models/heart_disease_model.pkl"]:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("heart_disease_model.pkl not found.")


df    = load_data()
model = load_model()

FEATURE_COLS = [c for c in df.columns if c != "Heart_Disease_Target"]
DATA_MEANS   = df[FEATURE_COLS].mean()

SCALE_BOUNDS = {
    "Resting_Blood_Pressure": (94,  200),
    "Cholesterol":            (126, 564),
    "Maximum_Heart_Rate":     (71,  202),
    "ST_Depression":          (0.0, 6.2),
}


def to_raw(col, scaled):
    lo, hi = SCALE_BOUNDS[col]
    return lo + scaled * (hi - lo)


def to_scaled(col, raw):
    lo, hi = SCALE_BOUNDS[col]
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════
# INPUT VALIDATION HINTS
# ══════════════════════════════════════════════════════════════════
def bp_hint(v):
    if v < 120:   return "hint-ok",   f"✓ Normal (< 120 mmHg)"
    if v < 130:   return "hint-warn", f"⚠ Elevated (120–129 mmHg)"
    if v < 140:   return "hint-warn", f"⚠ High Stage 1 (130–139 mmHg)"
    return             "hint-bad",    f"✗ High Stage 2 (≥ 140 mmHg)"

def chol_hint(v):
    if v < 200:   return "hint-ok",   f"✓ Desirable (< 200 mg/dL)"
    if v < 240:   return "hint-warn", f"⚠ Borderline High (200–239 mg/dL)"
    return             "hint-bad",    f"✗ High Risk (≥ 240 mg/dL)"

def hr_hint(v, age):
    pred_max = 220 - age
    pct = (v / pred_max) * 100
    if pct >= 85:  return "hint-ok",   f"✓ {pct:.0f}% of predicted max ({pred_max} bpm)"
    if pct >= 65:  return "hint-warn", f"⚠ {pct:.0f}% of predicted max — reduced capacity"
    return              "hint-bad",    f"✗ {pct:.0f}% of predicted max — significantly low"

def st_hint(v):
    if v < 1.0:   return "hint-ok",   f"✓ Normal (< 1.0 mm)"
    if v < 2.0:   return "hint-warn", f"⚠ Mild depression (1.0–2.0 mm)"
    return             "hint-bad",    f"✗ Significant ischemic marker (> 2.0 mm)"


# ══════════════════════════════════════════════════════════════════
# EXPERT SYSTEM — 12 Clinical Rules (with explanations)
# ══════════════════════════════════════════════════════════════════
def run_expert_system(age, bp_raw, chol_raw, max_hr_raw, st_dep_raw,
                      sex, chest_pain, ecg, exercise_angina,
                      st_slope, major_vessels, thalassemia):
    flags = []

    def check(cond, yes_msg, no_msg, yes_explain="", no_explain=""):
        flags.append((
            yes_msg if cond else no_msg,
            yes_explain if cond else no_explain,
            cond,
        ))

    check(
        chol_raw > 240,
        f"Cholesterol {chol_raw:.0f} mg/dL — elevated CVD risk",
        f"Cholesterol {chol_raw:.0f} mg/dL — within acceptable range",
        "High LDL cholesterol accelerates plaque buildup in coronary arteries, increasing the risk of blockage and heart attack.",
        "Cholesterol is within a safe range, reducing atherosclerosis risk.",
    )
    check(
        chol_raw > 200 and age > 50,
        f"Age {age} + cholesterol {chol_raw:.0f} mg/dL — compounded risk",
        "Age–cholesterol combination is acceptable",
        "After 50, vascular elasticity decreases. Elevated cholesterol in older patients compounds arterial stress significantly.",
        "The combination of age and cholesterol does not indicate compounded cardiovascular risk.",
    )
    check(
        bp_raw > 140,
        f"BP {bp_raw:.0f} mmHg — Stage 2 hypertension",
        f"BP {bp_raw:.0f} mmHg — within normal limits",
        "Sustained BP above 140 mmHg forces the heart to work harder, causing left ventricular hypertrophy over time.",
        "Blood pressure is within a healthy range, minimizing cardiac overload.",
    )
    pred_max  = 220 - age
    hr_thresh = 0.85 * pred_max
    check(
        max_hr_raw < hr_thresh,
        f"Max HR {max_hr_raw:.0f} bpm — below 85% of predicted ({hr_thresh:.0f} bpm)",
        f"Max HR {max_hr_raw:.0f} bpm — chronotropically appropriate",
        "Inability to reach target heart rate during exercise suggests impaired cardiac output and possible coronary artery disease.",
        "Heart rate reached an appropriate level during exercise, indicating good chronotropic response.",
    )
    check(
        exercise_angina == "Yes",
        "Exercise-induced angina present",
        "No exercise-induced angina reported",
        "Chest pain triggered by physical exertion is a strong indicator of obstructive coronary artery disease (CAD), as the heart demands more oxygen than narrowed arteries can supply.",
        "Absence of exercise angina reduces the likelihood of obstructive coronary artery disease.",
    )
    check(
        chest_pain == "Asymptomatic",
        "Asymptomatic chest pain pattern — silent ischemia risk",
        "Chest pain pattern does not suggest silent ischemia",
        "Paradoxically, asymptomatic patients often have the most severe CAD. The absence of pain may mask serious ischemia, delaying treatment.",
        "The chest pain pattern reported is not associated with elevated silent ischemia risk.",
    )
    check(
        st_dep_raw > 2.0,
        f"ST depression {st_dep_raw:.2f} mm — significant ischemic marker",
        f"ST depression {st_dep_raw:.2f} mm — within acceptable range",
        "ST segment depression > 2mm during exercise indicates myocardial ischemia — the heart muscle is not receiving adequate oxygen.",
        "ST depression is within acceptable limits, reducing concern for exercise-induced ischemia.",
    )
    check(
        st_slope == "Downsloping",
        "Downsloping ST slope — severe ischemia indicator",
        "ST slope is not downsloping",
        "A downsloping ST pattern during peak exercise has the highest specificity for significant coronary artery disease among all ST-change patterns.",
        "ST slope pattern is not associated with high-severity ischemia.",
    )
    check(
        int(major_vessels) >= 2,
        f"{major_vessels} major vessels positive — multi-vessel CAD",
        f"{major_vessels} major vessel(s) — no multi-vessel concern",
        "Multi-vessel coronary artery disease significantly elevates risk. Two or more affected vessels indicate systemic atherosclerosis requiring urgent intervention.",
        "Number of positive vessels does not indicate multi-vessel coronary artery disease.",
    )
    check(
        thalassemia == "Reversible Defect",
        "Reversible thalassemia defect — high CAD predictive value",
        "Thalassemia type is not a high-risk indicator",
        "A reversible perfusion defect on nuclear stress testing indicates viable but ischemic myocardium — strong evidence of obstructive CAD.",
        "Thalassemia stress test result does not indicate significant myocardial ischemia.",
    )
    check(
        age > 60 and bp_raw > 130,
        f"Age {age} + BP {bp_raw:.0f} mmHg — senior cardiovascular risk",
        "No senior hypertension compounding concern",
        "In patients over 60, even mildly elevated blood pressure significantly increases the risk of heart failure and coronary events due to reduced arterial compliance.",
        "Age and blood pressure combination does not indicate senior cardiovascular compounding risk.",
    )
    check(
        ecg in ["Left Ventricular Hypertrophy", "ST Wave Abnormality"],
        f"ECG: {ecg} — structural abnormality",
        "ECG result is normal",
        "ECG abnormalities such as LVH or ST-wave changes indicate the heart has already undergone structural or electrical remodeling, often from chronic pressure overload or ischemia.",
        "Resting ECG is normal, with no signs of structural or electrical heart abnormality.",
    )

    n_flagged = sum(1 for _, _, f in flags if f)
    return flags, n_flagged >= 3, n_flagged


# ══════════════════════════════════════════════════════════════════
# ACTIONABLE INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════════════
def generate_insights(ml_pred, expert_high_risk, ml_prob,
                      age, bp_raw, chol_raw, max_hr_raw, st_dep_raw,
                      exercise_angina, chest_pain, major_vessels, thalassemia):
    insights = []

    overall_risk = ml_pred == 1 or expert_high_risk

    if overall_risk:
        insights.append(("urgent", "🏥",
            "Schedule a Cardiology Consultation",
            "Based on your profile, we strongly recommend booking an appointment with a cardiologist as soon as possible for a comprehensive evaluation including stress testing and coronary imaging."))

    if bp_raw >= 140:
        insights.append(("urgent", "💊",
            "Address Hypertension Immediately",
            "Your blood pressure is in Stage 2 hypertension territory. Consult your doctor about antihypertensive medication. Reduce sodium intake to below 2,300 mg/day and limit alcohol consumption."))
    elif bp_raw >= 130:
        insights.append(("warning", "🧂",
            "Manage Elevated Blood Pressure",
            "Your BP is in the elevated-to-Stage 1 range. Adopt the DASH diet (rich in fruits, vegetables, whole grains), exercise regularly, and monitor your BP at home weekly."))

    if chol_raw >= 240:
        insights.append(("urgent", "🥗",
            "Reduce High Cholesterol",
            "Your cholesterol significantly exceeds the 200 mg/dL target. Eliminate saturated fats and trans fats. Increase soluble fiber (oats, beans, flaxseed). Ask your doctor about statin therapy."))
    elif chol_raw >= 200:
        insights.append(("warning", "🫒",
            "Lower Borderline Cholesterol",
            "Your cholesterol is borderline high. Replace saturated fats with healthy fats (olive oil, avocados, nuts). Aim for 30 minutes of aerobic exercise at least 5 days per week."))

    if exercise_angina == "Yes":
        insights.append(("urgent", "⛔",
            "Avoid Intense Physical Exertion",
            "Exercise-induced chest pain indicates your heart may not be getting enough blood during activity. Avoid strenuous exercise until cleared by a cardiologist. Do not ignore this symptom."))

    if st_dep_raw >= 2.0:
        insights.append(("urgent", "📋",
            "Request an Urgent Stress ECG Review",
            "Significant ST depression is a red flag for myocardial ischemia. Ask your doctor for a formal interpretation of your exercise stress test and possible referral for coronary angiography."))

    if int(major_vessels) >= 2:
        insights.append(("urgent", "🫀",
            "Multi-Vessel Disease — Specialist Required",
            "Two or more affected coronary vessels indicates systemic coronary artery disease. This typically requires intervention — either percutaneous coronary intervention (PCI) or bypass surgery (CABG). Seek a specialist referral urgently."))

    if age > 50 and not overall_risk:
        insights.append(("warning", "📅",
            "Annual Cardiac Screening Recommended",
            "At your age, annual cardiovascular screening is advisable even without current high risk. Track blood pressure, cholesterol, and fasting glucose yearly. Consider a baseline stress test."))

    if not overall_risk and bp_raw < 130 and chol_raw < 200:
        insights.append(("ok", "✅",
            "Maintain Your Healthy Lifestyle",
            "Your current profile is within safe ranges. Continue regular physical activity (150 min/week moderate intensity), a heart-healthy diet, adequate sleep, and stress management to preserve your cardiovascular health."))

    return insights


# ══════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════
def generate_pdf_report(age, sex, bp_raw, chol_raw, max_hr_raw, st_dep_raw,
                        chest_pain, ecg, exercise_angina, st_slope,
                        major_vessels, thalassemia,
                        ml_pred, ml_prob, expert_high_risk, n_flagged,
                        flags, insights):
    """Generate an HTML-based printable report as a downloadable file."""

    risk_color  = "#ff3b5c" if (ml_pred == 1 or expert_high_risk) else "#00e5a0"
    risk_label  = "HIGH RISK" if (ml_pred == 1 or expert_high_risk) else "LOW RISK"
    conf_pct    = ml_prob if ml_pred == 1 else (100 - ml_prob)
    now         = datetime.now().strftime("%B %d, %Y — %H:%M")

    flagged_rules = [(msg, exp) for msg, exp, f in flags if f]
    ok_rules      = [(msg, exp) for msg, exp, f in flags if not f]

    rules_html = ""
    for msg, exp in flagged_rules:
        rules_html += f'<div class="rule r-flag"><span class="dot red">●</span><div><strong>{msg}</strong><p>{exp}</p></div></div>'
    for msg, exp in ok_rules:
        rules_html += f'<div class="rule r-ok"><span class="dot green">●</span><div><strong>{msg}</strong><p>{exp}</p></div></div>'

    insights_html = ""
    for sev, icon, title, desc in insights:
        border = {"urgent":"#ff3b5c","warning":"#ffb547","ok":"#00e5a0"}.get(sev,"#888")
        insights_html += f'<div class="insight" style="border-left:3px solid {border}"><strong>{icon} {title}</strong><p>{desc}</p></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CardioSense Report — {now}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',sans-serif; background:#fff; color:#1a1a2e; font-size:13px; }}
  .page {{ max-width:780px; margin:0 auto; padding:40px 48px; }}
  .header {{ display:flex; justify-content:space-between; align-items:flex-start; border-bottom:2px solid #f0f0f0; padding-bottom:20px; margin-bottom:28px; }}
  .logo {{ font-size:20px; font-weight:700; color:#1a1a2e; }}
  .logo span {{ color:#ff3b5c; }}
  .meta {{ text-align:right; font-size:11px; color:#888; line-height:1.8; }}
  .verdict-box {{ border-radius:12px; padding:20px 28px; margin-bottom:28px; border:2px solid {risk_color}; background:{risk_color}18; display:flex; align-items:center; justify-content:space-between; }}
  .verdict-label {{ font-size:26px; font-weight:700; color:{risk_color}; }}
  .verdict-sub {{ font-size:12px; color:#666; margin-top:4px; }}
  .verdict-score {{ text-align:right; }}
  .verdict-score .pct {{ font-size:32px; font-weight:700; color:{risk_color}; }}
  .verdict-score .sub {{ font-size:11px; color:#888; }}
  h2 {{ font-size:13px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#888; margin:24px 0 12px; border-bottom:1px solid #f0f0f0; padding-bottom:6px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:4px; }}
  .field {{ background:#f8f9fc; border-radius:8px; padding:10px 14px; }}
  .field .lbl {{ font-size:10px; color:#999; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:2px; }}
  .field .val {{ font-size:14px; font-weight:600; color:#1a1a2e; }}
  .rule {{ display:flex; gap:10px; padding:9px 12px; border-radius:8px; margin:5px 0; background:#f9f9f9; align-items:flex-start; }}
  .r-flag {{ background:#fff5f6; }}
  .r-ok   {{ background:#f4fdf9; }}
  .rule .dot {{ font-size:10px; margin-top:3px; flex-shrink:0; }}
  .dot.red {{ color:#ff3b5c; }}
  .dot.green {{ color:#00c985; }}
  .rule strong {{ font-size:12px; font-weight:600; }}
  .rule p {{ font-size:11px; color:#666; margin-top:2px; line-height:1.5; }}
  .insight {{ padding:10px 14px; border-radius:8px; background:#f8f9fc; margin:6px 0; }}
  .insight strong {{ font-size:12px; font-weight:600; }}
  .insight p {{ font-size:11px; color:#555; margin-top:3px; line-height:1.5; }}
  .footer {{ margin-top:32px; padding-top:16px; border-top:1px solid #f0f0f0; font-size:10px; color:#bbb; text-align:center; line-height:1.8; }}
  @media print {{ body {{ -webkit-print-color-adjust:exact; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <div class="logo">Cardio<span>Sense</span> AI</div>
      <div style="font-size:11px;color:#999;margin-top:4px;">Cardiac Risk Assessment Report</div>
    </div>
    <div class="meta">
      Generated: {now}<br>
      Model: Decision Tree + Expert System<br>
      Version: 3.0
    </div>
  </div>

  <div class="verdict-box">
    <div>
      <div class="verdict-label">{risk_label}</div>
      <div class="verdict-sub">Combined ML + Expert System Verdict</div>
      <div style="font-size:11px;color:#888;margin-top:6px;">
        ML: {"High Risk" if ml_pred==1 else "Low Risk"} &nbsp;·&nbsp;
        Expert: {"High Risk" if expert_high_risk else "Low Risk"} ({n_flagged}/{len(flags)} rules triggered)
      </div>
    </div>
    <div class="verdict-score">
      <div class="pct">{conf_pct:.1f}%</div>
      <div class="sub">Model Confidence</div>
    </div>
  </div>

  <h2>Patient Profile</h2>
  <div class="grid">
    <div class="field"><div class="lbl">Age</div><div class="val">{age} years</div></div>
    <div class="field"><div class="lbl">Sex</div><div class="val">{sex}</div></div>
    <div class="field"><div class="lbl">Resting Blood Pressure</div><div class="val">{bp_raw} mmHg</div></div>
    <div class="field"><div class="lbl">Cholesterol</div><div class="val">{chol_raw:.0f} mg/dL</div></div>
    <div class="field"><div class="lbl">Max Heart Rate</div><div class="val">{max_hr_raw} bpm</div></div>
    <div class="field"><div class="lbl">ST Depression</div><div class="val">{st_dep_raw:.1f} mm</div></div>
    <div class="field"><div class="lbl">Chest Pain Type</div><div class="val">{chest_pain}</div></div>
    <div class="field"><div class="lbl">Resting ECG</div><div class="val">{ecg}</div></div>
    <div class="field"><div class="lbl">Exercise-Induced Angina</div><div class="val">{exercise_angina}</div></div>
    <div class="field"><div class="lbl">ST Slope</div><div class="val">{st_slope}</div></div>
    <div class="field"><div class="lbl">Major Vessels</div><div class="val">{major_vessels}</div></div>
    <div class="field"><div class="lbl">Thalassemia</div><div class="val">{thalassemia}</div></div>
  </div>

  <h2>Expert System Rule Evaluation ({n_flagged} Triggered)</h2>
  {rules_html}

  <h2>Recommended Actions</h2>
  {insights_html}

  <div class="footer">
    ⚕️ CardioSense is an academic demonstration tool only. It does NOT constitute medical advice, diagnosis, or treatment.<br>
    Always consult a qualified cardiologist for clinical decisions. Generated by CardioSense AI v3.0.
  </div>
</div>
</body>
</html>"""

    return html


def get_download_link(html_content, filename="CardioSense_Report.html"):
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}" style="text-decoration:none;">'\
           '<button style="background:linear-gradient(135deg,#1d6fff,#00d4ff);color:#fff;border:none;border-radius:10px;padding:10px 22px;font-family:Space Grotesk,sans-serif;font-weight:600;font-size:0.85rem;cursor:pointer;letter-spacing:0.04em;box-shadow:0 4px 18px rgba(29,111,255,0.35);">'\
           '📄 Download Report (HTML → Print as PDF)</button></a>'


# ══════════════════════════════════════════════════════════════════
# FEATURE VECTOR BUILDER
# ══════════════════════════════════════════════════════════════════
def build_feature_vector(age, bp_sc, chol_sc, hr_sc, st_sc,
                         sex, chest_pain, ecg, exercise_angina,
                         st_slope, major_vessels, thalassemia):
    row = {col: 0 for col in FEATURE_COLS}
    row["Age"]                    = age
    row["Resting_Blood_Pressure"] = bp_sc
    row["Cholesterol"]            = chol_sc
    row["Maximum_Heart_Rate"]     = hr_sc
    row["ST_Depression"]          = st_sc
    row["Sex_Female" if sex == "Female" else "Sex_Male"] = 1
    cp_map = {
        "Asymptomatic":    "Chest_Pain_Asymptomatic",
        "Atypical Angina": "Chest_Pain_Atypical_Angina",
        "Non-Anginal":     "Chest_Pain_Non_Anginal",
        "Typical Angina":  "Chest_Pain_Typical_Angina",
    }
    row[cp_map[chest_pain]] = 1
    ecg_map = {
        "Left Ventricular Hypertrophy": "ECG_Left_Ventricular_Hypertrophy",
        "Normal":                       "ECG_Normal",
        "ST Wave Abnormality":          "ECG_ST_Wave_Abnormality",
    }
    row[ecg_map[ecg]] = 1
    row["Exercise_Angina_Yes" if exercise_angina == "Yes" else "Exercise_Angina_No"] = 1
    slope_map = {
        "Downsloping": "ST_Slope_Downsloping",
        "Flat":        "ST_Slope_Flat",
        "Upsloping":   "ST_Slope_Upsloping",
    }
    row[slope_map[st_slope]] = 1
    mv_col = f"Major_Vessels_Count_{major_vessels}"
    if mv_col in row:
        row[mv_col] = 1
    thal_map = {
        "Fixed Defect 1":    "Thalassemia_Fixed_Defect_1",
        "Fixed Defect 2":    "Thalassemia_Fixed_Defect_2",
        "Normal":            "Thalassemia_Normal",
        "Reversible Defect": "Thalassemia_Reversible_Defect",
    }
    row[thal_map[thalassemia]] = 1
    return pd.DataFrame([row])[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — Step-by-Step Form with Smart Hints
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon-wrap">🫀</div>
        <div>
            <div class="logo-name">Cardio<em>Sense</em></div>
            <div class="logo-sub">AI Risk Assessment</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── STEP 1: Personal Data ──────────────────────────────────────
    with st.expander("👤  Step 1 — Personal Data", expanded=True):
        age = st.slider("Age (years)", 29, 77, 54)
        sex = st.selectbox("Sex", ["Male", "Female"])

    # ── STEP 2: Vital Measurements ────────────────────────────────
    with st.expander("📊  Step 2 — Vital Measurements", expanded=True):
        bp_raw = st.slider("Resting Blood Pressure (mmHg)", 94, 200, 130)
        bp_cls, bp_msg = bp_hint(bp_raw)
        st.markdown(f'<div class="hint-text {bp_cls}">{bp_msg}</div>', unsafe_allow_html=True)

        chol_raw = st.slider("Cholesterol (mg/dL)", 126, 564, 246)
        chol_cls, chol_msg = chol_hint(chol_raw)
        st.markdown(f'<div class="hint-text {chol_cls}">{chol_msg}</div>', unsafe_allow_html=True)

        max_hr_raw = st.slider("Maximum Heart Rate (bpm)", 71, 202, 150)
        hr_cls, hr_msg = hr_hint(max_hr_raw, age)
        st.markdown(f'<div class="hint-text {hr_cls}">{hr_msg}</div>', unsafe_allow_html=True)

        st_dep_raw = st.slider("ST Depression (mm)", 0.0, 6.2, 1.0, 0.1)
        st_cls, st_msg = st_hint(st_dep_raw)
        st.markdown(f'<div class="hint-text {st_cls}">{st_msg}</div>', unsafe_allow_html=True)

    # ── STEP 3: Clinical Profile ───────────────────────────────────
    with st.expander("🧬  Step 3 — Clinical Profile", expanded=False):
        chest_pain      = st.selectbox("Chest Pain Type",           ["Asymptomatic", "Atypical Angina", "Non-Anginal", "Typical Angina"])
        ecg             = st.selectbox("Resting ECG",               ["Normal", "Left Ventricular Hypertrophy", "ST Wave Abnormality"])
        exercise_angina = st.selectbox("Exercise-Induced Angina",   ["No", "Yes"])
        st_slope        = st.selectbox("Peak Exercise ST Slope",    ["Upsloping", "Flat", "Downsloping"])
        major_vessels   = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3, 4])
        thalassemia     = st.selectbox("Thalassemia Type",          ["Normal", "Fixed Defect 1", "Fixed Defect 2", "Reversible Defect"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Cardiac Analysis", use_container_width=True)

    st.markdown("""
    <div class="sidebar-footer">
        CardioSense v3.0 &nbsp;·&nbsp; Decision Tree + Expert Rules<br>
        <span class="dot">⚕</span>&nbsp; For academic use only
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="ecg-strip"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">↯ AI-Powered Cardiology</div>
    <h1 class="hero-title">CardioSense <span class="accent">Risk Engine</span></h1>
    <div class="ecg-line"></div>
    <p class="hero-subtitle">
        Combining a trained Decision Tree model with a 12-rule clinical Expert System
        for comprehensive, dual-perspective cardiovascular risk assessment.
    </p>
</div>
""", unsafe_allow_html=True)

bp_scaled   = to_scaled("Resting_Blood_Pressure", bp_raw)
chol_scaled = to_scaled("Cholesterol",            chol_raw)
hr_scaled   = to_scaled("Maximum_Heart_Rate",     max_hr_raw)
st_scaled   = to_scaled("ST_Depression",          st_dep_raw)


# ══════════════════════════════════════════════════════════════════
# PREDICTION PANEL
# ══════════════════════════════════════════════════════════════════
if predict_btn:

    # ── Loading animation ─────────────────────────────────────────
    with st.spinner(""):
        placeholder = st.empty()
        placeholder.markdown("""
        <div style="text-align:center;padding:2rem 0;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;color:#4d9fff;margin-bottom:12px;">
                🔬 Analyzing medical patterns...
            </div>
            <div class="loading-bar" style="max-width:380px;margin:0 auto;"></div>
            <div style="font-size:0.75rem;color:#606880;margin-top:10px;font-family:'JetBrains Mono',monospace;">
                Running Decision Tree · Evaluating 12 Expert Rules · Benchmarking
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.6)
        placeholder.empty()

    X        = build_feature_vector(age, bp_scaled, chol_scaled, hr_scaled, st_scaled,
                                    sex, chest_pain, ecg, exercise_angina,
                                    st_slope, major_vessels, thalassemia)
    ml_pred  = model.predict(X)[0]
    ml_proba = model.predict_proba(X)[0]
    ml_prob  = ml_proba[1] * 100

    flags, expert_high_risk, n_flagged = run_expert_system(
        age, bp_raw, chol_raw, max_hr_raw, st_dep_raw,
        sex, chest_pain, ecg, exercise_angina,
        st_slope, str(major_vessels), thalassemia,
    )

    # ── Confidence levels ─────────────────────────────────────────
    conf_pct   = ml_prob if ml_pred == 1 else (100 - ml_prob)
    conf_label = "High confidence" if conf_pct >= 75 else ("Moderate confidence" if conf_pct >= 55 else "Low confidence")

    # Gauge color: green → amber → red
    if ml_prob < 33:
        gauge_color = "#00e5a0"
    elif ml_prob < 66:
        gauge_color = "#ffb547"
    else:
        gauge_color = "#ff3b5c"

    # ── Verdict columns ───────────────────────────────────────────
    col_ml, col_exp = st.columns(2, gap="large")

    with col_ml:
        st.markdown('<div class="system-badge badge-ml">🤖 Machine Learning Model</div>',
                    unsafe_allow_html=True)
        verdict_cls = "verdict-high" if ml_pred == 1 else "verdict-low"
        verdict_ico = "🚨" if ml_pred == 1 else "✅"
        verdict_lbl = "HIGH RISK" if ml_pred == 1 else "LOW RISK"
        st.markdown(f"""
        <div class="verdict-card {verdict_cls} hud-card">
            <span class="verdict-icon">{verdict_ico}</span>
            <div class="verdict-label">{verdict_lbl}</div>
            <div class="verdict-conf">Disease probability: {ml_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Dynamic-color probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ml_prob,
            number={"suffix": "%", "font": {"size": 34, "family": "Syne, sans-serif", "color": gauge_color}},
            delta={"reference": 50, "increasing": {"color": "#ff3b5c"}, "decreasing": {"color": "#00e5a0"}},
            title={"text": "Disease Probability", "font": {"size": 12, "color": "#606880"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#2a2f48", "tickfont": {"size": 10, "color": "#606880"}},
                "bar":  {"color": gauge_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0,  33], "color": "rgba(0,229,160,0.07)"},
                    {"range": [33, 66], "color": "rgba(255,181,71,0.06)"},
                    {"range": [66,100], "color": "rgba(255,59,92,0.09)"},
                ],
                "threshold": {"line": {"color": "rgba(255,255,255,0.35)", "width": 2}, "thickness": 0.8, "value": 50},
            },
        ))
        fig_gauge.update_layout(height=220, **CHART_LAYOUT)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(
            f'<div class="confidence-label">{conf_label} — model is <strong>{conf_pct:.0f}%</strong> certain of this result</div>',
            unsafe_allow_html=True,
        )

    with col_exp:
        st.markdown('<div class="system-badge badge-expert">🧠 Expert System Rules</div>',
                    unsafe_allow_html=True)
        exp_cls = "verdict-high" if expert_high_risk else "verdict-low"
        exp_ico = "⚠️" if expert_high_risk else "🩺"
        exp_lbl = "HIGH RISK" if expert_high_risk else "LOW RISK"
        st.markdown(f"""
        <div class="verdict-card {exp_cls} hud-card">
            <span class="verdict-icon">{exp_ico}</span>
            <div class="verdict-label">{exp_lbl}</div>
            <div class="verdict-conf">{n_flagged} / {len(flags)} clinical rules triggered</div>
        </div>""", unsafe_allow_html=True)

        # Rules with explanations
        for msg, explain, flagged in flags:
            icon    = "🔴" if flagged else "🟢"
            css_cls = "rule-flag-item" if flagged else "rule-ok-item"
            explain_html = f'<div class="rule-explain">{explain}</div>' if flagged else ""
            st.markdown(
                f'<div class="rule-item {css_cls}">'
                f'<span class="rule-icon">{icon}</span>'
                f'<div><div class="rule-main">{msg}</div>{explain_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Combined verdict banner ───────────────────────────────────
    agree = (ml_pred == 1) == expert_high_risk
    if agree and ml_pred == 1:
        banner_cls = "banner-agree-high"
        banner_txt = "⚡  Both systems agree — HIGH RISK detected. Immediate cardiology consultation advised."
    elif agree:
        banner_cls = "banner-agree-low"
        banner_txt = "✅  Both systems agree — LOW RISK profile. Maintain healthy lifestyle habits."
    else:
        banner_cls = "banner-disagree"
        banner_txt = "⚠️  Systems disagree — Borderline case. Clinical evaluation by a cardiologist is recommended."

    st.markdown(f'<div class="verdict-banner {banner_cls}">{banner_txt}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ACTIONABLE INSIGHTS ───────────────────────────────────────
    insights = generate_insights(
        ml_pred, expert_high_risk, ml_prob,
        age, bp_raw, chol_raw, max_hr_raw, st_dep_raw,
        exercise_angina, chest_pain, major_vessels, thalassemia,
    )

    st.markdown('<div class="insights-section">', unsafe_allow_html=True)
    st.markdown('<div class="insights-header">💡 What Should You Do Now?</div>', unsafe_allow_html=True)
    for sev, icon, title, desc in insights:
        st.markdown(f"""
        <div class="insight-card {sev}">
            <div class="insight-icon">{icon}</div>
            <div>
                <div class="insight-title">{title}</div>
                <div class="insight-desc">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ANALYTICS DASHBOARD ───────────────────────────────────────
    st.markdown("""
    <p class="analytics-header">📊 Analytics Dashboard</p>
    <p class="analytics-sub">Your biomarkers benchmarked against the full patient dataset</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🕸  Radar Comparison", "📈  Distributions", "📋  Metric Breakdown"])

    # ── TAB 1: RADAR ─────────────────────────────────────────────
    with tab1:
        radar_labels = ["Age", "Blood Pressure", "Cholesterol", "Max Heart Rate", "ST Depression"]
        age_norm     = (age - 29) / (77 - 29)
        user_vals    = [age_norm, bp_scaled, chol_scaled, hr_scaled, st_scaled]
        avg_vals     = [
            (DATA_MEANS["Age"] - 29) / (77 - 29),
            DATA_MEANS["Resting_Blood_Pressure"],
            DATA_MEANS["Cholesterol"],
            DATA_MEANS["Maximum_Heart_Rate"],
            DATA_MEANS["ST_Depression"],
        ]
        # Normal reference (ideal values scaled)
        normal_vals = [
            (45 - 29) / (77 - 29),
            to_scaled("Resting_Blood_Pressure", 115),
            to_scaled("Cholesterol", 180),
            to_scaled("Maximum_Heart_Rate", 160),
            to_scaled("ST_Depression", 0.3),
        ]
        user_color     = "#ff3b5c" if ml_pred == 1 else "#00e5a0"
        user_fillcolor = "rgba(255,59,92,0.13)" if ml_pred == 1 else "rgba(0,229,160,0.11)"

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=normal_vals + [normal_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name="Healthy Reference",
            line=dict(color="#00d4ff", width=1.5, dash="dot"),
            fillcolor="rgba(0,212,255,0.05)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name="Dataset Average",
            line=dict(color="#4d9fff", width=1.8, dash="dash"),
            fillcolor="rgba(77,159,255,0.07)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name="Your Profile",
            line=dict(color=user_color, width=2.5),
            fillcolor=user_fillcolor,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.07)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=9, color="#606880")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.07)", linecolor="rgba(255,255,255,0.1)", tickfont=dict(size=11, color="#e8eaf6")),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True, height=430,
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(color="#e8eaf6")),
            **{k: v for k, v in CHART_LAYOUT.items() if k != "legend"},
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('<p style="text-align:center;color:#3a4060;font-size:0.74rem;">Values normalized 0–1 · Cyan = Healthy reference · Blue = Dataset avg · Colored = Your profile</p>', unsafe_allow_html=True)

    # ── TAB 2: DISTRIBUTIONS ─────────────────────────────────────
    with tab2:
        col_d1, col_d2 = st.columns(2, gap="medium")
        with col_d1:
            tc = df["Heart_Disease_Target"].value_counts().reset_index()
            tc.columns = ["Status", "Count"]
            tc["Status"] = tc["Status"].map({0: "No Disease", 1: "Heart Disease"})
            fig_donut = px.pie(tc, names="Status", values="Count", color="Status",
                               color_discrete_map={"Heart Disease": "#ff3b5c", "No Disease": "#00e5a0"}, hole=0.56)
            fig_donut.update_traces(textfont_size=12, marker=dict(line=dict(color="#06080f", width=3)))
            fig_donut.update_layout(
                title=dict(text="Dataset Risk Distribution", font=dict(color="#e8eaf6", size=13)),
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5, font=dict(color="#e8eaf6")),
                **{k: v for k, v in CHART_LAYOUT.items() if k != "legend"},
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_d2:
            fig_hist = px.histogram(df, x="Age", color="Heart_Disease_Target", barmode="overlay",
                                    nbins=25, color_discrete_map={0: "#00e5a0", 1: "#ff3b5c"},
                                    labels={"Heart_Disease_Target": "Disease", "Age": "Age (years)"}, opacity=0.72)
            fig_hist.add_vline(x=age, line_dash="dash", line_color="rgba(255,255,255,0.55)", line_width=1.5,
                               annotation_text=f"You ({age})", annotation_font=dict(color="#e8eaf6", size=11), annotation_position="top right")
            fig_hist.update_layout(
                title=dict(text="Age Distribution by Risk Group", font=dict(color="#e8eaf6", size=13)),
                height=300, xaxis={**GRID_STYLE}, yaxis={**GRID_STYLE},
                legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5, font=dict(color="#e8eaf6")),
                **{k: v for k, v in CHART_LAYOUT.items() if k != "legend"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── TAB 3: BAR COMPARISON with normal reference lines ─────────
    with tab3:
        metrics_raw = {
            "Blood Pressure\n(mmHg)":  (bp_raw,         to_raw("Resting_Blood_Pressure", DATA_MEANS["Resting_Blood_Pressure"]),  120),
            "Cholesterol\n(mg/dL)":    (chol_raw,        to_raw("Cholesterol",            DATA_MEANS["Cholesterol"]),             200),
            "Max Heart Rate\n(bpm)":   (max_hr_raw,      to_raw("Maximum_Heart_Rate",     DATA_MEANS["Maximum_Heart_Rate"]),      150),
            "ST Depression\n(mm×10)":  (st_dep_raw * 10, to_raw("ST_Depression",          DATA_MEANS["ST_Depression"]) * 10,      10),
        }
        bar_color = "#ff3b5c" if ml_pred == 1 else "#00e5a0"
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Your Values", x=list(metrics_raw.keys()), y=[v[0] for v in metrics_raw.values()],
                                 marker=dict(color=bar_color, opacity=0.85, line=dict(color=bar_color, width=1))))
        fig_bar.add_trace(go.Bar(name="Dataset Average", x=list(metrics_raw.keys()), y=[v[1] for v in metrics_raw.values()],
                                 marker=dict(color="rgba(77,159,255,0.6)", opacity=0.75, line=dict(color="#4d9fff", width=1))))
        # Add normal reference scatter (dots + line)
        fig_bar.add_trace(go.Scatter(
            name="Healthy Normal", mode="markers+lines",
            x=list(metrics_raw.keys()), y=[v[2] for v in metrics_raw.values()],
            marker=dict(color="#00d4ff", size=9, symbol="diamond", line=dict(color="#fff", width=1)),
            line=dict(color="#00d4ff", width=1.5, dash="dot"),
        ))
        fig_bar.update_layout(
            barmode="group",
            title=dict(text="Your Metrics vs. Dataset Average vs. Healthy Normal", font=dict(color="#e8eaf6", size=13)),
            height=400, bargap=0.24, bargroupgap=0.08,
            xaxis=dict(**{**GRID_STYLE, "tickfont": dict(size=11, color="#e8eaf6")}),
            yaxis=dict(**GRID_STYLE),
            legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5, font=dict(color="#e8eaf6")),
            **{k: v for k, v in CHART_LAYOUT.items() if k != "legend"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── PDF REPORT SECTION ────────────────────────────────────────
    st.markdown("---")
    report_html = generate_pdf_report(
        age, sex, bp_raw, chol_raw, max_hr_raw, st_dep_raw,
        chest_pain, ecg, exercise_angina, st_slope,
        major_vessels, thalassemia,
        ml_pred, ml_prob, expert_high_risk, n_flagged,
        flags, insights,
    )
    st.markdown("""
    <div class="pdf-section">
        <div class="pdf-info">
            <div class="pdf-title">📄 Download Full Report</div>
            <div class="pdf-sub">Complete cardiac risk report with patient profile, rule breakdown, and recommendations — ready to print as PDF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(get_download_link(report_html), unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚕️ CardioSense is an academic demonstration tool only.<br>
        It does <strong>not</strong> constitute medical advice, diagnosis, or treatment.
        Always consult a qualified cardiologist for clinical decisions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LANDING STATE
# ══════════════════════════════════════════════════════════════════
else:
    pct_disease = df["Heart_Disease_Target"].mean() * 100
    avg_age     = df["Age"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{len(df):,}",         "Patient Records"),
        (c2, f"{pct_disease:.1f}%",  "Disease Prevalence"),
        (c3, f"{avg_age:.0f} yrs",   "Avg Patient Age"),
        (c4, f"{len(FEATURE_COLS)}", "Clinical Features"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:\'Syne\',sans-serif;font-size:1.55rem;font-weight:700;color:#e8eaf6;margin-bottom:1.1rem;">How CardioSense Works</h2>', unsafe_allow_html=True)

    hw1, hw2, hw3 = st.columns(3, gap="medium")
    for col, num, color, title, desc in [
        (hw1, "01", "#ff3b5c", "Smart Input Form",
         "Enter your data step-by-step with real-time validation hints showing normal ranges for each measurement."),
        (hw2, "02", "#4d9fff", "Dual-Engine Analysis",
         "A Decision Tree model predicts risk probability while 12 expert clinical rules are evaluated with full explanations."),
        (hw3, "03", "#00e5a0", "Report & Recommendations",
         "Receive a full verdict, personalized action plan, charts, and a downloadable PDF report."),
    ]:
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="how-num" style="color:{color}">{num}</div>
                <div class="how-title">{title}</div>
                <div class="how-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:\'Syne\',sans-serif;font-size:1.4rem;font-weight:700;color:#e8eaf6;margin-bottom:0.8rem;">📂 Dataset Snapshot</h2>', unsafe_allow_html=True)
    st.dataframe(df.rename(columns={"Heart_Disease_Target": "Target"}).head(8), use_container_width=True, height=280)
    st.markdown("""
    <p style="text-align:center;color:#3a4060;font-size:0.78rem;margin-top:0.5rem;">
        <span class="pulse-dot"></span>
        Fill in the 3-step form in the sidebar then press
        <strong style="color:#ff3b5c;">⚡ Run Cardiac Analysis</strong>
    </p>""", unsafe_allow_html=True)