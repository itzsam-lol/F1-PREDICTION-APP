import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
import time

from src.prediction_engine import F1PredictionEngine
from src.data_collection import (
    DRIVERS_2026, CALENDAR_2026, TEAM_COLORS, TEAM_CAR_RATINGS,
    CIRCUIT_TYPE_MULTIPLIERS
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 2026 — Season Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Space Grotesk', sans-serif; }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070710 0%, #0d0d1a 60%, #0a0a12 100%) !important;
        border-right: 1px solid rgba(255,30,0,0.12) !important;
        box-shadow: 4px 0 30px rgba(0,0,0,0.6) !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.15rem !important; flex-direction: column !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important; align-items: center !important;
        padding: 0.6rem 1rem !important; border-radius: 12px !important;
        cursor: pointer !important; transition: all 0.25s ease !important;
        color: #888 !important; font-size: 0.88rem !important; font-weight: 500 !important;
        border: 1px solid transparent !important; margin: 1px 0 !important; width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,30,0,0.06) !important; color: #fff !important;
        border-color: rgba(255,30,0,0.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }

    /* ── Hero banner ─────────────────────────────────────── */
    .f1-hero {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%);
        border: 1px solid rgba(255,255,255,0.05); border-radius: 24px; padding: 3rem;
        margin-bottom: 2rem; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.4); position: relative;
    }
    .f1-hero h1 {
        font-family: 'Rajdhani', monospace; font-size: 3.5rem; font-weight: 700; letter-spacing: 0.05em;
        background: linear-gradient(90deg, #ff1801, #ff8700);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .f1-hero p { color: #888; margin-top: 1rem; font-size: 1.1rem; }

    /* ── Section header ──────────────────────────────────── */
    .section-header {
        font-family: 'Rajdhani', monospace; font-size: 1.4rem; font-weight: 700;
        color: #fff; border-left: 4px solid #ff1801; border-radius: 2px;
        padding-left: 0.8rem; margin: 2rem 0 1.2rem 0;
        text-transform: uppercase; letter-spacing: 0.1em;
    }

    /* ── Metric card ─────────────────────────────────────── */
    .metric-card {
        background: rgba(25,25,40,0.6); backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
        padding: 1.5rem; transition: all 0.3s cubic-bezier(0.4,0,0.2,1); text-align: center;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #ff1e00; box-shadow: 0 15px 35px rgba(255,30,0,0.15); }
    .metric-card .val { font-family: 'Rajdhani', monospace; font-size: 2.8rem; font-weight: 700; color: #ff1801; }
    .metric-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; }

    /* ── Driver card ─────────────────────────────────────── */
    .driver-card {
        background: rgba(30,30,45,0.5); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 20px;
        padding: 1.2rem; margin: 0.5rem 0; transition: all 0.3s;
    }
    .driver-card:hover { border-color: rgba(255,255,255,0.15); transform: scale(1.02); }

    /* ── Race expander cards ─────────────────────────────── */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 16px !important; background: rgba(255,255,255,0.02) !important;
        margin-bottom: 0.5rem !important; transition: all 0.25s ease !important;
        overflow: hidden !important;
    }
    .stExpander:hover {
        border-color: rgba(255,30,0,0.3) !important;
        background: rgba(255,30,0,0.03) !important;
        box-shadow: 0 4px 24px rgba(255,30,0,0.07) !important;
    }
    .stExpander summary { padding: 0.5rem 1rem !important; cursor: pointer !important; }

    /* ── Race header inside expander ─────────────────────── */
    .race-header-row { display: flex; align-items: center; gap: 1rem; width: 100%; padding: 0.4rem 0; }
    .race-rnd {
        background: linear-gradient(135deg, #ff1e00, #ff6b00); color: #fff;
        font-family: 'Rajdhani'; font-size: 0.7rem; font-weight: 900;
        padding: 0.3rem 0.55rem; border-radius: 8px; letter-spacing: 0.08em;
        min-width: 36px; text-align: center; flex-shrink: 0;
    }
    .race-flag { font-size: 1.5rem; flex-shrink: 0; line-height: 1; }
    .race-name-block { flex: 1; min-width: 0; }
    .race-name-txt { font-weight: 700; color: #fff; font-size: 0.97rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .race-circuit-txt { font-size: 0.74rem; color: #555; margin-top: 2px; }
    .race-meta-block { text-align: right; flex-shrink: 0; }
    .race-date-txt { font-size: 0.8rem; color: #bbb; font-weight: 500; }
    .race-type-pill {
        display: inline-block; font-size: 0.6rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
        padding: 0.15rem 0.55rem; border-radius: 999px; margin-top: 0.3rem;
        border: 1px solid currentColor; opacity: 0.9;
    }
    .sprint-badge {
        display: inline-block; background: linear-gradient(90deg,#9b30ff,#6600cc);
        color: #fff; font-size: 0.55rem; font-weight: 800; letter-spacing: 0.1em;
        padding: 0.15rem 0.45rem; border-radius: 5px; margin-left: 0.4rem;
        vertical-align: middle; text-transform: uppercase;
    }
    .race-sprint-badge {
        display:inline-block;background:linear-gradient(90deg,#9b30ff,#6600cc);
        color:#fff;font-size:0.55rem;font-weight:800;letter-spacing:0.1em;
        padding:0.15rem 0.45rem;border-radius:5px;margin-left:0.4rem;
        vertical-align:middle;text-transform:uppercase;
    }

    /* ── Track detail panel ──────────────────────────────── */
    .track-stat-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.8rem;
    }
    .track-stat {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 0.6rem 0.8rem;
    }
    .track-stat-label { font-size: 0.6rem; color: #555; text-transform: uppercase; letter-spacing: 0.08em; }
    .track-stat-value { font-size: 0.88rem; color: #ddd; font-weight: 600; margin-top: 0.15rem; }

    /* ── Sidebar stat card ───────────────────────────────── */
    .sb-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px; padding: 0.85rem 1rem; margin: 0.3rem 0;
    }
    .sb-card-label { font-size: 0.58rem; color: #444; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.2rem; }
    .sb-card-value { font-family: 'Rajdhani'; font-size: 1rem; font-weight: 900; color: #fff; }
    .sb-card-sub { font-size: 0.68rem; color: #555; margin-top: 0.15rem; }

    /* ── Podium ──────────────────────────────────────────── */
    .podium-p1 { background: rgba(255,215,0,0.05); border: 2px solid #FFD700; border-radius: 24px; padding: 2rem; text-align: center; }
    .podium-p2 { background: rgba(192,192,192,0.05); border: 1px solid #C0C0C0; border-radius: 20px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
    .podium-p3 { background: rgba(205,127,50,0.03); border: 1px solid #CD7F32; border-radius: 20px; padding: 1.5rem; text-align: center; margin-top: 2rem; }

    /* ── Reg / countdown / misc ──────────────────────────── */
    .reg-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 1.5rem; height: 100%; transition: all 0.3s; }
    .reg-card:hover { background: rgba(255,255,255,0.06); }
    .countdown-box { background: rgba(255,30,0,0.03); border: 1px solid #ff1e0033; border-radius: 24px; padding: 2rem; text-align: center; }
    .countdown-num { font-family: 'Rajdhani'; font-size: 3.5rem; font-weight: 900; color: #fff; filter: drop_shadow(0 0 15px rgba(255,30,0,0.4)); }
    .countdown-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 0.2em; margin-left: 0.4rem; vertical-align: super; }
    .stDataFrame { border-radius: 20px !important; }
    .stButton>button { border-radius: 12px !important; transition: all 0.3s; font-family: 'Rajdhani'; }
    .stSelectbox label, .stMultiSelect label { font-family: 'Space Grotesk'; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.7rem !important; color: #555 !important; }
</style>
""", unsafe_allow_html=True)

# ─── ICONS ───────────────────────────────────────────────────────────────────
def get_icon(name: str, size: int = 24, color: str = "currentColor", margin_right: int = 10) -> str:
    """Helper to return SVG icons as HTML string."""
    icons = {
        "home":        '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline>',
        "calendar":    '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line>',
        "predictor":   '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
        "trophy":      '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path><path d="M4 22h16"></path><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path>',
        "user":        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle>',
        "users":       '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 1-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path>',
        "analytics":   '<line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line>',
        "sparkles":    '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>',
        "check":       '<polyline points="20 6 9 17 4 12"></polyline>',
        "zap":         '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>',
        "activity":    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
        "award":       '<circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline>',
        "shield":      '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>',
        "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline>',
        "map-pin":     '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle>',
        "cpu":         '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line>',
        "clock":       '<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>',
        "layers":      '<polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline>',
        "settings":    '<circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>',
        "building":    '<rect x="2" y="7" width="20" height="15" rx="2" ry="2"></rect><polyline points="17 2 12 7 7 2"></polyline><line x1="12" y1="22" x2="12" y2="11"></line><path d="M7 11h2v2H7zM15 11h2v2H15zM7 15h2v2H7zM15 15h2v2H15z"></path>',
        "flag":        '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"></path><line x1="4" y1="22" x2="4" y2="15"></line>',
        "grid":        '<rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect>',
        "star":        '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>',
        "list":        '<line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line>',
        "refresh":     '<polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>',
        "bar-chart":   '<line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line>',
        "compass":     '<circle cx="12" cy="12" r="10"></circle><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon>',
        "race-flag":   '<rect x="2" y="2" width="5" height="5" fill="currentColor" rx="0.5"></rect><rect x="9" y="2" width="5" height="5" fill="none" stroke="currentColor"></rect><rect x="2" y="9" width="5" height="5" fill="none" stroke="currentColor"></rect><rect x="9" y="9" width="5" height="5" fill="currentColor" rx="0.5"></rect><line x1="2" y1="22" x2="2" y2="2"></line>',
    }
    path = icons.get(name, "")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:{margin_right}px;flex-shrink:0">{path}</svg>'

def flag_pill(iso: str, color: str = "#555") -> str:
    """Return a small styled ISO country-code badge for use in HTML."""
    return (f"<span style='display:inline-flex;align-items:center;justify-content:center;"
            f"background:rgba(255,255,255,0.07);color:{color};font-size:0.58rem;"
            f"font-weight:700;letter-spacing:0.06em;border:1px solid rgba(255,255,255,0.15);"
            f"border-radius:4px;padding:0.15rem 0.4rem;font-family:monospace;"
            f"min-width:1.8rem;text-align:center'>{iso}</span>")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
OPENING_RACE_DATE = date(2026, 3, 8)   # Australian GP

F1_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1280px-F1.svg.png"

TEAM_COLORS_HEX = TEAM_COLORS

# ISO-2 → country name lookup (used for display in predictor page)
COUNTRY_NAMES = {
    "AU": "Australia", "CN": "China",    "JP": "Japan",        "BH": "Bahrain",
    "SA": "Saudi Arabia", "US": "USA",   "IT": "Italy",        "MC": "Monaco",
    "ES": "Spain",    "CA": "Canada",    "AT": "Austria",       "GB": "UK",
    "HU": "Hungary",  "BE": "Belgium",   "NL": "Netherlands",   "AZ": "Azerbaijan",
    "SG": "Singapore","MX": "Mexico",   "BR": "Brazil",        "QA": "Qatar",
    "AE": "UAE",
}

# ─── ENGINE ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    return F1PredictionEngine()

# ─── PLOTLY DARK THEME ────────────────────────────────────────────────────────
def _hex_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a 6-digit hex + alpha float to 'rgba(r,g,b,a)' for Plotly."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color  # fallback if already rgba/named

def dark_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Rajdhani", size=16, color="#fff")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#aaa"),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc")),
        margin=dict(t=50, b=40, l=40, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    )
    return fig

# ── Driver headshot CDN URLs ─────────────────────────────────────────────────
DRIVER_HEADSHOTS = {
    "VER": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/verstappen.png",
    "LEC": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/leclerc.png",
    "HAM": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/hamilton.png",
    "NOR": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/norris.png",
    "PIA": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/piastri.png",
    "RUS": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/russell.png",
    "ALO": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/alonso.png",
    "SAI": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/sainz.png",
    "GAS": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/gasly.png",
    "ALB": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/albon.png",
    "TSU": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/tsunoda.png",
    "HUL": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/hulkenberg.png",
    "STR": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/stroll.png",
    "OCO": "https://media.formula1.com/image/upload/f_auto/q_auto/v0/fom-website/drivers/2024/ocon.png",
}

CIRCUIT_TYPE_ICON = {
    "street":    ("map-pin",  "#ff6b00"),
    "technical": ("settings", "#27F4D2"),
    "power":     ("zap",      "#FFD700"),
    "high_speed":("activity", "#ff1e00"),
}

# ─── CIRCUIT DETAILS ──────────────────────────────────────────────────────────
CIRCUIT_DETAILS = {
    "Albert Park":  {"length": 5.278, "lap_record": "1:20.235 (Leclerc, 2022)", "drs_zones": 3, "first_gp": 1996, "key_corner": "Turn 9-10 chicane", "characteristic": "Semi-street, fast flowing layout through Melbourne park"},
    "Shanghai":     {"length": 5.451, "lap_record": "1:32.238 (M.Schumacher, 2004)", "drs_zones": 2, "first_gp": 2004, "key_corner": "Turn 1-2 hairpin", "characteristic": "Long back straight, demanding hairpin complex"},
    "Suzuka":       {"length": 5.807, "lap_record": "1:30.983 (Hamilton, 2019)",   "drs_zones": 2, "first_gp": 1987, "key_corner": "130R / Casio chicane", "characteristic": "Figure-8 layout, high-speed esses, classic drivers' circuit"},
    "Bahrain":      {"length": 5.412, "lap_record": "1:31.447 (De la Rosa, 2005)", "drs_zones": 3, "first_gp": 2004, "key_corner": "Turn 4 hairpin",      "characteristic": "Power unit circuit, desert surface, lots of tyre degradation"},
    "Jeddah":       {"length": 6.174, "lap_record": "1:30.734 (Hamilton, 2021)",   "drs_zones": 3, "first_gp": 2021, "key_corner": "Turns 22-27 complex",  "characteristic": "Fastest street circuit, walls everywhere, no margin for error"},
    "Miami":        {"length": 5.412, "lap_record": "1:29.708 (Verstappen, 2023)", "drs_zones": 3, "first_gp": 2022, "key_corner": "Turn 17 tight hairpin", "characteristic": "Contemporary street circuit, high-energy atmosphere"},
    "Imola":        {"length": 4.909, "lap_record": "1:15.484 (Verstappen, 2022)", "drs_zones": 2, "first_gp": 1980, "key_corner": "Acque Minerali",       "characteristic": "Historic track, narrow, limited overtaking"},
    "Monaco":        {"length": 3.337, "lap_record": "1:12.909 (Leclerc, 2021)",   "drs_zones": 1, "first_gp": 1950, "key_corner": "Grand Hotel hairpin",   "characteristic": "Tightest circuit in F1, qualifying defines the race"},
    "Barcelona":    {"length": 4.657, "lap_record": "1:18.149 (Verstappen, 2022)", "drs_zones": 2, "first_gp": 1991, "key_corner": "Turn 3 (Renault)",     "characteristic": "Balanced layout, tyre management critical"},
    "Montreal":     {"length": 4.361, "lap_record": "1:13.078 (Bottas, 2019)",     "drs_zones": 2, "first_gp": 1978, "key_corner": "Wall of Champions chicane","characteristic": "Power circuit, low downforce, high safety car risk"},
    "Red Bull Ring":{"length": 4.318, "lap_record": "1:05.619 (Leclerc, 2020)",   "drs_zones": 3, "first_gp": 1970, "key_corner": "Turn 3 ascent",         "characteristic": "Short lap, lots of elevation change, clean air crucial"},
    "Silverstone":  {"length": 5.891, "lap_record": "1:27.097 (Hamilton, 2020)",   "drs_zones": 2, "first_gp": 1950, "key_corner": "Maggots-Becketts-Chapel","characteristic": "Home of British motorsport, ultra-fast flowing corners"},
    "Hungaroring":  {"length": 4.381, "lap_record": "1:16.627 (Hamilton, 2020)",   "drs_zones": 2, "first_gp": 1986, "key_corner": "Turn 4 (fast left)",   "characteristic": "Monaco without walls — tight, technical, limited DRS effect"},
    "Spa":          {"length": 7.004, "lap_record": "1:46.286 (Bottas, 2018)",     "drs_zones": 2, "first_gp": 1950, "key_corner": "Eau Rouge / Raidillon", "characteristic": "Greatest circuit in the world, unpredictable weather"},
    "Zandvoort":    {"length": 4.259, "lap_record": "1:11.097 (Verstappen, 2021)", "drs_zones": 2, "first_gp": 1952, "key_corner": "Tarzanbocht hairpin",  "characteristic": "Banked corners, tight, designed for close racing"},
    "Monza":         {"length": 5.793, "lap_record": "1:21.046 (Barrichello, 2004)","drs_zones": 2, "first_gp": 1950, "key_corner": "Parabolica / Variante Ascari","characteristic": "Temple of speed, lowest downforce circuit"},
    "Baku":          {"length": 6.003, "lap_record": "1:43.009 (Leclerc, 2019)",   "drs_zones": 2, "first_gp": 2016, "key_corner": "Turn 8 (castle hairpin)","characteristic": "Long straight, street circuit, chaotic safety cars"},
    "Singapore":     {"length": 4.940, "lap_record": "1:35.867 (Santos, 2023)",     "drs_zones": 3, "first_gp": 2008, "key_corner": "Turn 5 (Raffles Blvd)", "characteristic": "Night race, hottest/most humid circuit, most corners"},
    "Austin":        {"length": 5.513, "lap_record": "1:36.169 (Leclerc, 2019)",   "drs_zones": 2, "first_gp": 2012, "key_corner": "Turn 1 (blind crest)",  "characteristic": "Technical Hermann Tilke design, challenging elevation change"},
    "Mexico City":   {"length": 4.304, "lap_record": "1:17.774 (Bottas, 2021)",    "drs_zones": 3, "first_gp": 1963, "key_corner": "Estadio section",        "characteristic": "Highest altitude circuit (2285m), massive straight, low aero"},
    "Space Grotesklagos":    {"length": 4.309, "lap_record": "1:10.540 (Rubens, 2004)",    "drs_zones": 2, "first_gp": 1973, "key_corner": "Senna S",                "characteristic": "Anti-clockwise, undulating, unpredictable weather"},
    "Las Vegas":     {"length": 6.201, "lap_record": "1:35.490 (Leclerc, 2023)",   "drs_zones": 3, "first_gp": 2023, "key_corner": "Turn 14 hairpin (Apex)", "characteristic": "Strip circuit, nighttime glamour, fast straights"},
    "Lusail":       {"length": 5.380, "lap_record": "1:24.319 (Verstappen, 2023)", "drs_zones": 2, "first_gp": 2021, "key_corner": "Turn 1 braking zone",   "characteristic": "High-speed flowing layout, massive tyre degradation"},
    "Yas Marina":   {"length": 5.281, "lap_record": "1:26.103 (Verstappen, 2021)", "drs_zones": 2, "first_gp": 2009, "key_corner": "Turn 6 (marina section)","characteristic": "Season finale, mixed day/night race, Abu Dhabi"},
}


# ─── CIRCUIT GPS COORDINATES ─────────────────────────────────────────────────
# (lat, lon, osm_zoom) — used for embedded OpenStreetMap thumbnails
CIRCUIT_COORDS = {
    "Albert Park":  (-37.8497,  144.9680, 14),
    "Shanghai":     ( 31.3389,  121.2197, 14),
    "Suzuka":       ( 34.8431,  136.5400, 14),
    "Bahrain":      ( 26.0325,   50.5106, 14),
    "Jeddah":       ( 21.6319,   39.1044, 14),
    "Miami":        ( 25.9581,  -80.2389, 14),
    "Imola":        ( 44.3439,   11.7167, 14),
    "Monaco":       ( 43.7347,    7.4206, 15),
    "Barcelona":    ( 41.5700,    2.2611, 14),
    "Montreal":     ( 45.5000,  -73.5228, 14),
    "Red Bull Ring":( 47.2197,   14.7647, 14),
    "Silverstone":  ( 52.0786,   -1.0169, 14),
    "Hungaroring":  ( 47.5830,   19.2511, 14),
    "Spa":          ( 50.4372,    5.9714, 13),
    "Zandvoort":    ( 52.3888,    4.5409, 14),
    "Monza":        ( 45.6156,    9.2811, 14),
    "Baku":         ( 40.3725,   49.8533, 14),
    "Singapore":    (  1.2914,  103.8644, 15),
    "Austin":       ( 30.1328,  -97.6411, 14),
    "Mexico City":  ( 19.4042,  -99.0907, 14),
    "Space Grotesklagos":   (-23.7036,  -46.6997, 14),
    "Las Vegas":    ( 36.1147, -115.1728, 14),
    "Lusail":       ( 25.4900,   51.4542, 14),
    "Yas Marina":   ( 24.4672,   54.6031, 14),
}

def get_map_embed(circuit_name: str, height: int = 185) -> str:
    """Return an OpenStreetMap iframe centred on the circuit. No API key needed."""
    coords = CIRCUIT_COORDS.get(circuit_name)
    if not coords:
        return ""
    lat, lon, zoom = coords
    delta = 0.013 * (2 ** (14 - zoom))
    bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
    src = (
        f"https://www.openstreetmap.org/export/embed.html"
        f"?bbox={bbox}&layer=mapnik&marker={lat},{lon}"
    )
    return (
        f"<div style='border-radius:12px;overflow:hidden;"
        f"border:1px solid rgba(255,255,255,0.10);"
        f"box-shadow:0 6px 20px rgba(0,0,0,0.55);position:relative'>"
        f"<iframe src='{src}' width='100%' height='{height}px' "
        f"style='border:none;display:block;"
        f"filter:brightness(0.88) saturate(0.78) contrast(1.08)' "
        f"loading='lazy' title='{circuit_name} map'></iframe>"
        f"<div style='position:absolute;bottom:4px;right:6px;font-size:0.46rem;"
        f"color:#999;background:rgba(0,0,0,0.6);padding:1px 5px;border-radius:3px;"
        f"pointer-events:none'>© OpenStreetMap</div></div>"
    )

# ─── RACE WEEKEND SCHEDULE TEMPLATES ─────────────────────────────────────────
_NORMAL_WKD = [
    ("FRI", "Practice 1",          "#4a4a5a"),
    ("FRI", "Practice 2",          "#4a4a5a"),
    ("SAT", "Practice 3",          "#4a4a5a"),
    ("SAT", "Qualifying",          "#ff6b00"),
    ("SUN", "Race",                "#ff1e00"),
]
_SPRINT_WKD = [
    ("FRI", "Practice 1",               "#4a4a5a"),
    ("FRI", "Sprint Qualifying",         "#9b30ff"),
    ("SAT", "Sprint Race",               "#9b30ff"),
    ("SAT", "Qualifying",               "#ff6b00"),
    ("SUN", "Race",                     "#ff1e00"),
]

# --- TRACK SVG MAPS -------------------------------------------------------------------
# Accurate circuit fingerprint SVGs -- each shaped like its real-world layout
import base64
import os

def get_track_svg(circuit_name: str, color: str = "#ff1e00") -> str:
    mapping = {
        "Albert Park": "albert_park.webp",
        "Shanghai": "shanghai.webp",
        "Suzuka": "suzuka.webp",
        "Bahrain": "bahrain.webp",
        "Jeddah": "jeddah.webp",
        "Miami": "miami.webp",
        "Imola": "madrid.webp", # Imola -> Madrid fallback
        "Monaco": "monaco.webp",
        "Barcelona": "catalunya.webp",
        "Montreal": "villeneuve.webp",
        "Red Bull Ring": "red_bull_ring.webp",
        "Silverstone": "silverstone.webp",
        "Hungaroring": "hungaroring.webp",
        "Spa": "spa.webp",
        "Zandvoort": "zandvoort.webp",
        "Monza": "monza.webp",
        "Baku": "baku.webp",
        "Singapore": "marina_bay.webp",
        "Austin": "americas.webp",
        "Mexico City": "rodriguez.webp",
        "Space Grotesklagos": "interlagos.webp",
        "Las Vegas": "vegas.webp",
        "Lusail": "losail.webp",
        "Yas Marina": "yas_marina.webp"
    }
    
    filename = mapping.get(circuit_name)
    if not filename:
        return f"<div style='color:{color};text-align:center;'>Image not found</div>"
    
    path = os.path.join("data", "maps", filename)
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:image/webp;base64,{encoded}" style="width:100%;max-width:340px;filter:drop-shadow(0 0 16px {color}44)" alt="{circuit_name}"/>'
    except Exception as e:
        return f"<!-- Error loading image {path}: {e} -->"



# ─── FOOTER ──────────────────────────────────────────────────────────────────
def show_footer():
    st.markdown(f"""
    <hr style='border:none; border-top:1px solid rgba(255,255,255,0.08); margin: 3rem 0 1rem 0'>
    <div style='text-align:center; padding: 1rem 0 2rem 0; opacity: 0.7'>
        <img src='{F1_LOGO_URL}' style='width:120px; margin-bottom:1rem; opacity:0.8; filter: drop-shadow(0 0 10px rgba(255,255,255,0.1))'>
        <div style='font-size:0.75rem; color:#666; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:0.6rem'>
            2026 Season Intelligence &middot; Powered by ML
        </div>
        <div style='font-size:0.65rem; color:#444; max-width:600px; margin:0 auto'>
            Predictions are model-generated simulations for entertainment purposes only.
            This application is not affiliated with the Formula 1 group of companies, the FIA, or any F1 team.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
def render_sidebar(engine):
    with st.sidebar:
        # ── Header ──────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='background:linear-gradient(180deg,rgba(255,30,0,0.10),transparent);
                    padding:1.6rem 1rem 1rem 1rem; margin:-1rem -1rem 0 -1rem;
                    border-bottom:1px solid rgba(255,255,255,0.05); text-align:center'>
            <img src='{F1_LOGO_URL}'
                 style='width:100px; filter:drop-shadow(0 0 18px rgba(255,30,0,0.5)) brightness(1.1)'>
            <div style='margin-top:0.6rem; font-family:"Rajdhani",monospace; font-size:0.5rem;
                        font-weight:700; letter-spacing:0.35em; color:#ff1e00;
                        text-transform:uppercase'>2026 Intelligence Hub</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        # Nav items: (label, icon_name)
        NAV_ITEMS = [
            ("Home",             "home"),
            ("2026 Calendar",    "calendar"),
            ("Race Predictor",   "activity"),
            ("Season Simulator", "trophy"),
            ("Driver Profiles",  "user"),
            ("Analytics",        "bar-chart"),
        ]
        nav_labels = [k for k, _ in NAV_ITEMS]

        page_idx = st.radio(
            "Select Page", nav_labels, label_visibility="collapsed", index=0
        )
        page_choice = page_idx   # already the plain key

        # ── Divider ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style='margin:0.8rem 0; border-top:1px solid rgba(255,255,255,0.05)'></div>
        """, unsafe_allow_html=True)

        # ── Countdown card ───────────────────────────────────────────────────
        days_left = (OPENING_RACE_DATE - date.today()).days
        if days_left > 0:
            weeks, rem_days = divmod(days_left, 7)
            st.markdown(f"""
            <div class='sb-card'>
                <div class='sb-card-label'>{get_icon('clock', 12, '#444', 4)} Season Countdown</div>
                <div style='display:flex;align-items:baseline;gap:0.5rem;margin-top:0.3rem'>
                    <span style='font-family:Rajdhani;font-size:2rem;font-weight:900;
                                 color:#ff1e00;text-shadow:0 0 20px rgba(255,30,0,0.4)'>
                        {days_left}
                    </span>
                    <span style='font-size:0.6rem;color:#444;letter-spacing:0.1em'>DAYS</span>
                </div>
                <div class='sb-card-sub'>TO MELBOURNE · MARCH 8</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='sb-card' style='border-color:rgba(0,255,136,0.3)'>
                <div class='sb-card-label'>{get_icon('race-flag', 12, '#00ff88', 4)} Season Status</div>
                <div style='color:#00ff88;font-family:Rajdhani;font-size:0.8rem;font-weight:700;margin-top:0.3rem'>
                    LIVE SEASON ACTIVE
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Next race mini-card ───────────────────────────────────────────────
        today_ts = pd.Timestamp(date.today())
        upcoming = [r for r in CALENDAR_2026 if pd.Timestamp(r["date"]) >= today_ts]
        if upcoming:
            nr = upcoming[0]
            ct_color_map = {"power":"#ff6b00","high_speed":"#ff1e00","street":"#FF87BC","technical":"#27F4D2","balanced":"#aaa"}
            nr_color = ct_color_map.get(nr["circuit_type"], "#888")
            sprint_txt = " · <span style='color:#9b30ff;font-weight:700'>SPRINT</span>" if nr["sprint"] else ""
            fp = flag_pill(nr["flag"], "#888")
            st.markdown(f"""
            <div class='sb-card' style='border-color:rgba(255,255,255,0.07)'>
                <div class='sb-card-label'>{get_icon('compass', 12, '#444', 4)} Next Round · R{nr["round"]}</div>
                <div style='display:flex;align-items:center;gap:0.4rem;margin-top:0.3rem'>
                    {fp}
                    <span style='font-weight:700;color:#fff;font-size:0.86rem'>{nr["name"]}</span>
                </div>
                <div style='font-size:0.7rem;color:#444;margin-top:0.2rem'>
                    {nr["circuit"]} · <span style='color:{nr_color}'>{nr["circuit_type"].replace("_"," ").title()}</span>{sprint_txt}
                </div>
                <div style='font-size:0.68rem;color:#666;margin-top:0.22rem'>
                    {pd.Timestamp(nr["date"]).strftime("%b %d, %Y")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Divider ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style='margin:0.6rem 0; border-top:1px solid rgba(255,255,255,0.05)'></div>
        """, unsafe_allow_html=True)

        # ── Model status ──────────────────────────────────────────────────────
        st.markdown(f"""
        <div style='font-size:0.55rem;color:#2a2a3a;text-transform:uppercase;
                    letter-spacing:0.18em;padding:0 0.2rem 0.4rem 0.2rem;font-weight:700'>
            {get_icon('cpu', 11, '#2a2a3a', 4)}ML Engine
        </div>""", unsafe_allow_html=True)

        if not engine.is_trained:
            st.markdown(f"""
            <div class='sb-card' style='border-color:rgba(255,30,0,0.2)'>
                <div style='display:flex;align-items:center;gap:0.5rem'>
                    <div style='width:8px;height:8px;border-radius:50%;background:#ff1e00;
                                box-shadow:0 0 8px #ff1e00'></div>
                    <span style='font-size:0.72rem;color:#ff6b00;font-weight:600'>MODEL OFFLINE</span>
                </div>
                <div style='font-size:0.62rem;color:#333;margin-top:0.3rem'>
                    Initialize to enable predictions
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Initialize & Sync Live Data", use_container_width=True):
                with st.spinner("Fetching Live Data (may take a moment) & Training..."):
                    from src.live_data import LiveDataIngestor
                    ingestor = LiveDataIngestor(year=2026)
                    ingestor.fetch_latest_data()
                    engine.quick_demo_train()
                st.rerun()
        else:
            best = engine.ml_models.best_model_name
            res  = engine.ml_models.training_results
            mae  = res.get(best, {}).get("mae", "-")
            r2   = res.get(best, {}).get("r2",  "-")
            st.markdown(f"""
            <div class='sb-card' style='border-color:rgba(0,255,136,0.2)'>
                <div style='display:flex;align-items:center;gap:0.5rem'>
                    <div style='width:8px;height:8px;border-radius:50%;background:#00ff88;
                                box-shadow:0 0 8px #00ff88'></div>
                    <span style='font-size:0.72rem;color:#00ff88;font-weight:700;font-family:Rajdhani'>ACTIVE</span>
                </div>
                <div style='font-size:0.65rem;color:#444;margin-top:0.3rem'>
                    {best} · MAE {mae} · R² {r2}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Slim footer watermark ─────────────────────────────────────────────
        st.markdown(f"""
        <div style='margin-top:1.5rem;padding-top:0.8rem;border-top:1px solid rgba(255,255,255,0.04);
                    text-align:center'>
            <img src='{F1_LOGO_URL}' style='width:50px;opacity:0.12;filter:grayscale(1)'>
            <div style='font-size:0.5rem;color:#1e1e2e;margin-top:0.25rem;letter-spacing:0.1em'>
                F1 2026 PREDICTOR
            </div>
        </div>
        """, unsafe_allow_html=True)

        return page_choice



# ─── HOME PAGE ────────────────────────────────────────────────────────────────
def show_home():
    # Hero
    st.markdown(f"""
    <div class="f1-hero">
        <h1>{get_icon('sparkles', 40, '#ff1e00')} FORMULA 1 &middot; 2026 <span class='f1-badge'>NEW ERA</span></h1>
        <p>AI-powered predictions for the most transformative season in F1 history</p>
    </div>
    """, unsafe_allow_html=True)

    # Countdown
    days_left = (OPENING_RACE_DATE - date.today()).days
    if days_left > 0:
        weeks, days = divmod(days_left, 7)
        st.markdown(f"""
        <div class="countdown-box">
            <div style='font-family:Rajdhani; font-size:0.8rem; color:#ff6b00; letter-spacing:0.15em; margin-bottom:0.8rem'>COUNTDOWN TO AUSTRALIAN GP &middot; MARCH 8</div>
            <span class='countdown-num'>{weeks}</span><span class='countdown-label'>WKS</span>
            <span class='countdown-num'>{days}</span><span class='countdown-label'>DAYS</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='countdown-box' style='color:#00ff88; font-family:Rajdhani;'>THE 2026 F1 SEASON IS UNDERWAY!</div>", unsafe_allow_html=True)

    st.markdown("")

    # Stats row
    st.markdown("<div class='section-header'>2026 SEASON AT A GLANCE</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("22", "Drivers"), ("11", "Teams"), ("24", "Races"),
        ("6", "Sprint Rounds"), ("50/50", "Hybrid Split"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], stats):
        col.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='label'>{label}</div></div>", unsafe_allow_html=True)

    # Regulation highlights
    st.markdown(f"<div class='section-header'>{get_icon('settings', 18, '#ff6b00')} THE NEW ERA — 2026 REGULATIONS</div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    regs = [
        ("zap", "Active Aerodynamics", "Movable front & rear wings. X-Mode on straights, Z-Mode in corners — replacing DRS entirely."),
        ("zap", "350kW MGU-K", "Near 3× increase in electrical output. 50/50 split between ICE and ERS — a true hybrid formula."),
        ("zap", "Overtake Mode", "Within 1s of the car ahead? Deploy 0.5MJ extra energy. No more DRS trains."),
        ("zap", "30kg Lighter", "Shorter wheelbase, narrower tires, simplified aero. Nimbler, more agile racing cars."),
    ]
    for col, (icon_name, title, desc) in zip([r1,r2,r3,r4], regs):
        col.markdown(f"<div class='reg-card'><div class='reg-icon'>{get_icon(icon_name, 32, '#ff6b00')}</div><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

    # New teams
    st.markdown(f"<div class='section-header'>{get_icon('sparkles', 18, '#ff1e00')} THE CLASS OF 2026</div>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown("""
        <div class='driver-card' style='border-color:#9B000055'>
            <div style='color:#9B0000; font-family:Rajdhani; font-weight:700'>AUDI FORMULA RACING</div>
            <div style='font-size:0.85rem; color:#ccc; margin-top:0.3rem'>Nico Hülkenberg · Gabriel Bortoleto</div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.2rem'>Power Unit: Audi · First works entry in F1</div>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown("""
        <div class='driver-card' style='border-color:#00308755'>
            <div style='color:#003087; font-family:Rajdhani; font-weight:700'>CADILLAC FORMULA RACING</div>
            <div style='font-size:0.85rem; color:#ccc; margin-top:0.3rem'>Sergio Pérez · Valtteri Bottas</div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.2rem'>Power Unit: Ferrari · First US works team</div>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        st.markdown(f"""
        <div class='driver-card' style='border-color:#ff1e0055'>
            <div style='color:#ff6b00; font-family:Rajdhani; font-weight:700'>DEFENDING CHAMPION</div>
            <div style='font-size:0.85rem; color:#ccc; margin-top:0.3rem'>Lando Norris &middot; McLaren (2025 WDC)</div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.2rem'>McLaren also won 2025 Constructors' Championship</div>
        </div>
        """, unsafe_allow_html=True)

    # Team grid
    st.markdown(f"<div class='section-header'>{get_icon('flag', 18, '#ff1801')} CONSTRUCTOR LINEUP</div>", unsafe_allow_html=True)
    team_data = []
    for info in DRIVERS_2026:
        team_data.append({"team": info["team"], "driver": info["name"], "code": info["code"]})
    team_df = pd.DataFrame(team_data)
    teams_grouped = team_df.groupby("team")["driver"].apply(lambda x: " & ".join(x)).reset_index()

    fig = go.Figure()
    for i, row in teams_grouped.iterrows():
        color = TEAM_COLORS_HEX.get(row["team"], "#888")
        fig.add_trace(go.Bar(
            x=[row["team"]], y=[1], name=row["team"], marker_color=color,
            text=[row["driver"]], textposition="inside",
            hovertemplate=f"<b>{row['team']}</b><br>{row['driver']}<extra></extra>",
        ))
    fig.update_layout(
        showlegend=False, barmode="stack", xaxis_tickangle=-30, height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk", color="#aaa"),
        margin=dict(t=10, b=80, l=20, r=20),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── CALENDAR PAGE ───────────────────────────────────────────────────────────
def show_calendar():
    today = pd.Timestamp(date.today())
    cal_df = pd.DataFrame(CALENDAR_2026)
    cal_df["date"] = pd.to_datetime(cal_df["date"])

    # ── Hero ─────────────────────────────────────────────────────────────────
    upcoming = cal_df[cal_df["date"] >= today]
    completed = cal_df[cal_df["date"] < today]
    races_done = len(completed)
    races_left = len(upcoming)
    next_race  = upcoming.iloc[0] if not upcoming.empty else None

    status_badge = ""
    if next_race is not None:
        nr_days = (next_race["date"] - today).days
        sprint_tag = " <span style='background:#9b30ff;color:#fff;font-size:0.52rem;font-weight:800;padding:0.1rem 0.4rem;border-radius:5px;vertical-align:middle'>SPRINT</span>" if next_race["sprint"] else ""
        status_badge = (
            f"<div style='margin-top:0.7rem;display:inline-flex;align-items:center;gap:0.6rem;"
            f"background:rgba(255,30,0,0.08);border:1px solid rgba(255,30,0,0.2);"
            f"border-radius:20px;padding:0.35rem 0.9rem;font-size:0.78rem;color:#ccc'>"
            f"{get_icon('compass', 14, '#ff6b00', 6)}"
            f"<span>Next: <b style='color:#fff'>{next_race['name']}</b>{sprint_tag}</span>"
            f"<span style='color:#ff1e00;font-family:Rajdhani;font-weight:900'>"
            f"{'TODAY' if nr_days == 0 else f'in {nr_days}d'}</span></div>"
        )

    st.markdown(f"""<div class="f1-hero" style="padding:1.6rem 2rem;text-align:left">
        <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem'>
          <div>
            <h1 style="font-size:1.7rem;margin:0">{get_icon("calendar",26,"#ff1e00")} 2026 GRAND PRIX CALENDAR</h1>
            <p style="margin:0.4rem 0 0 0">24 Grands Prix &middot; 5 continents &middot; 6 Sprint weekends</p>
          </div>
          <div style='display:flex;gap:1.2rem'>
            <div style='text-align:center'>
              <div style='font-family:Rajdhani;font-size:1.6rem;font-weight:900;color:#ff1e00'>{races_done}</div>
              <div style='font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:0.1em'>Completed</div>
            </div>
            <div style='text-align:center'>
              <div style='font-family:Rajdhani;font-size:1.6rem;font-weight:900;color:#27F4D2'>{races_left}</div>
              <div style='font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:0.1em'>Remaining</div>
            </div>
          </div>
        </div>
        {status_badge}
    </div>""", unsafe_allow_html=True)

    # ── Filter bar ───────────────────────────────────────────────────────────
    FILTER_OPTS = ["All", "Street", "Technical", "Power", "High Speed", "Sprint"]
    ct_map = {"Street": "street", "Technical": "technical",
               "Power": "power", "High Speed": "high_speed"}

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        chosen_filter = st.radio(
            "filter", FILTER_OPTS, horizontal=True, label_visibility="collapsed", index=0
        )
    with fc2:
        show_past = st.checkbox("Show past races", value=True)

    filtered = cal_df.copy()
    if not show_past:
        filtered = filtered[filtered["date"] >= today]
    if chosen_filter == "Sprint":
        filtered = filtered[filtered["sprint"] == True]
    elif chosen_filter != "All":
        filtered = filtered[filtered["circuit_type"] == ct_map[chosen_filter]]

    # ── Circuit-type color map ────────────────────────────────────────────────
    ct_color_map = {
        "power":     "#ff6b00",
        "high_speed":"#ff1e00",
        "street":    "#FF87BC",
        "technical": "#27F4D2",
        "balanced":  "#aaa",
    }

    # ── Group by month ────────────────────────────────────────────────────────
    filtered = filtered.sort_values("date").reset_index(drop=True)
    filtered["month_label"] = filtered["date"].dt.strftime("%B %Y")

    current_month = None

    # We'll render cards in groups of 2 (two-column grid)
    rows = list(filtered.iterrows())
    i = 0
    while i < len(rows):
        _, race_a = rows[i]

        # Month divider
        if race_a["month_label"] != current_month:
            current_month = race_a["month_label"]
            month_races = filtered[filtered["month_label"] == current_month]
            race_count = len(month_races)
            sprint_count = int(month_races["sprint"].sum())
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;
                        margin:1.6rem 0 0.7rem 0'>
              <div style='font-family:Rajdhani;font-size:0.75rem;font-weight:900;
                          color:#fff;letter-spacing:0.15em;text-transform:uppercase'>
                {current_month}
              </div>
              <div style='flex:1;height:1px;background:linear-gradient(90deg,rgba(255,255,255,0.12),transparent)'></div>
              <div style='font-size:0.6rem;color:#444'>
                {race_count} race{"s" if race_count != 1 else ""}
                {f"&nbsp;&middot;&nbsp;<span style='color:#9b30ff'>{sprint_count} sprint</span>" if sprint_count else ""}
              </div>
            </div>""", unsafe_allow_html=True)

        # Pair the cards (or lone card at end of month / filter)
        pair_in_month = []
        for j in range(i, min(i + 2, len(rows))):
            _, r = rows[j]
            if r["month_label"] == race_a["month_label"]:
                pair_in_month.append(r)
            else:
                break

        cols = st.columns(len(pair_in_month))
        for col, race in zip(cols, pair_in_month):
            _render_race_card(col, race, ct_color_map, today)

        i += len(pair_in_month)

    # ── Season timeline chart ─────────────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:2.5rem'>SEASON TIMELINE</div>",
                unsafe_allow_html=True)
    month_counts = (
        cal_df.groupby(cal_df["date"].dt.month_name()).size()
        .reindex(["March","April","May","June","July","August",
                  "September","October","November","December"], fill_value=0)
        .reset_index()
    )
    month_counts.columns = ["Month", "Races"]
    fig = px.bar(month_counts, x="Month", y="Races", color="Races",
                 color_continuous_scale=["#1a1a2e", "#ff6b00", "#ff1e00"])
    dark_layout(fig, "Races Per Month", 260)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_race_card(col, race, ct_color_map, today):
    """Render a single race card with map + detail expander."""
    ct_color   = ct_color_map.get(race["circuit_type"], "#888")
    ct_label   = race["circuit_type"].replace("_", " ").upper()
    flag_e     = flag_pill(race["flag"], "#888")
    date_str   = race["date"].strftime("%b %d, %Y")
    day_name   = race["date"].strftime("%A")
    is_past    = race["date"] < today
    is_next    = False
    upcoming_df = pd.DataFrame(CALENDAR_2026)
    upcoming_df["date"] = pd.to_datetime(upcoming_df["date"])
    nxt = upcoming_df[upcoming_df["date"] >= today]
    if not nxt.empty and nxt.iloc[0]["round"] == race["round"]:
        is_next = True

    sprint_badge = (
        "<span style='background:linear-gradient(90deg,#9b30ff,#6600cc);"
        "color:#fff;font-size:0.5rem;font-weight:800;letter-spacing:0.1em;"
        "padding:0.15rem 0.45rem;border-radius:5px;margin-left:0.4rem;"
        "vertical-align:middle;text-transform:uppercase'>SPRINT</span>"
        if race["sprint"] else ""
    )

    border_color = "#ff1e00" if is_next else ("rgba(255,255,255,0.04)" if is_past else "rgba(255,255,255,0.08)")
    past_opacity = "opacity:0.55;" if is_past else ""
    glow = "box-shadow:0 0 24px rgba(255,30,0,0.18);" if is_next else ""

    with col:
        # ── Card header (always visible) ──────────────────────────────────
        st.markdown(f"""
        <div style='background:rgba(18,18,30,0.85);
                    border:1px solid {border_color};border-radius:18px;
                    padding:1rem 1.1rem 0.7rem 1.1rem;
                    transition:all 0.25s;{glow}{past_opacity}margin-bottom:0.2rem'>
          <div style='display:flex;align-items:flex-start;justify-content:space-between;gap:0.5rem'>
            <div style='display:flex;align-items:center;gap:0.55rem;flex:1;min-width:0'>
              <div style='background:linear-gradient(135deg,#ff1e00,#ff6b00);
                          color:#fff;font-family:Rajdhani;font-size:0.62rem;
                          font-weight:900;padding:0.25rem 0.5rem;
                          border-radius:7px;letter-spacing:0.06em;flex-shrink:0'>
                R{race["round"]}
              </div>
              {flag_e}
              <div style='min-width:0'>
                <div style='font-weight:700;color:{"#888" if is_past else "#fff"};
                             font-size:0.9rem;white-space:nowrap;
                             overflow:hidden;text-overflow:ellipsis'>
                  {race["name"]}{sprint_badge}
                </div>
                <div style='font-size:0.68rem;color:#444;margin-top:2px'>
                  {race["circuit"]}
                </div>
              </div>
            </div>
            <div style='text-align:right;flex-shrink:0'>
              <div style='font-size:0.75rem;color:{"#555" if is_past else "#bbb"};font-weight:500'>
                {date_str}
              </div>
              <div style='display:inline-block;font-size:0.56rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:0.08em;
                          padding:0.12rem 0.45rem;border-radius:999px;
                          border:1px solid {ct_color};color:{ct_color};
                          margin-top:0.3rem;opacity:{"0.5" if is_past else "1"}'>
                {ct_label}
              </div>
              {'<div style="font-size:0.6rem;color:#00ff88;font-weight:700;margin-top:0.2rem">✓ COMPLETED</div>' if is_past else ('<div style="font-size:0.6rem;color:#ff1e00;font-weight:700;margin-top:0.2rem;animation:pulse 1.5s infinite">◉ NEXT RACE</div>' if is_next else "")}
            </div>
          </div>

          <!-- Location map thumbnail -->
          {get_track_svg(race["circuit"], color=ct_color)}
        </div>
        """, unsafe_allow_html=True)

        # ── Expander: track SVG + stats + weekend schedule ───────────────
        with st.expander(f"Details — {race['circuit']}", expanded=False):
            svg_col, stat_col = st.columns([1, 1.3])
            details = CIRCUIT_DETAILS.get(race["circuit"], {})

            with svg_col:
                map_html = get_map_embed(race["circuit"], height=200)
                st.markdown(
                    f"<div style='display:flex;justify-content:center;"
                    f"padding:0.5rem 0; width:100%'>{map_html}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center;font-family:Rajdhani;"
                    f"font-size:0.6rem;color:{ct_color};letter-spacing:0.12em;"
                    f"text-transform:uppercase'>{race['circuit']}</div>",
                    unsafe_allow_html=True,
                )

            with stat_col:
                if details:
                    st.markdown(f"""
                    <div class='track-stat-grid'>
                      <div class='track-stat'>
                        <div class='track-stat-label'>Track Length</div>
                        <div class='track-stat-value'>{details.get("length","–")} km</div>
                      </div>
                      <div class='track-stat'>
                        <div class='track-stat-label'>DRS Zones</div>
                        <div class='track-stat-value'>{details.get("drs_zones","–")}</div>
                      </div>
                      <div class='track-stat'>
                        <div class='track-stat-label'>First GP</div>
                        <div class='track-stat-value'>{details.get("first_gp","–")}</div>
                      </div>
                      <div class='track-stat'>
                        <div class='track-stat-label'>Laps</div>
                        <div class='track-stat-value'>{race["laps"]}</div>
                      </div>
                    </div>
                    <div class='track-stat' style='margin-top:0.5rem'>
                      <div class='track-stat-label'>Lap Record</div>
                      <div class='track-stat-value' style='color:#ff6b00;font-size:0.78rem'>
                        {details.get("lap_record","–")}
                      </div>
                    </div>
                    <div class='track-stat' style='margin-top:0.5rem'>
                      <div class='track-stat-label'>Key Corner</div>
                      <div class='track-stat-value' style='font-size:0.78rem'>
                        {details.get("key_corner","–")}
                      </div>
                    </div>
                    <div style='margin-top:0.6rem;padding:0.55rem 0.8rem;
                                background:rgba(255,255,255,0.02);border-radius:8px;
                                border-left:3px solid {ct_color};
                                font-size:0.72rem;color:#aaa;line-height:1.55'>
                      {details.get("characteristic","")}
                    </div>""", unsafe_allow_html=True)

            # Weekend schedule
            schedule = _SPRINT_WKD if race["sprint"] else _NORMAL_WKD
            st.markdown(
                f"<div style='margin-top:0.9rem;margin-bottom:0.3rem;"
                f"font-size:0.62rem;color:#444;text-transform:uppercase;"
                f"letter-spacing:0.12em;font-weight:700'>"
                f"{get_icon('clock',11,'#444',4)} Weekend Schedule</div>",
                unsafe_allow_html=True
            )
            rows_html = ""
            for day, session, color in schedule:
                rows_html += (
                    f"<div style='display:flex;align-items:center;gap:0.5rem;"
                    f"padding:0.28rem 0;border-bottom:1px solid rgba(255,255,255,0.04)'>"
                    f"<span style='font-family:Rajdhani;font-size:0.52rem;font-weight:700;"
                    f"color:#333;min-width:28px'>{day}</span>"
                    f"<span style='width:3px;height:3px;border-radius:50%;"
                    f"background:{color};flex-shrink:0'></span>"
                    f"<span style='font-size:0.72rem;color:{color};font-weight:600'>"
                    f"{session}</span></div>"
                )
            st.markdown(f"<div style='padding:0 0.2rem'>{rows_html}</div>",
                        unsafe_allow_html=True)


# ─── RACE PREDICTOR PAGE ─────────────────────────────────────────────────────
def show_race_predictor(engine):
    st.markdown(f"""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">{get_icon('predictor', 28, '#ff1e00')} RACE PREDICTOR</h1>
        <p>Select any 2026 Grand Prix &middot; Monte Carlo simulation &middot; Animated podium</p>
    </div>""", unsafe_allow_html=True)

    if not engine.is_trained:
        st.warning("Train the model first using the sidebar button!")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        race_names = [r["name"] for r in CALENDAR_2026]
        selected_race_name = st.selectbox("Select Grand Prix", race_names)
    with col2:
        n_sims = st.select_slider("Monte Carlo Sims", [500, 1000, 2000, 5000], value=2000)
    with col3:
        show_laps = st.checkbox("Lap-by-Lap Trace", value=True)

    race_info = next(r for r in CALENDAR_2026 if r["name"] == selected_race_name)
    country_name = COUNTRY_NAMES.get(race_info.get("flag",""), race_info["country"])

    if st.button(f"PREDICT {selected_race_name.upper()}", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations for {selected_race_name}..."):
            det_df, mc_df = engine.predict_race(
                race_info["circuit"], include_monte_carlo=True, n_sims=n_sims
            )

        ct_color = {
            "power": "#ff6b00", "high_speed": "#ff1e00", "street": "#FF87BC",
            "technical": "#27F4D2", "balanced": "#aaa"
        }.get(race_info["circuit_type"], "#888")
        sprint_badge = "<span class='race-sprint-badge'>SPRINT WEEKEND</span>" if race_info["sprint"] else ""

        st.markdown(f"""<div style='background:linear-gradient(135deg,#0d0d1a,#15152a);border:1px solid #ffffff18;border-radius:20px;padding:1.5rem 2rem;margin-bottom:2rem;display:flex;gap:1.5rem;align-items:center'><div style='flex:1'><div style='font-family:Rajdhani;font-size:1.4rem;font-weight:900;color:#fff'>{selected_race_name}</div><div style='color:#888;font-size:0.9rem'>{race_info["circuit"]} &middot; {country_name} &middot; <span style='color:{ct_color}'>{race_info["circuit_type"].replace("_"," ").title()}</span></div>{sprint_badge}</div></div>""", unsafe_allow_html=True)

        # Podium ceremony
        top3 = mc_df.head(3)
        st.markdown("<div class='section-header'>PREDICTED PODIUM</div>", unsafe_allow_html=True)
        _, p2c, p1c, p3c, _ = st.columns([0.5, 1, 1.2, 1, 0.5])

        for col, idx, medal_cls, medal_label in [
            (p2c, 1, "podium-p2", "P2"),
            (p1c, 0, "podium-p1", "WINNER"),
            (p3c, 2, "podium-p3", "P3"),
        ]:
            d = top3.iloc[idx]
            tc = d["team_color"]
            big = idx == 0
            icon_color = "#FFD700" if idx == 0 else ("#C0C0C0" if idx == 1 else "#CD7F32")
            col.markdown(f"""<div class='{medal_cls}'><div style='margin-bottom:0.5rem'>{get_icon('trophy', 32 if big else 24, icon_color)}</div><div style='font-family:Rajdhani;font-size:0.7rem;font-weight:700;color:{icon_color};margin-bottom:0.4rem'>{medal_label}</div><h2 style='{"font-size:1.8rem;" if big else "font-size:1.4rem;"}'>{d["driver_code"]}</h2><p style='font-size:{"0.95rem" if big else "0.8rem"};{"font-weight:600;" if big else ""}'>{d["driver_name"]}</p><div style='display:inline-block;background:{tc};color:#fff;padding:0.1rem 0.5rem;border-radius:6px;font-size:0.65rem;font-weight:700'>{d["team"]}</div><p style='margin-top:0.8rem;font-size:{"1.1rem" if big else "0.9rem"}'>Win <b>{d["win_prob"]:.1%}</b></p></div>""", unsafe_allow_html=True)

        # Win probability chart
        st.markdown("<div class='section-header'>WIN PROBABILITY</div>", unsafe_allow_html=True)
        top12 = mc_df.head(12)
        fig = go.Figure(go.Bar(
            x=top12["driver_code"],
            y=(top12["win_prob"] * 100).round(1),
            marker_color=top12["team_color"].tolist(),
            text=[f"{p:.1f}%" for p in top12["win_prob"] * 100],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Win probability: %{y:.1f}%<extra></extra>",
        ))
        dark_layout(fig, "Win Probability (%)", 380)
        fig.update_yaxes(title_text="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

        # ── 3D Monte Carlo Probability Surface ────────────────────────────
        st.markdown("<div class='section-header'>3D MONTE CARLO SIMULATION LANDSCAPE</div>", unsafe_allow_html=True)
        top10_mc = mc_df.head(10)
        prob_cols = ["win_prob", "podium_prob", "top5_prob", "top10_prob"]
        prob_labels = ["Win", "Podium", "Top 5", "Points"]
        z_surface = (top10_mc[prob_cols].values * 100)
        driver_labels = top10_mc["driver_code"].tolist()
        team_colors_list = top10_mc["team_color"].tolist()

        fig_3d = go.Figure()
        fig_3d.add_trace(go.Surface(
            z=z_surface,
            x=list(range(len(prob_labels))),
            y=list(range(len(driver_labels))),
            colorscale=[[0, "#0a0a1a"], [0.25, "#1a0a2e"], [0.5, "#ff6b00"], [0.75, "#ff3300"], [1, "#ff1801"]],
            opacity=0.85,
            showscale=False,
            hovertemplate="Driver: %{text}<br>Category: %{customdata}<br>Probability: %{z:.1f}%<extra></extra>",
            text=[[d]*len(prob_labels) for d in driver_labels],
            customdata=[[p for p in prob_labels]]*len(driver_labels),
        ))
        for i, (code, tc) in enumerate(zip(driver_labels, team_colors_list)):
            for j, pl in enumerate(prob_labels):
                fig_3d.add_trace(go.Scatter3d(
                    x=[j], y=[i], z=[z_surface[i][j]],
                    mode='markers',
                    marker=dict(size=5, color=tc, line=dict(width=1, color='#fff')),
                    showlegend=False,
                    hovertemplate=f"<b>{code}</b><br>{pl}: {z_surface[i][j]:.1f}%<extra></extra>"
                ))
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="Probability Tier", tickvals=list(range(len(prob_labels))), ticktext=prob_labels, backgroundcolor="rgba(0,0,0,0)", gridcolor="#222"),
                yaxis=dict(title="Driver", tickvals=list(range(len(driver_labels))), ticktext=driver_labels, backgroundcolor="rgba(0,0,0,0)", gridcolor="#222"),
                zaxis=dict(title="Probability (%)", backgroundcolor="rgba(0,0,0,0)", gridcolor="#222"),
                bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=550,
            margin=dict(l=0, r=0, b=0, t=10),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Generate lap data if needed
        lap_df = None
        if show_laps:
            with st.spinner("Generating lap trace..."):
                if "predicted_pos" not in det_df.columns and "expected_pos" in mc_df.columns:
                    sim_input = mc_df.copy()
                    sim_input["predicted_pos"] = sim_input["expected_pos"].round().astype(int)
                else:
                    sim_input = det_df
                lap_df = engine.generate_lap_simulation(sim_input, n_laps=race_info.get("laps", 57))

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Full Grid", "Lap-by-Lap Trace", "Position Heatmap", "3D Race Trajectory"])

        with tab1:
            disp_cols = ["driver_code", "driver_name", "team", "expected_pos",
                         "win_prob", "podium_prob", "top5_prob", "top10_prob", "dnf_prob"]
            tbl = mc_df[[c for c in disp_cols if c in mc_df.columns]].copy()
            for pc in ["win_prob", "podium_prob", "top5_prob", "top10_prob", "dnf_prob"]:
                if pc in tbl.columns:
                    tbl[pc] = tbl[pc].apply(lambda x: f"{x:.1%}")
            tbl = tbl.rename(columns={
                "driver_code": "Driver", "driver_name": "Name", "team": "Team",
                "expected_pos": "Exp.Pos", "win_prob": "Win%", "podium_prob": "Podium%",
                "top5_prob": "Top5%", "top10_prob": "Points%", "dnf_prob": "DNF%",
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        with tab2:
            if show_laps and lap_df is not None:
                top12_codes = mc_df.head(12)["driver_code"].tolist()
                lap_top = lap_df[lap_df["driver_code"].isin(top12_codes)]
                fig2 = go.Figure()
                for code in top12_codes:
                    sub = lap_top[lap_top["driver_code"] == code]
                    if sub.empty:
                        continue
                    tc = TEAM_COLORS.get(sub["team"].iloc[0], "#888")
                    fig2.add_trace(go.Scatter(
                        x=sub["lap"], y=sub["position"], mode="lines", name=code,
                        line=dict(color=tc, width=2.5),
                        hovertemplate=f"<b>{code}</b> Lap %{{x}}: P%{{y:.0f}}<extra></extra>",
                    ))
                dark_layout(fig2, "Lap-by-Lap Position Trace (Top 12)", 500)
                fig2.update_yaxes(autorange="reversed", title_text="Position", dtick=2)
                fig2.update_xaxes(title_text="Lap")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Enable 'Lap-by-Lap Trace' checkbox to see this view.")

        with tab3:
            top15 = mc_df.head(15)
            z_vals = (top15[["win_prob", "podium_prob", "top5_prob", "top10_prob"]].values * 100)
            fig3 = go.Figure(go.Heatmap(
                z=z_vals, x=["Win", "Podium", "Top 5", "Top 10"],
                y=top15["driver_code"].tolist(),
                colorscale=[[0, "#0a0a0f"], [0.35, "#3a0a0a"], [0.7, "#ff6b00"], [1, "#ff1e00"]],
                text=[[f"{v:.1f}%" for v in row] for row in z_vals],
                texttemplate="%{text}",
            ))
            dark_layout(fig3, "Driver Probability Heatmap", 500)
            fig3.update_yaxes(autorange="reversed")
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            if show_laps and lap_df is not None:
                top10_codes = mc_df.head(10)["driver_code"].tolist()
                lap_top = lap_df[lap_df["driver_code"].isin(top10_codes)]
                fig4 = go.Figure()
                for i, code in enumerate(top10_codes):
                    sub = lap_top[lap_top["driver_code"] == code]
                    if sub.empty: continue
                    tc = TEAM_COLORS.get(sub["team"].iloc[0], "#888")
                    fig4.add_trace(go.Scatter3d(
                        x=sub["lap"], y=[code]*len(sub), z=sub["position"],
                        mode='lines+markers', name=code,
                        marker=dict(size=4, color=tc),
                        line=dict(color=tc, width=4)
                    ))
                fig4.update_layout(
                    title=dict(text="3D Race Trajectory (Top 10)", font=dict(family="Rajdhani", size=18, color="#fff")),
                    scene=dict(
                        xaxis=dict(title="Lap", backgroundcolor="rgba(0,0,0,0)", gridcolor="#333"),
                        yaxis=dict(title="Driver", backgroundcolor="rgba(0,0,0,0)", gridcolor="#333"),
                        zaxis=dict(title="Position", autorange="reversed", backgroundcolor="rgba(0,0,0,0)", gridcolor="#333"),
                        bgcolor="rgba(0,0,0,0)"
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=600, margin=dict(l=0, r=0, b=0, t=40), showlegend=False
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Enable 'Lap-by-Lap Trace' checkbox to generate the 3D race simulation.")

        # ── Actual Results Comparison (for concluded races) ───────────────
        race_date = pd.Timestamp(race_info["date"])
        if race_date < pd.Timestamp.now():
            st.markdown("<div class='section-header'>PREDICTION vs ACTUAL RESULTS</div>", unsafe_allow_html=True)
            try:
                from src.live_data import LiveDataIngestor
                ingestor = LiveDataIngestor()
                live_df = ingestor.load_cached_data()
                if not live_df.empty:
                    round_df = live_df[live_df["round"] == race_info["round"]]
                    if not round_df.empty:
                        comp_rows = []
                        for _, actual in round_df.iterrows():
                            code = actual["driver_code"]
                            pred_row = mc_df[mc_df["driver_code"] == code]
                            pred_pos = pred_row["expected_pos"].values[0] if not pred_row.empty else None
                            actual_pos = int(actual["finish_position"])
                            delta = round(pred_pos - actual_pos, 1) if pred_pos else None
                            comp_rows.append({
                                "Driver": code,
                                "Actual Pos": actual_pos,
                                "Predicted Pos": round(pred_pos, 1) if pred_pos else "N/A",
                                "Delta": f"{'+' if delta and delta > 0 else ''}{delta}" if delta else "N/A",
                                "Status": actual.get("status", ""),
                            })
                        comp_df = pd.DataFrame(comp_rows).sort_values("Actual Pos")

                        ac1, ac2 = st.columns([1.5, 1])
                        with ac1:
                            st.dataframe(comp_df, use_container_width=True, hide_index=True)
                        with ac2:
                            valid = comp_df[comp_df["Predicted Pos"] != "N/A"].copy()
                            if not valid.empty:
                                valid["Predicted Pos"] = valid["Predicted Pos"].astype(float)
                                valid["Actual Pos"] = valid["Actual Pos"].astype(int)
                                mae = abs(valid["Predicted Pos"] - valid["Actual Pos"]).mean()
                                top5_actual = set(valid[valid["Actual Pos"] <= 5]["Driver"])
                                top5_pred_set = set(valid.nsmallest(5, "Predicted Pos")["Driver"])
                                overlap = len(top5_actual & top5_pred_set)
                                accuracy_color = "#00ff88" if mae < 3 else "#ff6b00"
                                accuracy_msg = "Excellent prediction accuracy." if mae < 3 else ("Moderate deviation -- model calibrating." if mae < 5 else "Significant delta -- unexpected race events.")
                                st.markdown(f"""<div style='background:rgba(18,18,30,0.9);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:1.5rem'><div style='font-family:Rajdhani;font-size:1.1rem;font-weight:700;color:#fff;margin-bottom:1rem'>Prediction Accuracy</div><div style='display:flex;gap:1rem;flex-wrap:wrap'><div style='flex:1;text-align:center;padding:0.8rem;background:rgba(255,255,255,0.03);border-radius:10px'><div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:{accuracy_color}'>{mae:.1f}</div><div style='font-size:0.65rem;color:#666;text-transform:uppercase'>Mean Abs Error</div></div><div style='flex:1;text-align:center;padding:0.8rem;background:rgba(255,255,255,0.03);border-radius:10px'><div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:#27F4D2'>{overlap}/5</div><div style='font-size:0.65rem;color:#666;text-transform:uppercase'>Top-5 Overlap</div></div></div><div style='margin-top:1rem;font-size:0.75rem;color:#888;line-height:1.6'>{accuracy_msg}</div></div>""", unsafe_allow_html=True)

                        # Scatter plot: predicted vs actual
                        valid2 = comp_df[comp_df["Predicted Pos"] != "N/A"].copy()
                        if not valid2.empty:
                            valid2["Predicted Pos"] = valid2["Predicted Pos"].astype(float)
                            valid2["Actual Pos"] = valid2["Actual Pos"].astype(int)
                            fig_comp = go.Figure()
                            for _, r in valid2.iterrows():
                                tc = TEAM_COLORS.get(
                                    next((d["team"] for d in DRIVERS_2026 if d["code"] == r["Driver"]), ""), "#888"
                                )
                                fig_comp.add_trace(go.Scatter(
                                    x=[r["Actual Pos"]], y=[r["Predicted Pos"]],
                                    mode="markers+text", text=[r["Driver"]],
                                    textposition="top center",
                                    marker=dict(size=12, color=tc, line=dict(width=1, color="#fff")),
                                    showlegend=False,
                                    hovertemplate=f"<b>{r['Driver']}</b><br>Actual: P{int(r['Actual Pos'])}<br>Predicted: P{r['Predicted Pos']:.1f}<extra></extra>"
                                ))
                            fig_comp.add_trace(go.Scatter(
                                x=[1, 22], y=[1, 22], mode="lines",
                                line=dict(dash="dash", color="rgba(255,255,255,0.15)", width=1),
                                showlegend=False, hoverinfo="skip"
                            ))
                            dark_layout(fig_comp, "Predicted vs Actual Finish Position", 420)
                            fig_comp.update_xaxes(title_text="Actual Position", dtick=2)
                            fig_comp.update_yaxes(title_text="Predicted Position", dtick=2)
                            st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        st.info("This race has concluded but no results data cached yet. Click 'Initialize & Sync Live Data' in the sidebar.")
                else:
                    st.info("No live data available. Click 'Initialize & Sync Live Data' to fetch actual results.")
            except Exception as e:
                st.warning(f"Could not load actual results: {e}")


# ─── SEASON SIMULATOR PAGE ────────────────────────────────────────────────────
def show_season_simulator(engine):
    st.markdown(f"""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">{get_icon('trophy', 32, '#ff1e00')} SEASON SIMULATOR</h1>
        <p>Full 2026 championship forecast across all 24 rounds</p>
    </div>""", unsafe_allow_html=True)

    if not engine.is_trained:
        st.warning("Train the model first!")
        return

    if st.button("SIMULATE THE FULL 2026 SEASON", use_container_width=True):
        with st.spinner("Simulating all 24 races..."):
            standings = engine.predict_full_season()

        st.success("2026 Championship simulation complete!")

        top5 = standings.head(5)
        st.markdown("<div class='section-header'>WORLD CHAMPIONSHIP — TOP 5</div>", unsafe_allow_html=True)
        cols5 = st.columns(5)
        for i, (col, (_, row)) in enumerate(zip(cols5, top5.iterrows())):
            tc = row["team_color"]
            col.markdown(f"""
            <div class='driver-card' style='border-top:4px solid {tc};text-align:center'>
                <div style='margin-bottom:0.5rem'>{get_icon('trophy', 24, tc)}</div>
                <div style='font-family:Rajdhani;font-weight:900;color:{tc};font-size:1rem'>{row["driver_code"]}</div>
                <div style='font-size:0.75rem;color:#888'>{row["driver_name"]}</div>
                <div style='font-family:Rajdhani;font-size:1.8rem;color:#fff;font-weight:900;margin-top:0.4rem'>
                    {int(row["points"])} <span style='font-size:0.6rem;color:#555'>PTS</span>
                </div>
                <div style='font-size:0.75rem;color:#666;margin-top:0.2rem'>{int(row["wins"])} WINS</div>
            </div>
            """, unsafe_allow_html=True)

        # Full drivers chart
        fig = go.Figure()
        for _, row in standings.iterrows():
            fig.add_trace(go.Bar(
                x=[row["driver_code"]], y=[row["points"]],
                marker_color=row["team_color"],
                text=[f"{int(row['points'])}"],
                textposition="outside",
                hovertemplate=(
                    f"<b>{row['driver_name']}</b><br>{row['team']}<br>"
                    f"Pts: {int(row['points'])} | Wins: {int(row['wins'])}<extra></extra>"
                ),
            ))
        dark_layout(fig, "2026 Predicted Driver Championship", 460)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        fig.update_yaxes(title_text="Points")
        st.plotly_chart(fig, use_container_width=True)

        # Constructors
        st.markdown(f"<div class='section-header'>{get_icon('layers', 18, '#ff1e00')} CONSTRUCTORS' CHAMPIONSHIP</div>", unsafe_allow_html=True)
        con_df = (
            standings.groupby(["team", "team_color"])
            .agg(points=("points", "sum"), wins=("wins", "sum"))
            .reset_index()
            .sort_values("points", ascending=False)
        )
        fig2 = go.Figure()
        for _, row in con_df.iterrows():
            fig2.add_trace(go.Bar(
                x=[row["team"]], y=[row["points"]],
                marker_color=row["team_color"],
                text=[f"{int(row['points'])}"],
                textposition="outside",
                hovertemplate=f"<b>{row['team']}</b><br>Pts: {int(row['points'])} | Wins: {int(row['wins'])}<extra></extra>",
            ))
        dark_layout(fig2, "2026 Predicted Constructors' Championship", 400)
        fig2.update_layout(showlegend=False, xaxis_tickangle=-30)
        fig2.update_yaxes(title_text="Points")
        st.plotly_chart(fig2, use_container_width=True)

        # Full table
        st.markdown(f"<div class='section-header'>{get_icon('list', 18, '#ff1e00')} FULL STANDINGS TABLE</div>", unsafe_allow_html=True)
        disp = standings[["position", "driver_code", "driver_name", "team", "points", "wins", "podiums"]].copy()
        disp["points"] = disp["points"].astype(int)
        disp = disp.rename(columns={
            "position": "Pos", "driver_code": "Code", "driver_name": "Driver",
            "team": "Team", "points": "Pts", "wins": "Wins", "podiums": "Podiums",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ─── DRIVER PROFILES PAGE ─────────────────────────────────────────────────────
def show_driver_profiles():
    st.markdown(f"""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">{get_icon('user', 28, '#ff1e00')} DRIVER PROFILES</h1>
        <p>All 22 drivers &middot; Performance ratings &middot; Team details</p>
    </div>""", unsafe_allow_html=True)

    all_teams = sorted(set(d["team"] for d in DRIVERS_2026))
    sel_team = st.selectbox("Filter by team", ["All Teams"] + all_teams)

    rows = []
    for info in DRIVERS_2026:
        if sel_team != "All Teams" and info["team"] != sel_team:
            continue
        overall = round(
            info.get("pace", 7) * 0.30 + info.get("consistency", 7) * 0.20 +
            info.get("qualifying", 7) * 0.20 + info.get("overtaking", 7) * 0.15 +
            info.get("wet_skill", 7) * 0.10 + info.get("reg_adaptation", 7) * 0.05, 2
        )
        rows.append({**info, "overall": overall})
    rows.sort(key=lambda x: -x["overall"])

    for i in range(0, len(rows), 2):
        c1, c2 = st.columns(2)
        for col, driver in zip([c1, c2], rows[i:i + 2]):
            tc = TEAM_COLORS.get(driver["team"], "#888")
            rookie_label = (" <span style='background:#ff6b00;color:#fff;font-size:0.58rem;"
                            "padding:0.1rem 0.3rem;border-radius:3px;font-weight:700'>ROOKIE</span>"
                            if driver["rookie"] else "")
            skill_keys = [
                ("pace", "Pace"), ("consistency", "Consistency"), ("qualifying", "Qualifying"),
                ("overtaking", "Overtaking"), ("wet_skill", "Wet Skill"), ("reg_adaptation", "Reg Adapt."),
            ]
            bars_html = "".join([
                f"<div style='margin:0.2rem 0'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#aaa;margin-bottom:2px'>"
                f"<span>{label}</span><span>{driver.get(key, 7):.1f}</span></div>"
                f"<div class='prob-bar-bg' style='height:4px;background:rgba(255,255,255,0.05)'><div class='prob-bar-fill' style='height:4px;width:{driver.get(key, 7) / 10 * 100}%;background:{tc}'></div></div>"
                f"</div>"
                for key, label in skill_keys
            ])
            headshot_url = f"https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/2024Drivers/{driver['name'].split()[-1].lower()}.png"
            
            col.markdown(f"""
            <div class='driver-card' style='border-left:5px solid {tc}'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:0.8rem'>
                    <div style='display:flex;gap:0.8rem;align-items:center;flex:1'>
                        <img src='{headshot_url}' style='width:64px;height:64px;object-fit:cover;border-radius:50%;background:rgba(255,255,255,0.05);border:2px solid {tc}'>
                        <div>
                            <div style='font-size:1.4rem;font-weight:900;color:{tc};font-family:Rajdhani'>
                                {driver["code"]}{rookie_label}</div>
                            <div style='font-size:0.88rem;color:#ccc;font-weight:600'>{driver["name"]}</div>
                            <div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.05em'>
                                {driver["team"]} &middot; {driver["nationality"]}</div>
                        </div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-family:Rajdhani;font-size:1.8rem;font-weight:900;color:#ff1e00'>{driver["overall"]:.1f}</div>
                        <div style='font-size:0.6rem;color:#555;text-transform:uppercase'>Rating</div>
                    </div>
                </div>
                <div style='margin-top:1rem'>{bars_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"<div class='section-header'>{get_icon('bar-chart', 18, '#ff1e00')} DRIVER COMPARISON RADAR</div>", unsafe_allow_html=True)
    all_codes = [r["code"] for r in rows]
    default_sel = all_codes[:4] if len(all_codes) >= 4 else all_codes
    sel_drivers = st.multiselect("Select drivers to compare", all_codes, default=default_sel)
    if sel_drivers:
        categories = ["Pace", "Consistency", "Qualifying", "Overtaking", "Wet Skill", "Reg Adapt."]
        fig = go.Figure()
        for code in sel_drivers:
            drv = next((r for r in rows if r["code"] == code), None)
            if not drv:
                continue
            vals = [drv.get("pace", 7), drv.get("consistency", 7), drv.get("qualifying", 7),
                    drv.get("overtaking", 7), drv.get("wet_skill", 7), drv.get("reg_adaptation", 7)]
            tc = TEAM_COLORS.get(drv["team"], "#888")
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=categories + [categories[0]],
                fill="toself", name=code, line_color=tc, fillcolor=_hex_rgba(tc, 0.2),
            ))
        dark_layout(fig, "Attribute Radar Comparison", 480)
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[5, 10], gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#666")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#aaa")),
        ))
        st.plotly_chart(fig, use_container_width=True)


# ─── ANALYTICS PAGE ───────────────────────────────────────────────────────────
def show_analytics():
    st.markdown(f"""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">{get_icon('analytics', 32, '#ff1e00')} ANALYTICS</h1>
        <p>Team power rankings &middot; 2026 regulation impact &middot; Circuit intelligence</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Team Power Rankings", "Regulation Impact", "Circuit Intel"])

    with tab1:
        st.markdown("<div class='section-header'>2026 PRE-SEASON POWER RANKINGS</div>", unsafe_allow_html=True)
        team_rows = []
        for team, ratings in TEAM_CAR_RATINGS.items():
            overall = round(sum(ratings.values()) / len(ratings), 2)
            team_rows.append({"Team": team, "Overall": overall, **ratings, "Color": TEAM_COLORS.get(team, "#888")})
        team_df = pd.DataFrame(team_rows).sort_values("Overall", ascending=False)

        metrics = ["speed", "aero_efficiency", "reliability", "active_aero_mastery"]
        metric_labels = ["Speed", "Aero Efficiency", "Reliability", "Active Aero"]

        fig = go.Figure()
        for key, label in zip(metrics, metric_labels):
            fig.add_trace(go.Bar(name=label, x=team_df["Team"], y=team_df[key]))
        dark_layout(fig, "Team Rating Breakdown (10-point scale)", 420)
        fig.update_layout(barmode="group", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        z = team_df[metrics].values
        fig2 = go.Figure(go.Heatmap(
            z=z, x=metric_labels, y=team_df["Team"].tolist(),
            colorscale=[[0, "#0a0a0f"], [0.5, "#4a0a0a"], [1, "#ff1e00"]],
            text=[[f"{v:.1f}" for v in row] for row in z],
            texttemplate="%{text}",
        ))
        dark_layout(fig2, "Team Performance Heatmap", 360)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        impact = {
            "McLaren": 1.5, "Mercedes": 1.2, "Ferrari": 0.8, "Red Bull Racing": -0.5,
            "Williams": 0.6, "Aston Martin": -0.3, "Racing Bulls": -0.2, "Alpine": 0.2,
            "Haas": 0.1, "Audi": -2.5, "Cadillac": -3.0,
        }
        a1, a2 = st.columns(2)
        with a1:
            teams_i = list(impact.keys())
            vals_i = list(impact.values())
            colors_i = [TEAM_COLORS.get(t, "#888") if v >= 0 else "#550000" for t, v in impact.items()]
            fig3 = go.Figure(go.Bar(
                x=teams_i, y=vals_i, marker_color=colors_i,
                text=[f"{'+' if v > 0 else ''}{v:.1f}" for v in vals_i],
                textposition="outside",
            ))
            dark_layout(fig3, "Estimated Regulation Gain/Loss (grid positions)", 380)
            fig3.update_layout(xaxis_tickangle=-40)
            fig3.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dash")
            st.plotly_chart(fig3, use_container_width=True)
        with a2:
            aa_teams = sorted(TEAM_CAR_RATINGS, key=lambda t: -TEAM_CAR_RATINGS[t]["active_aero_mastery"])
            aa_vals = [TEAM_CAR_RATINGS[t]["active_aero_mastery"] for t in aa_teams]
            aa_colors = [TEAM_COLORS.get(t, "#888") for t in aa_teams]
            fig4 = go.Figure(go.Bar(
                x=aa_teams, y=aa_vals, marker_color=aa_colors,
                text=[f"{v:.1f}" for v in aa_vals], textposition="outside",
            ))
            dark_layout(fig4, "Active Aero Mastery Rating (2026 key metric)", 380)
            fig4.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig4, use_container_width=True)

        regs_detail = [
            (f'{get_icon("zap", 16, "#ff6b00", 5)} Active Aero — X-Mode',
             "On straights: wings flatten for minimum drag. Available simultaneously to all cars — not gated by 1-second gap. Top speeds up significantly."),
            (f'{get_icon("zap", 16, "#27F4D2", 5)} Active Aero — Z-Mode',
             "In corners: wings pitch back for max downforce. The system auto-transitions, fully replacing DRS."),
            (f'{get_icon("zap", 16, "#FFD700", 5)} Overtake Mode',
             "Within 1s of the car ahead, an extra 0.5 MJ unlocked for the following lap. New overtaking philosophy."),
            (f'{get_icon("building", 16, "#ff6b00", 5)} New Teams Reality',
             "Audi and Cadillac are entirely new operations. Historical data: new teams take 2-3 seasons to reach midfield. Expect P9-11 in 2026."),
        ]
        st.markdown("<div class='section-header'>KEY CHANGES EXPLAINED</div>", unsafe_allow_html=True)
        for i in range(0, 4, 2):
            rc1, rc2 = st.columns(2)
            for rcol, (title, desc) in zip([rc1, rc2], regs_detail[i:i + 2]):
                rcol.markdown(
                    f"<div class='reg-card' style='text-align:left'><h4>{title}</h4>"
                    f"<p style='margin-top:0.5rem;font-size:0.82rem'>{desc}</p></div>",
                    unsafe_allow_html=True,
                )

    with tab3:
        from src.data_collection import CIRCUIT_TYPE_MULTIPLIERS
        cal_df = pd.DataFrame(CALENDAR_2026)
        ct_counts = cal_df["circuit_type"].value_counts().reset_index()
        ct_counts.columns = ["Type", "Count"]
        ct_colors_map = {"power": "#ff6b00", "high_speed": "#ff1e00", "street": "#FF87BC",
                         "technical": "#27F4D2", "balanced": "#aaa"}

        cia1, cia2 = st.columns(2)
        with cia1:
            fig5 = px.pie(ct_counts, names="Type", values="Count",
                          color="Type", color_discrete_map=ct_colors_map, hole=0.45)
            dark_layout(fig5, "Calendar by Circuit Type", 340)
            fig5.update_traces(textinfo="label+percent", textfont=dict(family="Space Grotesk"))
            st.plotly_chart(fig5, use_container_width=True)

        with cia2:
            ctype_list = list(CIRCUIT_TYPE_MULTIPLIERS.keys())
            team_list = list(TEAM_CAR_RATINGS.keys())
            matrix = [
                [CIRCUIT_TYPE_MULTIPLIERS.get(ct, {}).get(team, 1.0) for ct in ctype_list]
                for team in team_list
            ]
            fig6 = go.Figure(go.Heatmap(
                z=matrix, x=[c.replace("_", " ").title() for c in ctype_list], y=team_list,
                colorscale=[[0, "#0a0a0f"], [0.5, "#1a1a2e"], [1, "#ff1e00"]],
                text=[[f"{v:.2f}" for v in row] for row in matrix],
                texttemplate="%{text}",
            ))
            dark_layout(fig6, "Circuit Type Performance Multiplier", 340)
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown("<div class='section-header'>SPRINT WEEKENDS 2026</div>", unsafe_allow_html=True)
        sprint_df = cal_df[cal_df["sprint"] == True][["round", "name", "circuit", "date"]].copy()
        sprint_df["date"] = sprint_df["date"].astype(str)
        sprint_df = sprint_df.rename(columns={"round": "Round", "name": "Grand Prix",
                                               "circuit": "Circuit", "date": "Date"})
        st.dataframe(sprint_df, use_container_width=True, hide_index=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    engine = load_engine()
    page = render_sidebar(engine)

    if page == "Home":
        show_home()
    elif page == "2026 Calendar":
        show_calendar()
    elif page == "Race Predictor":
        show_race_predictor(engine)
    elif page == "Season Simulator":
        show_season_simulator(engine)
    elif page == "Driver Profiles":
        show_driver_profiles()
    elif page == "Analytics":
        show_analytics()
    show_footer()


if __name__ == "__main__":
    main()
