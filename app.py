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
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dashboard Overhaul */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.95) !important;
        backdrop-filter: blur(25px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin: 10px;
        height: calc(100vh - 20px);
    }
    
    .f1-hero {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        position: relative;
    }
    .f1-hero h1 {
        font-family: 'Orbitron', monospace; font-size: 3rem; font-weight: 900;
        background: linear-gradient(90deg, #ff1e00, #ff6b00);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .f1-hero p { color: #888; margin-top: 1rem; font-size: 1.1rem; }

    .section-header {
        font-family: 'Orbitron', monospace; font-size: 1.1rem; font-weight: 700;
        color: #fff; border-left: 4px solid #ff1e00; border-radius: 2px;
        padding-left: 0.8rem; margin: 2rem 0 1.2rem 0;
        text-transform: uppercase; letter-spacing: 0.1em;
    }

    .metric-card {
        background: rgba(25, 25, 40, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #ff1e00; box-shadow: 0 15px 35px rgba(255, 30, 0, 0.15); }
    .metric-card .val { font-family: 'Orbitron'; font-size: 2.2rem; font-weight: 900; color: #ff1e00; }
    .metric-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; }

    .driver-card {
        background: rgba(30,30,45, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.2rem; margin: 0.5rem 0;
        transition: all 0.3s;
    }
    .driver-card:hover { border-color: rgba(255,255,255,0.15); transform: scale(1.02); }

    .race-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 1.2rem; margin: 0.7rem 0;
        display: flex; align-items: center; gap: 1.5rem;
        transition: all 0.3s;
    }
    .race-card:hover { background: rgba(255,255,255,0.05); border-color: rgba(255,30,0,0.3); }

    .podium-p1 { background: rgba(255,215,0,0.05); border: 2px solid #FFD700; border-radius: 24px; padding: 2rem; text-align: center; }
    .podium-p2 { background: rgba(192,192,192,0.05); border: 1px solid #C0C0C0; border-radius: 20px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
    .podium-p3 { background: rgba(205,127,50,0.03); border: 1px solid #CD7F32; border-radius: 20px; padding: 1.5rem; text-align: center; margin-top: 2rem; }

    .reg-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px; padding: 1.5rem; height: 100%; transition: all 0.3s;
    }
    .reg-card:hover { background: rgba(255,255,255,0.06); }

    .countdown-box {
        background: rgba(255,30,0,0.03); border: 1px solid #ff1e0033;
        border-radius: 24px; padding: 2rem; text-align: center;
    }
    .countdown-num { font-family: 'Orbitron'; font-size: 3.5rem; font-weight: 900; color: #fff; filter: drop-shadow(0 0 15px rgba(255,30,0,0.4)); }

    .stDataFrame { border-radius: 20px !important; }
    .stButton>button { border-radius: 12px !important; transition: all 0.3s; font-family: 'Orbitron'; }
    
    /* Remove Emojis from select labels etc */
    .stSelectbox label, .stMultiSelect label { font-family: 'Inter'; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.7rem !important; color: #555 !important; }
</style>

</style>
""", unsafe_allow_html=True)

# ─── ICONS ───────────────────────────────────────────────────────────────────
def get_icon(name: str, size: int = 24, color: str = "currentColor") -> str:
    """Helper to return SVG icons as HTML string."""
    icons = {
        "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline>',
        "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line>',
        "predictor": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
        "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"></path><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"></path><path d="M4 22h16"></path><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"></path><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"></path><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"></path>',
        "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle>',
        "analytics": '<line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line>',
        "sparkles": '<path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path><path d="m5 3 1 1"></path><path d="m19 3-1 1"></path><path d="m5 21 1-1"></path><path d="m19 21-1-1"></path>',
        "check": '<polyline points="20 6 9 17 4 12"></polyline>',
        "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>',
        "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>',
        "award": '<circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline>',
        "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>',
        "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline>'
    }
    path = icons.get(name, "")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;">{path}</svg>'

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
OPENING_RACE_DATE = date(2026, 3, 8)   # Australian GP

F1_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1280px-F1.svg.png"

TEAM_COLORS_HEX = TEAM_COLORS

FLAG_EMOJI = {
    "NED": "Netherlands","FRA": "France","GBR": "UK","ITA": "Italy","AUS": "Australia","ESP": "Spain",
    "GER": "Germany","BRA": "Brazil","MEX": "Mexico","FIN": "Finland","MON": "Monaco","CAN": "Canada",
    "NZL": "New Zealand","THA": "Thailand","ARG": "Argentina",
}

CIRCUIT_FLAG = {
    "AU":"Australia","CN":"China","JP":"Japan","BH":"Bahrain","SA":"Saudi Arabia","US":"USA",
    "CA":"Canada","MC":"Monaco","ES":"Spain","AT":"Austria","GB":"UK","BE":"Belgium",
    "HU":"Hungary","NL":"Netherlands","IT":"Italy","AZ":"Azerbaijan","SG":"Singapore","MX":"Mexico",
    "BR":"Brazil","QA":"Qatar","AE":"UAE",
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
        title=dict(text=title, font=dict(family="Orbitron", size=14, color="#fff")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#aaa"),
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
    "street":    ("🏙️", "#ff6b00"),
    "technical": ("⚙️", "#27F4D2"),
    "power":     ("⚡", "#FFD700"),
    "high_speed":("💨", "#ff1e00"),
}

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
        # F1 Logo
        st.markdown(f"""
        <div style='text-align:center; padding: 1.5rem 0 1rem 0'>
            <img src='{F1_LOGO_URL}' style='width:140px; filter: drop-shadow(0 0 15px rgba(255,30,0,0.3))'>
            <div style='font-size:0.65rem; color:#ff1e00; letter-spacing:0.3em; text-transform:uppercase; margin-top:0.8rem; font-weight:700'>
                Intelligence Hub
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 1.5rem'></div>", unsafe_allow_html=True)
        
        # Navigation
        st.markdown(f"<div style='font-size:0.7rem; color:#555; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:0.1em; padding-left:10px'>System Menu</div>", unsafe_allow_html=True)
        
        nav_options = {
            "Home": "home",
            "2026 Calendar": "calendar",
            "Race Predictor": "predictor",
            "Season Simulator": "trophy",
            "Driver Profiles": "user",
            "Analytics": "analytics"
        }
        
        # Use a hidden radio but custom-looking links
        page_choice = st.radio(
            "Navigate", 
            list(nav_options.keys()),
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid rgba(255,255,255,0.05)'></div>", unsafe_allow_html=True)
        
        # Season countdown
        st.markdown("<div style='font-size:0.7rem; color:#555; text-align:center; letter-spacing:0.1em; text-transform:uppercase'>2026 Neutral Split</div>", unsafe_allow_html=True)
        days_left = (OPENING_RACE_DATE - date.today()).days
        if days_left > 0:
            st.markdown(
                f"<div style='text-align:center; font-family:Orbitron; font-size:2rem; font-weight:900; "
                f"color:#ff1e00; text-shadow:0 0 20px rgba(255,30,0,0.4)'>{days_left}</div>"
                f"<div style='text-align:center; font-size:0.6rem; color:#555; letter-spacing:0.15em'>DAYS TO MELBOURNE</div>",
                unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; color:#00ff88; font-weight:700; font-family:Orbitron; font-size:0.85rem'>LIVE ROUND ACTIVE</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 1.5rem 0; border-top: 1px solid rgba(255,255,255,0.05)'></div>", unsafe_allow_html=True)
        
        # Model status
        st.markdown("<div style='font-size:0.7rem; color:#555; text-align:center; letter-spacing:0.1em; text-transform:uppercase'>ML Engine Status</div>", unsafe_allow_html=True)
        if not engine.is_trained:
            if st.button("Initialize Model", use_container_width=True):
                with st.spinner("Processing Stacking Ensemble..."):
                    engine.quick_demo_train()
                st.rerun()
        else:
            best = engine.ml_models.best_model_name
            res  = engine.ml_models.training_results
            mae  = res.get(best, {}).get("mae", "–")
            r2   = res.get(best, {}).get("r2", "–")
            st.markdown(
                f"<div style='text-align:center; color:#00ff88; font-family:Orbitron; font-size:0.8rem; font-weight:700; margin-top:0.5rem'>{get_icon('check', 14, '#00ff88')} STACKING ACTIVE</div>"
                f"<div style='text-align:center; font-size:0.65rem; color:#777; margin-top:0.3rem'>"
                f"MAE {mae} pos &nbsp;·&nbsp; R² {r2}</div>",
                unsafe_allow_html=True)
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
            <div style='font-family:Orbitron; font-size:0.8rem; color:#ff6b00; letter-spacing:0.15em; margin-bottom:0.8rem'>COUNTDOWN TO AUSTRALIAN GP &middot; MARCH 8</div>
            <span class='countdown-num'>{weeks}</span><span class='countdown-label'>WKS</span>
            <span class='countdown-num'>{days}</span><span class='countdown-label'>DAYS</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='countdown-box' style='color:#00ff88; font-family:Orbitron;'>🏁 THE 2026 F1 SEASON IS UNDERWAY!</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='section-header'>🔧 THE NEW ERA — 2026 REGULATIONS</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='section-header'>🆕 THE CLASS OF 2026</div>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown("""
        <div class='driver-card' style='border-color:#9B000055'>
            <div style='color:#9B0000; font-family:Orbitron; font-weight:700'>AUDI FORMULA RACING</div>
            <div style='font-size:0.85rem; color:#ccc; margin-top:0.3rem'>Nico Hülkenberg · Gabriel Bortoleto</div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.2rem'>Power Unit: Audi · First works entry in F1</div>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown("""
        <div class='driver-card' style='border-color:#00308755'>
            <div style='color:#003087; font-family:Orbitron; font-weight:700'>CADILLAC FORMULA RACING</div>
            <div style='font-size:0.85rem; color:#ccc; margin-top:0.3rem'>Sergio Pérez · Valtteri Bottas</div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.2rem'>Power Unit: Ferrari · First US works team</div>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        st.markdown(f"""
        <div class='driver-card' style='border-color:#ff1e0055'>
            <div style='color:#ff6b00; font-family:Orbitron; font-weight:700'>DEFENDING CHAMPION</div>
            <div style='font-size:0.85rem; color:#ccc; margin-top:0.3rem'>Lando Norris &middot; McLaren (2025 WDC)</div>
            <div style='font-size:0.75rem; color:#888; margin-top:0.2rem'>McLaren also won 2025 Constructors' Championship</div>
        </div>
        """, unsafe_allow_html=True)

    # Team grid
    st.markdown("<div class='section-header'>🏎️ CONSTRUCTOR LINEUP</div>", unsafe_allow_html=True)
    team_data = []
    for info in DRIVERS_2026:
        team = info["team"]
        team_data.append({"team": team, "driver": info["name"], "code": info["code"]})
    team_df = pd.DataFrame(team_data)
    teams_grouped = team_df.groupby("team")["driver"].apply(lambda x: " & ".join(x)).reset_index()

    fig = go.Figure()
    for i, row in teams_grouped.iterrows():
        color = TEAM_COLORS_HEX.get(row["team"], "#888")
        fig.add_trace(go.Bar(
            x=[row["team"]],
            y=[1],
            name=row["team"],
            marker_color=color,
            text=[row["driver"]],
            textposition="inside",
            hovertemplate=f"<b>{row['team']}</b><br>{row['driver']}<extra></extra>",
        ))
    fig.update_layout(
        showlegend=False, barmode="stack",
        xaxis_tickangle=-30, height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#aaa"),
        margin=dict(t=10, b=80, l=20, r=20),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── CALENDAR PAGE ───────────────────────────────────────────────────────────
def show_calendar():
    st.markdown("""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">📅 2026 RACE CALENDAR</h1>
        <p>24 Grands Prix across 5 continents · 6 Sprint weekends</p>
    </div>""", unsafe_allow_html=True)

    cal_df = pd.DataFrame(CALENDAR_2026)
    cal_df["date"] = pd.to_datetime(cal_df["date"])

    # Filter controls
    fc1, fc2 = st.columns(2)
    with fc1:
        ct_filter = st.multiselect("Circuit Type", ["all types"] + list(cal_df["circuit_type"].unique()), default=["all types"])
    with fc2:
        sprint_filter = st.checkbox("Sprint weekends only", value=False)

    filtered = cal_df.copy()
    if sprint_filter:
        filtered = filtered[filtered["sprint"] == True]
    if ct_filter and "all types" not in ct_filter:
        filtered = filtered[filtered["circuit_type"].isin(ct_filter)]

    for _, race in filtered.iterrows():
        country_name = CIRCUIT_FLAG.get(race["country"], "International")
        sprint_badge = "<span class='race-sprint-badge'>SPRINT</span>" if race["sprint"] else ""
        ct_color = {"power":"#ff6b00","high_speed":"#ff1e00","street":"#FF87BC","technical":"#27F4D2","balanced":"#aaa"}.get(race["circuit_type"],"#888")
        st.markdown(f"""
        <div class='race-card'>
            <div class='race-round'>R{race["round"]}</div>
            <div style='flex:1'>
                <div style='font-weight:600; color:#fff; font-size:1rem'>{race["name"]} {sprint_badge}</div>
                <div style='font-size:0.8rem; color:#888; margin-top:0.1rem'>{race["circuit"]} &middot; {country_name}</div>
            </div>
            <div style='text-align:right'>
                <div style='font-size:0.85rem; color:#ccc'>{race["date"].strftime("%b %d, %Y")}</div>
                <div style='font-size:0.7rem; color:{ct_color}; font-weight:600; text-transform:uppercase; margin-top:0.1rem'>{race["type"].replace("_"," ")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Calendar chart
    st.markdown("<div class='section-header'>SEASON VISUALIZATION</div>", unsafe_allow_html=True)
    months = cal_df["date"].dt.month_name().unique()
    month_counts = cal_df.groupby(cal_df["date"].dt.month_name()).size().reindex(
        ["March","April","May","June","July","August","September","October","November","December"], fill_value=0
    ).reset_index()
    month_counts.columns = ["Month", "Races"]

    fig = px.bar(month_counts, x="Month", y="Races", color="Races",
                 color_continuous_scale=["#1a1a2e","#ff6b00","#ff1e00"])
    dark_layout(fig, "Races Per Month", 300)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ─── RACE PREDICTOR PAGE ─────────────────────────────────────────────────────
def show_race_predictor(engine):
    st.markdown(f"""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">{get_icon('predictor', 32, '#ff1e00')} RACE PREDICTOR</h1>
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
    country_name = CIRCUIT_FLAG.get(race_info["country"], "International")

    if st.button(f"PREDICT {selected_race_name.upper()}", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations for {selected_race_name}..."):
            det_df, mc_df = engine.predict_race(
                race_info["circuit"], include_monte_carlo=True, n_sims=n_sims
            )

        ct_color = {
            "power": "#ff6b00", "high_speed": "#ff1e00", "street": "#FF87BC",
            "technical": "#27F4D2", "balanced": "#aaa"
        }.get(race_info["type"], "#888")
        sprint_badge = "<span class='race-sprint-badge'>SPRINT WEEKEND</span>" if race_info["sprint"] else ""

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d0d1a,#15152a);border:1px solid #ffffff18;
                    border-radius:20px;padding:1.5rem 2rem;margin-bottom:2rem;
                    display:flex;gap:1.5rem;align-items:center'>
            <div style='flex:1'>
                <div style='font-family:Orbitron;font-size:1.4rem;font-weight:900;color:#fff'>{selected_race_name}</div>
                <div style='color:#888;font-size:0.9rem'>{race_info["circuit"]} &middot; {country_name} &middot;
                <span style='color:{ct_color}'>{race_info["type"].replace("_"," ").title()}</span></div>
                {sprint_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            col.markdown(f"""
            <div class='{medal_cls}'>
                <div style='margin-bottom:0.5rem'>{get_icon('trophy', 32 if big else 24, icon_color)}</div>
                <div style='font-family:Orbitron; font-size:0.7rem; font-weight:700; color:{icon_color}; margin-bottom:0.4rem'>{medal_label}</div>
                <h2 style='{"font-size:1.8rem;" if big else "font-size:1.4rem;"}'>{d["driver_code"]}</h2>
                <p style='font-size:{"0.95rem" if big else "0.8rem"};{"font-weight:600;" if big else ""}'>
                    {d["driver_name"]}</p>
                <div style='display:inline-block;background:{tc};color:#fff;padding:0.1rem 0.5rem;
                    border-radius:6px;font-size:0.65rem;font-weight:700'>{d["team"]}</div>
                <p style='margin-top:0.8rem;font-size:{"1.1rem" if big else "0.9rem"}'>
                    Win <b>{d["win_prob"]:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

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

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Full Grid", "Lap-by-Lap Trace", "Position Heatmap"])

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
            if show_laps:
                with st.spinner("Generating lap trace..."):
                    if "predicted_pos" not in det_df.columns and "expected_pos" in mc_df.columns:
                        sim_input = mc_df.copy()
                        sim_input["predicted_pos"] = sim_input["expected_pos"].round().astype(int)
                    else:
                        sim_input = det_df
                    lap_df = engine.generate_lap_simulation(sim_input, n_laps=57)

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
                        hovertemplate=f"<b>{code}</b> — Lap %{{x}}: P%{{y:.0f}}<extra></extra>",
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
                <div style='font-family:Orbitron;font-weight:900;color:{tc};font-size:1rem'>{row["driver_code"]}</div>
                <div style='font-size:0.75rem;color:#888'>{row["driver_name"]}</div>
                <div style='font-family:Orbitron;font-size:1.8rem;color:#fff;font-weight:900;margin-top:0.4rem'>
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
        st.markdown("<div class='section-header'>🏗️ CONSTRUCTORS' CHAMPIONSHIP</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-header'>📋 FULL STANDINGS TABLE</div>", unsafe_allow_html=True)
        disp = standings[["position", "driver_code", "driver_name", "team", "points", "wins", "podiums"]].copy()
        disp["points"] = disp["points"].astype(int)
        disp = disp.rename(columns={
            "position": "Pos", "driver_code": "Code", "driver_name": "Driver",
            "team": "Team", "points": "Pts", "wins": "Wins", "podiums": "Podiums",
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ─── DRIVER PROFILES PAGE ─────────────────────────────────────────────────────
def show_driver_profiles():
    st.markdown("""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">👤 DRIVER PROFILES</h1>
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
                            <div style='font-size:1.4rem;font-weight:900;color:{tc};font-family:Orbitron'>
                                {driver["code"]}{rookie_label}</div>
                            <div style='font-size:0.88rem;color:#ccc;font-weight:600'>{driver["name"]}</div>
                            <div style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.05em'>
                                {driver["team"]} &middot; {driver["nationality"]}</div>
                        </div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-family:Orbitron;font-size:1.8rem;font-weight:900;color:#ff1e00'>{driver["overall"]:.1f}</div>
                        <div style='font-size:0.6rem;color:#555;text-transform:uppercase'>Rating</div>
                    </div>
                </div>
                <div style='margin-top:1rem'>{bars_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📊 DRIVER COMPARISON RADAR</div>", unsafe_allow_html=True)
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
            ("⚡ Active Aero — X-Mode",
             "On straights: wings flatten for minimum drag. Available simultaneously to all cars — not gated by 1-second gap. Top speeds up significantly."),
            ("⚡ Active Aero — Z-Mode",
             "In corners: wings pitch back for max downforce. The system auto-transitions, fully replacing DRS."),
            ("🔋 Overtake Mode",
             "Within 1s of the car ahead, an extra 0.5 MJ unlocked for the following lap. New overtaking philosophy."),
            ("🏭 New Teams Reality",
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
            fig5.update_traces(textinfo="label+percent", textfont=dict(family="Inter"))
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
