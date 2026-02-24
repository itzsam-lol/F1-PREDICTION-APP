"""
Patch app.py:
1. Insert CIRCUIT_COORDS, get_map_embed(), and weekend schedule templates
   right after the CIRCUIT_DETAILS dict (before the track SVG section).
2. Replace the entire show_calendar() function with the new premium version.
"""
import re

with open("app.py", "r", encoding="utf-8") as f:
    src = f.read()

# ── 1. INSERT NEW DATA BLOCK BEFORE track SVG section ──────────────────────
NEW_DATA_BLOCK = '''
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
    "Interlagos":   (-23.7036,  -46.6997, 14),
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
        f"<div style=\'border-radius:12px;overflow:hidden;"
        f"border:1px solid rgba(255,255,255,0.10);"
        f"box-shadow:0 6px 20px rgba(0,0,0,0.55);position:relative\'>"
        f"<iframe src=\'{src}\' width=\'100%\' height=\'{height}px\' "
        f"style=\'border:none;display:block;"
        f"filter:brightness(0.88) saturate(0.78) contrast(1.08)\' "
        f"loading=\'lazy\' title=\'{circuit_name} map\'></iframe>"
        f"<div style=\'position:absolute;bottom:4px;right:6px;font-size:0.46rem;"
        f"color:#999;background:rgba(0,0,0,0.6);padding:1px 5px;border-radius:3px;"
        f"pointer-events:none\'>\u00a9 OpenStreetMap</div></div>"
    )

# ─── RACE WEEKEND SCHEDULE TEMPLATES ─────────────────────────────────────────
_NORMAL_WKD = [
    ("FRI", "Practice 1",          "#4a4a5a"),
    ("FRI", "Practice 2",          "#4a4a5a"),
    ("SAT", "Practice 3",          "#4a4a5a"),
    ("SAT", "Qualifying",          "#ff6b00"),
    ("SUN", "Race  \U0001f3c1",    "#ff1e00"),
]
_SPRINT_WKD = [
    ("FRI", "Practice 1",               "#4a4a5a"),
    ("FRI", "Sprint Qualifying",         "#9b30ff"),
    ("SAT", "Sprint Race  \U0001f680",   "#9b30ff"),
    ("SAT", "Qualifying",               "#ff6b00"),
    ("SUN", "Race  \U0001f3c1",         "#ff1e00"),
]

'''

# Insert before the track SVG section
marker = "# --- TRACK SVG MAPS"
assert marker in src, f"Marker not found: {marker}"
src = src.replace(marker, NEW_DATA_BLOCK + marker, 1)

# ── 2. REPLACE show_calendar() ─────────────────────────────────────────────
NEW_CALENDAR_FN = '''
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
        sprint_tag = " <span style=\'background:#9b30ff;color:#fff;font-size:0.52rem;font-weight:800;padding:0.1rem 0.4rem;border-radius:5px;vertical-align:middle\'>SPRINT</span>" if next_race["sprint"] else ""
        status_badge = (
            f"<div style=\'margin-top:0.7rem;display:inline-flex;align-items:center;gap:0.6rem;"
            f"background:rgba(255,30,0,0.08);border:1px solid rgba(255,30,0,0.2);"
            f"border-radius:20px;padding:0.35rem 0.9rem;font-size:0.78rem;color:#ccc\'>"
            f"{get_icon(\'compass\', 14, \'#ff6b00\', 6)}"
            f"<span>Next: <b style=\'color:#fff\'>{next_race[\'name\']}</b>{sprint_tag}</span>"
            f"<span style=\'color:#ff1e00;font-family:Orbitron;font-weight:900\'>"
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
              <div style='font-family:Orbitron;font-size:1.6rem;font-weight:900;color:#ff1e00'>{races_done}</div>
              <div style='font-size:0.6rem;color:#555;text-transform:uppercase;letter-spacing:0.1em'>Completed</div>
            </div>
            <div style='text-align:center'>
              <div style='font-family:Orbitron;font-size:1.6rem;font-weight:900;color:#27F4D2'>{races_left}</div>
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

    st.markdown("<div style=\'height:0.3rem\'></div>", unsafe_allow_html=True)
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

    # We\'ll render cards in groups of 2 (two-column grid)
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
            <div style=\'display:flex;align-items:center;gap:0.8rem;
                        margin:1.6rem 0 0.7rem 0\'>
              <div style=\'font-family:Orbitron;font-size:0.75rem;font-weight:900;
                          color:#fff;letter-spacing:0.15em;text-transform:uppercase\'>
                {current_month}
              </div>
              <div style=\'flex:1;height:1px;background:linear-gradient(90deg,rgba(255,255,255,0.12),transparent)\'></div>
              <div style=\'font-size:0.6rem;color:#444\'>
                {race_count} race{"s" if race_count != 1 else ""}
                {f"&nbsp;&middot;&nbsp;<span style=\'color:#9b30ff\'>{sprint_count} sprint</span>" if sprint_count else ""}
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
    st.markdown("<div class=\'section-header\' style=\'margin-top:2.5rem\'>SEASON TIMELINE</div>",
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
        "<span style=\'background:linear-gradient(90deg,#9b30ff,#6600cc);"
        "color:#fff;font-size:0.5rem;font-weight:800;letter-spacing:0.1em;"
        "padding:0.15rem 0.45rem;border-radius:5px;margin-left:0.4rem;"
        "vertical-align:middle;text-transform:uppercase\'>SPRINT</span>"
        if race["sprint"] else ""
    )

    border_color = "#ff1e00" if is_next else ("rgba(255,255,255,0.04)" if is_past else "rgba(255,255,255,0.08)")
    past_opacity = "opacity:0.55;" if is_past else ""
    glow = "box-shadow:0 0 24px rgba(255,30,0,0.18);" if is_next else ""

    with col:
        # ── Card header (always visible) ──────────────────────────────────
        st.markdown(f"""
        <div style=\'background:rgba(18,18,30,0.85);
                    border:1px solid {border_color};border-radius:18px;
                    padding:1rem 1.1rem 0.7rem 1.1rem;
                    transition:all 0.25s;{glow}{past_opacity}margin-bottom:0.2rem\'>
          <div style=\'display:flex;align-items:flex-start;justify-content:space-between;gap:0.5rem\'>
            <div style=\'display:flex;align-items:center;gap:0.55rem;flex:1;min-width:0\'>
              <div style=\'background:linear-gradient(135deg,#ff1e00,#ff6b00);
                          color:#fff;font-family:Orbitron;font-size:0.62rem;
                          font-weight:900;padding:0.25rem 0.5rem;
                          border-radius:7px;letter-spacing:0.06em;flex-shrink:0\'>
                R{race["round"]}
              </div>
              {flag_e}
              <div style=\'min-width:0\'>
                <div style=\'font-weight:700;color:{"#888" if is_past else "#fff"};
                             font-size:0.9rem;white-space:nowrap;
                             overflow:hidden;text-overflow:ellipsis\'>
                  {race["name"]}{sprint_badge}
                </div>
                <div style=\'font-size:0.68rem;color:#444;margin-top:2px\'>
                  {race["circuit"]}
                </div>
              </div>
            </div>
            <div style=\'text-align:right;flex-shrink:0\'>
              <div style=\'font-size:0.75rem;color:{"#555" if is_past else "#bbb"};font-weight:500\'>
                {date_str}
              </div>
              <div style=\'display:inline-block;font-size:0.56rem;font-weight:700;
                          text-transform:uppercase;letter-spacing:0.08em;
                          padding:0.12rem 0.45rem;border-radius:999px;
                          border:1px solid {ct_color};color:{ct_color};
                          margin-top:0.3rem;opacity:{"0.5" if is_past else "1"}\'>
                {ct_label}
              </div>
              {\'<div style="font-size:0.6rem;color:#00ff88;font-weight:700;margin-top:0.2rem">✓ COMPLETED</div>\' if is_past else (\'<div style="font-size:0.6rem;color:#ff1e00;font-weight:700;margin-top:0.2rem;animation:pulse 1.5s infinite">◉ NEXT RACE</div>\' if is_next else "")}
            </div>
          </div>

          <!-- Location map thumbnail -->
          {get_map_embed(race["circuit"], height=165)}
        </div>
        """, unsafe_allow_html=True)

        # ── Expander: track SVG + stats + weekend schedule ───────────────
        with st.expander(f"Details — {race['circuit']}", expanded=False):
            svg_col, stat_col = st.columns([1, 1.3])
            details = CIRCUIT_DETAILS.get(race["circuit"], {})

            with svg_col:
                svg_html = get_track_svg(race["circuit"], color=ct_color)
                st.markdown(
                    f"<div style=\'display:flex;justify-content:center;"
                    f"padding:0.5rem 0\'>{svg_html}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style=\'text-align:center;font-family:Orbitron;"
                    f"font-size:0.6rem;color:{ct_color};letter-spacing:0.12em;"
                    f"text-transform:uppercase\'>{race[\'circuit\']}</div>",
                    unsafe_allow_html=True,
                )

            with stat_col:
                if details:
                    st.markdown(f"""
                    <div class=\'track-stat-grid\'>
                      <div class=\'track-stat\'>
                        <div class=\'track-stat-label\'>Track Length</div>
                        <div class=\'track-stat-value\'>{details.get("length","–")} km</div>
                      </div>
                      <div class=\'track-stat\'>
                        <div class=\'track-stat-label\'>DRS Zones</div>
                        <div class=\'track-stat-value\'>{details.get("drs_zones","–")}</div>
                      </div>
                      <div class=\'track-stat\'>
                        <div class=\'track-stat-label\'>First GP</div>
                        <div class=\'track-stat-value\'>{details.get("first_gp","–")}</div>
                      </div>
                      <div class=\'track-stat\'>
                        <div class=\'track-stat-label\'>Laps</div>
                        <div class=\'track-stat-value\'>{race["laps"]}</div>
                      </div>
                    </div>
                    <div class=\'track-stat\' style=\'margin-top:0.5rem\'>
                      <div class=\'track-stat-label\'>Lap Record</div>
                      <div class=\'track-stat-value\' style=\'color:#ff6b00;font-size:0.78rem\'>
                        {details.get("lap_record","–")}
                      </div>
                    </div>
                    <div class=\'track-stat\' style=\'margin-top:0.5rem\'>
                      <div class=\'track-stat-label\'>Key Corner</div>
                      <div class=\'track-stat-value\' style=\'font-size:0.78rem\'>
                        {details.get("key_corner","–")}
                      </div>
                    </div>
                    <div style=\'margin-top:0.6rem;padding:0.55rem 0.8rem;
                                background:rgba(255,255,255,0.02);border-radius:8px;
                                border-left:3px solid {ct_color};
                                font-size:0.72rem;color:#aaa;line-height:1.55\'>
                      {details.get("characteristic","")}
                    </div>""", unsafe_allow_html=True)

            # Weekend schedule
            schedule = _SPRINT_WKD if race["sprint"] else _NORMAL_WKD
            st.markdown(
                f"<div style=\'margin-top:0.9rem;margin-bottom:0.3rem;"
                f"font-size:0.62rem;color:#444;text-transform:uppercase;"
                f"letter-spacing:0.12em;font-weight:700\'>"
                f"{get_icon(\'clock\',11,\'#444\',4)} Weekend Schedule</div>",
                unsafe_allow_html=True
            )
            rows_html = ""
            for day, session, color in schedule:
                rows_html += (
                    f"<div style=\'display:flex;align-items:center;gap:0.5rem;"
                    f"padding:0.28rem 0;border-bottom:1px solid rgba(255,255,255,0.04)\'>"
                    f"<span style=\'font-family:Orbitron;font-size:0.52rem;font-weight:700;"
                    f"color:#333;min-width:28px\'>{day}</span>"
                    f"<span style=\'width:3px;height:3px;border-radius:50%;"
                    f"background:{color};flex-shrink:0\'></span>"
                    f"<span style=\'font-size:0.72rem;color:{color};font-weight:600\'>"
                    f"{session}</span></div>"
                )
            st.markdown(f"<div style=\'padding:0 0.2rem\'>{rows_html}</div>",
                        unsafe_allow_html=True)

'''

# Replace the old show_calendar function
old_fn_start = "# ─── CALENDAR PAGE ───────────────────────────────────────────────────────────\ndef show_calendar():"
old_fn_end   = "# ─── RACE PREDICTOR PAGE ─────────────────────────────────────────────────────"

idx_start = src.find(old_fn_start)
idx_end   = src.find(old_fn_end)
assert idx_start != -1, "show_calendar start not found"
assert idx_end   != -1, "show_calendar end not found"

src = src[:idx_start] + NEW_CALENDAR_FN + "\n" + src[idx_end:]

with open("app.py", "w", encoding="utf-8") as f:
    f.write(src)

print("✅ Patch applied successfully!")
print(f"   Inserted CIRCUIT_COORDS, get_map_embed, weekend schedules")
print(f"   Replaced show_calendar() with premium card-grid version")
print(f"   New file length: {len(src.splitlines())} lines")
