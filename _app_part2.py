
# ─── RACE PREDICTOR PAGE ─────────────────────────────────────────────────────
def show_race_predictor(engine):
    st.markdown("""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">🔮 RACE PREDICTOR</h1>
        <p>Select any 2026 Grand Prix · Monte Carlo simulation · Animated podium</p>
    </div>""", unsafe_allow_html=True)

    if not engine.is_trained:
        st.warning("⚠️ Train the model first using the sidebar button!")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        race_names = [r["name"] for r in CALENDAR_2026]
        selected_race_name = st.selectbox("🏁 Select Grand Prix", race_names)
    with col2:
        n_sims = st.select_slider("Monte Carlo Sims", [500, 1000, 2000, 5000], value=2000)
    with col3:
        show_laps = st.checkbox("Lap-by-Lap Trace", value=True)

    race_info = next(r for r in CALENDAR_2026 if r["name"] == selected_race_name)
    flag = CIRCUIT_FLAG.get(race_info["country"], "🌍")

    if st.button(f"⚡ PREDICT {selected_race_name.upper()}", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations for {selected_race_name}..."):
            det_df, mc_df = engine.predict_race(
                race_info["circuit"], include_monte_carlo=True, n_sims=n_sims
            )

        ct_color = {
            "power": "#ff6b00", "aero": "#27F4D2", "street": "#FF87BC",
            "street_hybrid": "#FFD700", "balanced": "#aaa"
        }.get(race_info["circuit_type"], "#888")
        sprint_badge = "<span class='race-sprint-badge'>SPRINT WEEKEND</span>" if race_info["sprint"] else ""

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d0d1a,#15152a);border:1px solid #ffffff18;
                    border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;
                    display:flex;gap:1rem;align-items:center'>
            <span style='font-size:2.5rem'>{flag}</span>
            <div>
                <div style='font-family:Orbitron;font-size:1.2rem;font-weight:900;color:#fff'>{selected_race_name}</div>
                <div style='color:#888;font-size:0.8rem'>{race_info["circuit"]} &middot; {race_info["location"]} &middot;
                <span style='color:{ct_color}'>{race_info["circuit_type"].replace("_"," ").title()}</span></div>
                {sprint_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Podium ceremony
        top3 = mc_df.head(3)
        st.markdown("<div class='section-header'>🥇 PREDICTED PODIUM</div>", unsafe_allow_html=True)
        _, p2c, p1c, p3c, _ = st.columns([0.5, 1, 1.2, 1, 0.5])

        for col, idx, medal_cls, medal_icon in [
            (p2c, 1, "podium-p2", "🥈"),
            (p1c, 0, "podium-p1", "🏆"),
            (p3c, 2, "podium-p3", "🥉"),
        ]:
            d = top3.iloc[idx]
            tc = d["team_color"]
            big = idx == 0
            col.markdown(f"""
            <div class='{medal_cls}'>
                <div style='font-size:{"2rem" if big else "1.5rem"}'>{medal_icon}</div>
                <h2 style='{"font-size:1.8rem;" if big else ""}'>{d["driver_code"]}</h2>
                <p style='font-size:{"0.95rem" if big else "0.8rem"};{"font-weight:600;" if big else ""}'>
                    {d["driver_name"]}</p>
                <div style='display:inline-block;background:{tc};color:#fff;padding:0.1rem 0.5rem;
                    border-radius:4px;font-size:0.65rem;font-weight:700'>{d["team"]}</div>
                <p style='margin-top:0.5rem;font-size:{"1rem" if big else "0.85rem"}'>
                    Win: <b>{d["win_prob"]:.1%}</b> &middot; Podium: <b>{d["podium_prob"]:.1%}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # Win probability chart
        st.markdown("<div class='section-header'>📊 WIN PROBABILITY</div>", unsafe_allow_html=True)
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
        tab1, tab2, tab3 = st.tabs(["📋 Full Grid", "📈 Lap-by-Lap Trace", "🎯 Position Heatmap"])

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
    st.markdown("""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">🏆 SEASON SIMULATOR</h1>
        <p>Full 2026 championship forecast across all 24 rounds</p>
    </div>""", unsafe_allow_html=True)

    if not engine.is_trained:
        st.warning("⚠️ Train the model first!")
        return

    if st.button("🚀 SIMULATE THE FULL 2026 SEASON", use_container_width=True):
        with st.spinner("Simulating all 24 races (~10s)..."):
            standings = engine.predict_full_season()

        st.success("✅ 2026 Championship simulation complete!")

        top5 = standings.head(5)
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        st.markdown("<div class='section-header'>🏆 WORLD CHAMPIONSHIP — TOP 5</div>", unsafe_allow_html=True)
        cols5 = st.columns(5)
        for i, (col, (_, row)) in enumerate(zip(cols5, top5.iterrows())):
            tc = row["team_color"]
            col.markdown(f"""
            <div class='driver-card' style='border-left:4px solid {tc};text-align:center'>
                <div style='font-size:1.4rem'>{medals[i]}</div>
                <div style='font-family:Orbitron;font-weight:900;color:{tc};font-size:1rem'>{row["driver_code"]}</div>
                <div style='font-size:0.72rem;color:#aaa'>{row["driver_name"]}</div>
                <div style='font-family:Orbitron;font-size:1.5rem;color:#ff1e00;font-weight:900;margin-top:0.3rem'>
                    {int(row["points"])} <span style='font-size:0.6rem;color:#888'>PTS</span>
                </div>
                <div style='font-size:0.7rem;color:#666'>{int(row["wins"])}W &middot; {int(row["podiums"])}Pd</div>
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

    all_teams = sorted(set(d["team"] for d in DRIVERS_2026.values()))
    sel_team = st.selectbox("Filter by team", ["All Teams"] + all_teams)

    rows = []
    for code, info in DRIVERS_2026.items():
        if sel_team != "All Teams" and info["team"] != sel_team:
            continue
        r = DRIVER_BASE_RATINGS.get(code, {})
        overall = round(
            r.get("pace", 7) * 0.30 + r.get("consistency", 7) * 0.20 +
            r.get("qualifying", 7) * 0.20 + r.get("overtaking", 7) * 0.15 +
            r.get("wet_skill", 7) * 0.10 + r.get("reg_adaptation", 7) * 0.05, 2
        )
        rows.append({"code": code, **info, **r, "overall": overall})
    rows.sort(key=lambda x: -x["overall"])

    for i in range(0, len(rows), 2):
        c1, c2 = st.columns(2)
        for col, driver in zip([c1, c2], rows[i:i + 2]):
            tc = TEAM_COLORS.get(driver["team"], "#888")
            flag = FLAG_EMOJI.get(driver["nationality"], "🌍")
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
                f"<div class='prob-bar-bg'><div class='prob-bar-fill' style='width:{driver.get(key, 7) / 10 * 100}%;background:{tc}'></div></div>"
                f"</div>"
                for key, label in skill_keys
            ])
            col.markdown(f"""
            <div class='driver-card' style='border-left:4px solid {tc}'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                    <div>
                        <div style='font-size:1.5rem;font-weight:900;color:{tc};font-family:Orbitron'>
                            {driver["code"]}{rookie_label}</div>
                        <div style='font-size:0.9rem;color:#ccc;font-weight:500'>{driver["name"]} {flag}</div>
                        <div style='font-size:0.72rem;color:#666'>#{driver["number"]} &middot; {driver["team"]}</div>
                    </div>
                    <div style='text-align:right'>
                        <div style='font-family:Orbitron;font-size:1.8rem;font-weight:900;color:{tc}'>{driver["overall"]:.1f}</div>
                        <div style='font-size:0.6rem;color:#555;text-transform:uppercase'>Rating</div>
                    </div>
                </div>
                <div style='margin-top:0.7rem'>{bars_html}</div>
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
                fill="toself", name=code, line_color=tc, fillcolor=tc + "33",
            ))
        dark_layout(fig, "Attribute Radar Comparison", 480)
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[5, 10], gridcolor="#ffffff15", tickfont=dict(color="#666")),
            angularaxis=dict(gridcolor="#ffffff15", tickfont=dict(color="#aaa")),
        ))
        st.plotly_chart(fig, use_container_width=True)


# ─── ANALYTICS PAGE ───────────────────────────────────────────────────────────
def show_analytics():
    st.markdown("""<div class="f1-hero" style="padding:1.5rem 2rem">
        <h1 style="font-size:1.8rem">📊 ANALYTICS</h1>
        <p>Team power rankings &middot; 2026 reg impact &middot; Circuit intelligence</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏎️ Team Power Rankings", "⚡ Regulation Impact", "🏁 Circuit Intel"])

    with tab1:
        st.markdown("<div class='section-header'>2026 PRE-SEASON TEAM POWER RANKINGS</div>", unsafe_allow_html=True)
        team_rows = []
        for team, ratings in TEAM_CAR_RATINGS.items():
            overall = round(sum(ratings.values()) / len(ratings), 2)
            team_rows.append({"Team": team, "Overall": overall, **ratings, "Color": TEAM_COLORS.get(team, "#888")})
        team_df = pd.DataFrame(team_rows).sort_values("Overall", ascending=False)

        metrics = ["base_pace", "aero_efficiency", "pu_power", "reliability", "active_aero_mastery"]
        metric_labels = ["Base Pace", "Aero Efficiency", "PU Power", "Reliability", "Active Aero"]

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
            fig3.add_hline(y=0, line_color="#ffffff33", line_dash="dash")
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
        ct_colors_map = {"power": "#ff6b00", "aero": "#27F4D2", "street": "#FF87BC",
                         "street_hybrid": "#FFD700", "balanced": "#aaa"}

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

    if page == "🏠 Home":
        show_home()
    elif page == "📅 2026 Calendar":
        show_calendar()
    elif page == "🔮 Race Predictor":
        show_race_predictor(engine)
    elif page == "🏆 Season Simulator":
        show_season_simulator(engine)
    elif page == "👤 Driver Profiles":
        show_driver_profiles()
    elif page == "📊 Analytics":
        show_analytics()


if __name__ == "__main__":
    main()
