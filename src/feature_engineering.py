import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self._feature_cols = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # ── Driver composite ─────────────────────────────────────────────────
        data["driver_composite"] = (
            data["driver_pace"] * 0.30 +
            data["driver_consistency"] * 0.20 +
            data["driver_qualifying"] * 0.20 +
            data["driver_overtaking"] * 0.15 +
            data["driver_wet_skill"] * 0.08 +
            data["driver_reg_adaptation"] * 0.07
        )

        # ── Car composite ────────────────────────────────────────────────────
        data["car_composite"] = (
            data["car_speed"] * 0.35 +
            data["car_reliability"] * 0.25 +
            data["car_aero"] * 0.25 +
            data["car_active_aero"] * 0.15
        )

        # ── Combined performance score ───────────────────────────────────────
        data["combined_score"] = data["driver_composite"] * 0.55 + data["car_composite"] * 0.45

        # ── Real 2024 data features ──────────────────────────────────────────
        if "real_avg_finish_2024" in data.columns:
            data["historical_form"] = data["real_avg_finish_2024"]
            data["historical_wins"] = data.get("real_wins_2024", 0)
            data["historical_podiums"] = data.get("real_podiums_2024", 0)
            data["historical_dnf_rate"] = data.get("real_dnf_rate", 0.1)
        else:
            data["historical_form"] = 10.0
            data["historical_wins"] = 0
            data["historical_podiums"] = 0
            data["historical_dnf_rate"] = 0.1

        # Normalize historical form (lower avg finish = better, invert to score)
        data["form_score"] = (20.0 - data["historical_form"]) / 20.0 * 10.0

        # ── Circuit-specific features ────────────────────────────────────────
        if "circuit_type_mult" in data.columns:
            data["circuit_multiplier"] = data["circuit_type_mult"]
        else:
            data["circuit_multiplier"] = 1.0

        if "grid_finish_corr" in data.columns:
            data["grid_importance"] = data["grid_finish_corr"]
        else:
            data["grid_importance"] = 0.65

        # Grid position effect weighted by circuit type
        if "grid_position" in data.columns:
            data["grid_effect"] = data["grid_position"] * data["grid_importance"]
        else:
            data["grid_effect"] = 10.0

        # ── Penalties ────────────────────────────────────────────────────────
        new_team_base = data.get("new_team", pd.Series(0, index=data.index))
        data["new_team_penalty"] = new_team_base * 7.0   # ~P15-18 baseline

        rookie_base = data.get("rookie", pd.Series(0, index=data.index))
        data["rookie_penalty"] = rookie_base * 1.8

        # ── Active aero (2026 regulation) ────────────────────────────────────
        data["active_aero_score"] = data["car_active_aero"] * data["circuit_multiplier"]
        data["driver_reg_factor"] = data["driver_reg_adaptation"] * 0.8

        # ── Street circuit penalty for less experienced teams ────────────────
        is_street = data.get("is_street", pd.Series(0, index=data.index))
        data["street_penalty"] = is_street * new_team_base * 1.5

        # ── DNF risk score ────────────────────────────────────────────────────
        data["dnf_risk"] = (
            data["historical_dnf_rate"] * 0.6 +
            (1 - data["car_reliability"] / 10.0) * 0.4
        )

        # ── Expected position estimate (meta-feature) ────────────────────────
        # Lower = better
        data["expected_position_est"] = (
            data["historical_form"] * 0.40 +
            (11 - data["combined_score"]) * 0.30 +
            data["new_team_penalty"] * 0.20 +
            data["grid_effect"] * 0.10
        )

        # ── Race pace vs quali gap ────────────────────────────────────────────
        data["race_quali_delta"] = data["driver_pace"] - data["driver_qualifying"]

        # ── Finishing position label ─────────────────────────────────────────
        if "finish_position" in data.columns:
            data["finish_position"] = pd.to_numeric(data["finish_position"], errors="coerce")

        return data

    def get_feature_columns(self, has_grid=True):
        cols = [
            "driver_composite", "car_composite", "combined_score",
            "circuit_multiplier", "grid_importance", "active_aero_score",
            "driver_reg_factor", "new_team_penalty", "rookie_penalty",
            "street_penalty", "dnf_risk", "expected_position_est",
            "form_score", "historical_form", "historical_wins",
            "historical_podiums", "historical_dnf_rate", "race_quali_delta",
            "car_speed", "car_reliability", "car_aero", "car_active_aero",
            "driver_pace", "driver_consistency", "driver_qualifying",
            "driver_overtaking", "driver_wet_skill", "driver_reg_adaptation",
            "is_street", "new_team", "rookie",
        ]
        if has_grid:
            cols.append("grid_effect")
        return cols
