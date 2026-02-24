import pandas as pd
import numpy as np
from .data_collection import F1DataCollector, CALENDAR_2026
from .feature_engineering import FeatureEngineer
from .ml_models import F1PredictionModels


class F1PredictionEngine:
    def __init__(self):
        self.data_collector = F1DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.ml_models = F1PredictionModels()
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def quick_demo_train(self):
        """Train on synthetic historical data — no external API needed"""
        hist_df = self.data_collector.collect_historical_data(year_range=(2022, 2025))
        eng_df  = self.feature_engineer.engineer_features(hist_df)
        feat_cols = self.feature_engineer.get_feature_columns(has_grid=True)
        feat_cols = [c for c in feat_cols if c in eng_df.columns]
        X, y = self.ml_models.prepare_training_data(eng_df, feat_cols)
        results = self.ml_models.train_models(X, y)
        self.is_trained = True
        return results

    def train_full(self):
        """Alias for quick_demo_train (no live API calls in 2026 pre-season)"""
        return self.quick_demo_train()

    # ── Single Race Prediction ─────────────────────────────────────────────────

    def predict_race(self, circuit_name: str, include_monte_carlo: bool = True, n_sims: int = 5000):
        """
        Predict a race at the given circuit.
        Returns (deterministic_df, monte_carlo_df or None)
        """
        if not self.is_trained:
            raise ValueError("Train the model first!")

        race_df  = self.data_collector.get_race_features(circuit_name)
        eng_df   = self.feature_engineer.engineer_features(race_df)
        feat_cols= [c for c in self.ml_models.feature_columns if c in eng_df.columns]

        # Ensure columns are aligned
        for col in self.ml_models.feature_columns:
            if col not in eng_df.columns:
                eng_df[col] = 0.0

        determ = self.ml_models.predict_race_outcome(eng_df)

        mc = None
        if include_monte_carlo:
            mc = self.ml_models.monte_carlo_race(eng_df, n_sims=n_sims)

        return determ, mc

    def generate_lap_simulation(self, race_result_df: pd.DataFrame, n_laps: int = 57):
        """
        Generate a plausible lap-by-lap position trace for visualization.
        Uses predicted final order + Gaussian noise to simulate position changes.
        """
        np.random.seed(99)
        drivers = race_result_df["driver_code"].tolist()
        names   = race_result_df["driver_name"].tolist()
        teams   = race_result_df["team"].tolist()
        colors  = race_result_df["team_color"].tolist()
        final_pos = race_result_df["predicted_pos"].tolist() if "predicted_pos" in race_result_df.columns else list(range(1, len(drivers) + 1))

        # Start positions slightly shuffled from qualfiying (assumed ~final order)
        start_pos = list(range(1, len(drivers) + 1))
        np.random.shuffle(start_pos)

        traces = {d: [start_pos[i]] for i, d in enumerate(drivers)}

        for lap in range(1, n_laps + 1):
            frac = lap / n_laps
            for i, driver in enumerate(drivers):
                target = final_pos[i]
                current = traces[driver][-1]
                # Drift toward target with noise
                noise = np.random.normal(0, 1.2 * (1 - frac) + 0.3)
                new_pos = current + (target - current) * 0.08 + noise
                new_pos = np.clip(new_pos, 1, len(drivers))
                traces[driver].append(round(new_pos, 2))

        # Build long-form DataFrame for Plotly
        rows = []
        for lap in range(n_laps + 1):
            for i, driver in enumerate(drivers):
                rows.append({
                    "lap": lap,
                    "driver_code": driver,
                    "driver_name": names[i],
                    "team": teams[i],
                    "team_color": colors[i],
                    "position": traces[driver][lap],
                })
        return pd.DataFrame(rows)

    # ── Season Simulation ──────────────────────────────────────────────────────

    def predict_full_season(self):
        """Predict all 24 races and return championship standings"""
        if not self.is_trained:
            raise ValueError("Train the model first!")

        all_race_data = []
        for race in CALENDAR_2026:
            circuit = race["circuit"]
            race_df = self.data_collector.get_race_features(circuit)
            eng_df  = self.feature_engineer.engineer_features(race_df)
            for col in self.ml_models.feature_columns:
                if col not in eng_df.columns:
                    eng_df[col] = 0.0
            all_race_data.append((race["name"], eng_df))

        return self.ml_models.predict_full_season(all_race_data)

    def get_championship_snapshot(self, up_to_round: int = 24):
        """Race-by-race cumulative points for animation"""
        if not self.is_trained:
            raise ValueError("Train the model first!")

        POINTS   = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        driver_list = self.data_collector.drivers  # list of dicts
        driver_codes = [d["code"] for d in driver_list]
        cumul    = {code: 0 for code in driver_codes}
        timeline = []

        for race in CALENDAR_2026[:up_to_round]:
            circuit  = race["circuit"]
            race_df  = self.data_collector.get_race_features(circuit)
            eng_df   = self.feature_engineer.engineer_features(race_df)
            for col in self.ml_models.feature_columns:
                if col not in eng_df.columns:
                    eng_df[col] = 0.0
            mc = self.ml_models.monte_carlo_race(eng_df, n_sims=1000)

            for _, row in mc.iterrows():
                pos = int(round(row["expected_pos"]))
                pts = POINTS.get(pos, 0)
                cumul[row["driver_code"]] = cumul.get(row["driver_code"], 0) + pts

            snapshot = {"race": race["name"], "round": race["round"]}
            snapshot.update(cumul.copy())
            timeline.append(snapshot)

        return pd.DataFrame(timeline)
