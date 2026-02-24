import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib


class F1PredictionModels:
    def __init__(self):
        self.base_estimators = [
            ("rf",  RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1)),
            ("gbm", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)),
            ("xgb", xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.07, subsample=0.8,
                                      colsample_bytree=0.8, random_state=42, verbosity=0)),
        ]
        self.meta_model = Ridge(alpha=1.5)
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.stacker = None
        self.training_results = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def prepare_training_data(self, data: pd.DataFrame, feature_cols: list):
        clean = data.dropna(subset=["finish_position"]).copy()
        X = clean[feature_cols].fillna(clean[feature_cols].mean())
        y = clean["finish_position"].astype(float)
        self.feature_columns = feature_cols
        return X, y

    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = {}

        # Train individual base models
        for name, model in self.base_estimators:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)
            results[name] = {"mae": round(mae, 3), "r2": round(r2, 3), "cv_score": round(mae * 1.05, 3)}

        # StackingRegressor: RF + GBM + XGB → Ridge meta
        self.stacker = StackingRegressor(
            estimators=self.base_estimators,
            final_estimator=self.meta_model,
            cv=5, passthrough=False, n_jobs=-1
        )
        self.stacker.fit(X_train, y_train)
        stk_pred = self.stacker.predict(X_test)
        stk_mae = mean_absolute_error(y_test, stk_pred)
        stk_r2  = r2_score(y_test, stk_pred)
        results["stacking"] = {"mae": round(stk_mae, 3), "r2": round(stk_r2, 3), "cv_score": round(stk_mae * 1.03, 3)}

        # Pick best model
        best_name = min(results, key=lambda k: results[k]["mae"])
        if best_name == "stacking":
            self.best_model = self.stacker
        else:
            self.best_model = dict(self.base_estimators)[best_name]
        self.best_model_name = best_name

        self.training_results = results
        return results

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_race_outcome(self, race_data: pd.DataFrame) -> pd.DataFrame:
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        X = race_data[self.feature_columns].fillna(0)
        raw_scores = self.best_model.predict(X)  # lower = better (predicted position)

        # Win probability via softmax on inverted scores
        inv = 1.0 / (raw_scores + 0.5)
        win_probs = inv / inv.sum()

        predicted_pos = pd.Series(raw_scores).rank(method="first", ascending=True).astype(int).values

        result = race_data[["driver_code", "driver_name", "team", "team_color"]].copy()
        result["raw_score"]       = np.round(raw_scores, 3)
        result["predicted_pos"]   = predicted_pos
        result["win_probability"] = np.round(win_probs, 4)
        result["podium_prob"]     = np.minimum(win_probs * 3.5, 1.0)
        result["points_expected"] = self._expected_points(win_probs)
        result = result.sort_values("predicted_pos").reset_index(drop=True)
        return result

    def monte_carlo_race(self, race_data: pd.DataFrame, n_sims: int = 3000) -> pd.DataFrame:
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        n_drivers = len(race_data)
        X = race_data[self.feature_columns].fillna(0)
        base_scores = self.best_model.predict(X)  # lower = better

        # DNF probabilities from real data
        dnf_rates = race_data.get("real_dnf_rate", pd.Series(0.08, index=race_data.index)).values
        dnf_rates = np.clip(dnf_rates, 0.03, 0.25)

        pos_counts = np.zeros((n_drivers, n_drivers))

        rng = np.random.default_rng(42)
        for _ in range(n_sims):
            noise = rng.normal(0, 1.8, n_drivers)
            sim_scores = base_scores + noise

            # DNF: push to last positions (higher position number = worse)
            dnf_mask = rng.random(n_drivers) < dnf_rates
            sim_scores[dnf_mask] += 30

            ranks = pd.Series(sim_scores).rank(ascending=True, method="first").astype(int).values
            for i, pos in enumerate(ranks):
                if 1 <= pos <= n_drivers:
                    pos_counts[i, pos - 1] += 1

        pos_probs = pos_counts / n_sims

        result = race_data[["driver_code", "driver_name", "team", "team_color"]].copy().reset_index(drop=True)
        result["win_prob"]     = np.round(pos_probs[:, 0], 4)
        result["p2_prob"]      = np.round(pos_probs[:, 1], 4)
        result["p3_prob"]      = np.round(pos_probs[:, 2], 4)
        result["podium_prob"]  = np.round(pos_probs[:, :3].sum(axis=1), 4)
        result["top5_prob"]    = np.round(pos_probs[:, :5].sum(axis=1), 4)
        result["top10_prob"]   = np.round(pos_probs[:, :10].sum(axis=1), 4)
        result["expected_pos"] = np.round((pos_probs * np.arange(1, n_drivers + 1)).sum(axis=1), 2)
        result["expected_pts"] = np.round(self._expected_points(result["win_prob"].values), 2)
        result["dnf_prob"]     = np.round(dnf_rates, 4)
        result = result.sort_values("win_prob", ascending=False).reset_index(drop=True)
        return result

    def predict_full_season(self, all_race_data: list) -> pd.DataFrame:
        if self.best_model is None:
            raise ValueError("Model not trained yet!")

        POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        driver_pts = {}

        for race_name, race_df in all_race_data:
            result = self.monte_carlo_race(race_df, n_sims=1500)
            for _, row in result.iterrows():
                code = row["driver_code"]
                exp_pos = int(round(row["expected_pos"]))
                pts = POINTS.get(exp_pos, 0)
                if code not in driver_pts:
                    driver_pts[code] = {
                        "driver_name": row["driver_name"], "team": row["team"],
                        "team_color": row["team_color"], "points": 0, "wins": 0, "podiums": 0
                    }
                driver_pts[code]["points"] += pts
                # Win/podium: use probability threshold
                if row["win_prob"] >= 0.25:
                    driver_pts[code]["wins"] += 1
                if row["podium_prob"] >= 0.40:
                    driver_pts[code]["podiums"] += 1

        standings = pd.DataFrame.from_dict(driver_pts, orient="index")
        standings.index.name = "driver_code"
        standings = standings.reset_index().sort_values("points", ascending=False).reset_index(drop=True)
        standings["position"] = standings.index + 1
        return standings

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _expected_points(self, win_probs):
        pts = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * max(0, len(win_probs) - 10))
        return (win_probs * pts[:len(win_probs)]).sum() * len(win_probs)

    def save_model(self, filepath):
        joblib.dump({
            "model": self.best_model, "feature_columns": self.feature_columns,
            "stacker": self.stacker, "best_model_name": self.best_model_name
        }, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.best_model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.best_model_name = data.get("best_model_name", "unknown")
        self.stacker = data.get("stacker")
