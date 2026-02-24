import sys
sys.path.insert(0, 'c:/Users/Satyam/Desktop/f1-prediction-app')
from src.data_collection import F1DataCollector, CALENDAR_2026, DRIVERS_2026
from src.feature_engineering import FeatureEngineer
from src.ml_models import F1PredictionModels
from src.prediction_engine import F1PredictionEngine

print("=== MODULE IMPORT TEST ===")
dc = F1DataCollector()
print(f"Drivers: {len(DRIVERS_2026)}")
print(f"Calendar races: {len(CALENDAR_2026)}")

drivers_df = dc.get_drivers_df()
print(f"Drivers DF shape: {drivers_df.shape}")

cal_df = dc.get_calendar_df()
print(f"Calendar DF shape: {cal_df.shape}")

print()
print("=== TRAINING ENGINE TEST ===")
engine = F1PredictionEngine()
results = engine.quick_demo_train()
print(f"Training results keys: {list(results.keys())}")
best = min(results, key=lambda x: results[x]["mae"])
print(f"Best model: {best} (MAE={results[best]['mae']}, R2={results[best]['r2']})")

print()
print("=== SINGLE RACE PREDICTION TEST ===")
det, mc = engine.predict_race("Albert Park", n_sims=500)
print(f"Prediction rows: {len(mc)}")
print("Top 5 predictions:")
for _, row in mc.head(5).iterrows():
    print(f"  {row['driver_code']:4s} ({row['team']}): Win={row['win_prob']:.1%}, ExpPos={row['expected_pos']:.1f}")

print()
print("=== LAP SIMULATION TEST ===")
sim_input = mc.copy()
sim_input["predicted_pos"] = mc["expected_pos"].round().astype(int).values
lap_df = engine.generate_lap_simulation(sim_input, n_laps=20)
print(f"Lap trace rows: {len(lap_df)}")

print()
print("=== SEASON PREDICTION TEST ===")
standings = engine.predict_full_season()
print(f"Standings rows: {len(standings)}")
print("Top 5 championship:")
for _, row in standings.head(5).iterrows():
    print(f"  P{int(row['position'])}: {row['driver_code']:4s} ({row['team']}) - {int(row['points'])} pts, {int(row['wins'])} wins")

print()
print("=== ALL TESTS PASSED ===")
