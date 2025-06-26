import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

class F1PredictionModels:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'linear_regression': LinearRegression()
        }
        self.best_model = None
        self.feature_columns = None
    
    def prepare_training_data(self, data):
        """Prepare data for training"""
        # Select features for training
        feature_cols = [
            'grid_position', 'avg_lap_time', 'fastest_lap', 'consistency',
            'recent_avg_position', 'recent_avg_points', 'season_avg_position',
            'season_points', 'driver_encoded', 'team_encoded', 'circuit_encoded'
        ]
        
        # Remove rows with missing target values
        clean_data = data.dropna(subset=['finish_position'])
        
        X = clean_data[feature_cols].fillna(clean_data[feature_cols].mean())
        y = clean_data['finish_position']
        
        self.feature_columns = feature_cols
        return X, y
    
    def train_models(self, X, y):
        """Train and evaluate multiple models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            results[name] = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'cv_score': -cv_scores.mean()
            }
            
            print(f"{name} - MAE: {mae:.3f}, R²: {r2:.3f}, CV Score: {-cv_scores.mean():.3f}")
        
        # Select best model based on MAE
        best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        return results
    
    def predict_race_outcome(self, current_data):
        """Predict race outcomes for current data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        X_current = current_data[self.feature_columns].fillna(current_data[self.feature_columns].mean())
        predictions = self.best_model.predict(X_current)
        
        return predictions
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.feature_columns = model_data['feature_columns']
