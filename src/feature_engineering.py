import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_driver_features(self, historical_data):
        """Create driver-specific performance features"""
        driver_stats = []
        
        for race_data in historical_data:
            results = race_data['results']
            laps = race_data['laps']
            
            for _, driver in results.iterrows():
                # Calculate performance metrics
                driver_laps = laps[laps['Driver'] == driver['Abbreviation']]
                
                features = {
                    'driver': driver['Abbreviation'],
                    'team': driver['TeamName'],
                    'grid_position': driver['GridPosition'],
                    'finish_position': driver['Position'],
                    'points': driver['Points'],
                    'avg_lap_time': driver_laps['LapTime'].mean().total_seconds() if not driver_laps.empty else np.nan,
                    'fastest_lap': driver_laps['LapTime'].min().total_seconds() if not driver_laps.empty else np.nan,
                    'consistency': driver_laps['LapTime'].std().total_seconds() if len(driver_laps) > 1 else np.nan,
                    'circuit': race_data['circuit'],
                    'year': race_data['year']
                }
                driver_stats.append(features)
        
        return pd.DataFrame(driver_stats)
    
    def create_team_features(self, driver_features):
        """Create team-specific performance features"""
        team_stats = driver_features.groupby(['team', 'circuit', 'year']).agg({
            'points': 'sum',
            'finish_position': 'mean',
            'avg_lap_time': 'mean',
            'fastest_lap': 'min',
            'consistency': 'mean'
        }).reset_index()
        
        return team_stats
    
    def create_circuit_features(self, driver_features):
        """Create circuit-specific features"""
        circuit_stats = driver_features.groupby(['driver', 'circuit']).agg({
            'finish_position': 'mean',
            'points': 'mean',
            'avg_lap_time': 'mean'
        }).reset_index()
        
        circuit_stats.columns = ['driver', 'circuit', 'avg_finish_at_circuit', 
                                'avg_points_at_circuit', 'avg_laptime_at_circuit']
        
        return circuit_stats
    
    def engineer_features(self, data):
        """Main feature engineering pipeline"""
        # Create rolling averages for recent performance
        data = data.sort_values(['driver', 'year'])
        
        # Recent form (last 5 races)
        data['recent_avg_position'] = data.groupby('driver')['finish_position'].rolling(5, min_periods=1).mean().values
        data['recent_avg_points'] = data.groupby('driver')['points'].rolling(5, min_periods=1).mean().values
        
        # Season performance
        data['season_avg_position'] = data.groupby(['driver', 'year'])['finish_position'].expanding().mean().values
        data['season_points'] = data.groupby(['driver', 'year'])['points'].expanding().sum().values
        
        # Encode categorical variables
        categorical_cols = ['driver', 'team', 'circuit']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col])
            else:
                data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col])
        
        return data
