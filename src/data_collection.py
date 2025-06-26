import requests
import pandas as pd
import fastf1
from datetime import datetime, timedelta

class F1DataCollector:
    def __init__(self):
        self.base_url = "https://api.openf1.org/v1"
        
    def get_session_data(self, session_key):
        """Fetch session data from OpenF1 API"""
        response = requests.get(f"{self.base_url}/sessions?session_key={session_key}")
        return response.json()
    
    def get_lap_data(self, session_key, driver_number=None):
        """Fetch lap times data"""
        url = f"{self.base_url}/laps?session_key={session_key}"
        if driver_number:
            url += f"&driver_number={driver_number}"
        response = requests.get(url)
        return pd.DataFrame(response.json())
    
    def get_driver_data(self, session_key):
        """Fetch driver information"""
        response = requests.get(f"{self.base_url}/drivers?session_key={session_key}")
        return pd.DataFrame(response.json())
    
    def get_position_data(self, session_key):
        """Fetch position data throughout the race"""
        response = requests.get(f"{self.base_url}/position?session_key={session_key}")
        return pd.DataFrame(response.json())
    
    def collect_historical_data(self, year_range=(2022, 2024)):
        """Collect historical race data for training"""
        historical_data = []
        
        for year in range(year_range[0], year_range[1] + 1):
            try:
                schedule = fastf1.get_event_schedule(year)
                for _, event in schedule.iterrows():
                    if event['EventFormat'] == 'conventional':
                        session = fastf1.get_session(year, event['RoundNumber'], 'R')
                        session.load()
                        
                        # Extract relevant features
                        race_data = {
                            'year': year,
                            'round': event['RoundNumber'],
                            'circuit': event['Location'],
                            'results': session.results,
                            'laps': session.laps,
                            'weather': session.weather_data
                        }
                        historical_data.append(race_data)
            except Exception as e:
                print(f"Error collecting data for {year}: {e}")
                continue
                
        return historical_data
