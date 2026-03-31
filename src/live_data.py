import os
import fastf1
import pandas as pd
from datetime import datetime

class LiveDataIngestor:
    def __init__(self, year=2026, cache_dir='data/fastf1_cache', live_out='data/live'):
        self.year = year
        self.cache_dir = cache_dir
        self.live_out = live_out
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.live_out, exist_ok=True)
        fastf1.Cache.enable_cache(self.cache_dir)

    def fetch_latest_data(self):
        """Fetches the latest race results up to the current date and saves to CSV."""
        try:
            schedule = fastf1.get_event_schedule(self.year)
        except Exception as e:
            print(f"No schedule available for {self.year}: {e}")
            return pd.DataFrame()
        
        # Only interested in events that have happened
        now = pd.Timestamp.now()
        
        all_results = []
        
        for idx, event in schedule.iterrows():
            if event['EventDate'] < now and event['EventFormat'] != 'testing':
                try:
                    session = fastf1.get_session(self.year, event['RoundNumber'], 'R')
                    session.load(telemetry=False, weather=False, messages=False)
                    res = session.results
                    if res.empty:
                        continue
                    
                    # Build numeric positions
                    df = pd.DataFrame({
                        'year': self.year,
                        'round': event['RoundNumber'],
                        'circuit_name': event['Location'],
                        'driver_code': res['Abbreviation'],
                        'finish_position': pd.to_numeric(res['Position'], errors='coerce').fillna(20),
                        'grid_position': pd.to_numeric(res['GridPosition'], errors='coerce').fillna(20),
                        'status': res['Status'],
                        'points': pd.to_numeric(res['Points'], errors='coerce').fillna(0)
                    })
                    
                    # DNF is if finish position is bad or status is not Finished/+x Laps
                    df['dnf'] = df['status'].apply(lambda x: 0 if 'Finished' in str(x) or 'Lap' in str(x) else 1)
                    
                    all_results.append(df)
                    print(f"Loaded round {event['RoundNumber']}")
                except Exception as e:
                    print(f"Could not load {event['RoundNumber']}: {e}")
        
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            out_path = os.path.join(self.live_out, f'{self.year}_results.csv')
            combined.to_csv(out_path, index=False)
            print(f"Saved live data: {out_path}")
            return combined
            
        return pd.DataFrame()
        
    def load_cached_data(self):
        """Loads cached real 2026 data without fetching."""
        out_path = os.path.join(self.live_out, f'{self.year}_results.csv')
        if os.path.exists(out_path):
            return pd.read_csv(out_path)
        return pd.DataFrame()

if __name__ == "__main__":
    ingestor = LiveDataIngestor()
    df = ingestor.fetch_latest_data()
    print("Live Data retrieved:", len(df))
