import fastf1
fastf1.Cache.enable_cache('data') # enable FastF1 cache in data/ folder

try:
    session = fastf1.get_session(2026, 1, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    print("2026 Round 1 Race Results loaded!")
    print(session.results.head())
except Exception as e:
    print(f"Error 2026: {e}")
