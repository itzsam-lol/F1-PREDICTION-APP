import fastf1
import datetime

try:
    print(fastf1.get_event_schedule(2026))
except Exception as e:
    print(f"Error 2026: {e}")

try:
    print(fastf1.get_event_schedule(2024))
except Exception as e:
    print(f"Error 2024: {e}")
