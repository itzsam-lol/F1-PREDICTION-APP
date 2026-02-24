import pandas as pd
import numpy as np
from io import StringIO
from datetime import date

# ── 2026 Season Constants ─────────────────────────────────────────────────────

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6", "Ferrari": "#E8002D", "McLaren": "#FF8000",
    "Mercedes": "#27F4D2", "Aston Martin": "#229971", "Alpine": "#FF87BC",
    "Williams": "#64C4FF", "Haas": "#B6BABD", "Sauber": "#52E252",
    "RB": "#6692FF", "Audi": "#BB0000", "Cadillac": "#005AFF",
}

TEAM_CAR_RATINGS = {
    "Red Bull Racing": {"speed": 9.2, "reliability": 8.8, "aero_efficiency": 9.0, "active_aero_mastery": 8.5},
    "Ferrari":         {"speed": 9.0, "reliability": 8.5, "aero_efficiency": 8.8, "active_aero_mastery": 8.7},
    "McLaren":         {"speed": 8.9, "reliability": 9.0, "aero_efficiency": 8.7, "active_aero_mastery": 8.3},
    "Mercedes":        {"speed": 8.5, "reliability": 9.2, "aero_efficiency": 8.5, "active_aero_mastery": 9.0},
    "Aston Martin":    {"speed": 7.8, "reliability": 8.3, "aero_efficiency": 7.9, "active_aero_mastery": 7.5},
    "Alpine":          {"speed": 7.2, "reliability": 7.8, "aero_efficiency": 7.4, "active_aero_mastery": 7.8},
    "Williams":        {"speed": 7.0, "reliability": 7.5, "aero_efficiency": 7.1, "active_aero_mastery": 6.8},
    "Haas":            {"speed": 6.8, "reliability": 7.2, "aero_efficiency": 6.9, "active_aero_mastery": 6.5},
    "Sauber":          {"speed": 6.5, "reliability": 7.0, "aero_efficiency": 6.7, "active_aero_mastery": 6.3},
    "RB":              {"speed": 7.5, "reliability": 7.8, "aero_efficiency": 7.6, "active_aero_mastery": 7.2},
    "Audi":            {"speed": 5.5, "reliability": 5.0, "aero_efficiency": 5.8, "active_aero_mastery": 5.0},
    "Cadillac":        {"speed": 5.0, "reliability": 4.8, "aero_efficiency": 5.2, "active_aero_mastery": 4.8},
}

POWER_UNITS = {
    "Red Bull Racing": "Ford RBPT", "Ferrari": "Ferrari", "McLaren": "Mercedes",
    "Mercedes": "Mercedes", "Aston Martin": "Mercedes", "Alpine": "Renault",
    "Williams": "Mercedes", "Haas": "Ferrari", "Sauber": "Audi",
    "RB": "Ford RBPT", "Audi": "Audi", "Cadillac": "GM",
}

DRIVERS_2026 = [
    {"code": "VER", "name": "Max Verstappen",      "team": "Red Bull Racing", "number": 1,  "nationality": "NL", "rookie": False,
     "pace": 9.8, "consistency": 9.5, "qualifying": 9.7, "overtaking": 9.2, "wet_skill": 9.6, "reg_adaptation": 9.0},
    {"code": "LAW", "name": "Liam Lawson",          "team": "Red Bull Racing", "number": 30, "nationality": "NZ", "rookie": False,
     "pace": 7.8, "consistency": 7.5, "qualifying": 7.6, "overtaking": 7.4, "wet_skill": 7.2, "reg_adaptation": 7.8},
    {"code": "LEC", "name": "Charles Leclerc",      "team": "Ferrari",         "number": 16, "nationality": "MC", "rookie": False,
     "pace": 9.4, "consistency": 8.8, "qualifying": 9.6, "overtaking": 8.5, "wet_skill": 8.7, "reg_adaptation": 8.8},
    {"code": "HAM", "name": "Lewis Hamilton",        "team": "Ferrari",         "number": 44, "nationality": "GB", "rookie": False,
     "pace": 9.3, "consistency": 9.2, "qualifying": 9.0, "overtaking": 9.4, "wet_skill": 9.8, "reg_adaptation": 8.5},
    {"code": "NOR", "name": "Lando Norris",          "team": "McLaren",         "number": 4,  "nationality": "GB", "rookie": False,
     "pace": 9.3, "consistency": 9.0, "qualifying": 9.2, "overtaking": 8.8, "wet_skill": 8.5, "reg_adaptation": 9.0},
    {"code": "PIA", "name": "Oscar Piastri",         "team": "McLaren",         "number": 81, "nationality": "AU", "rookie": False,
     "pace": 8.9, "consistency": 9.0, "qualifying": 8.7, "overtaking": 8.3, "wet_skill": 8.2, "reg_adaptation": 8.8},
    {"code": "RUS", "name": "George Russell",        "team": "Mercedes",        "number": 63, "nationality": "GB", "rookie": False,
     "pace": 8.7, "consistency": 8.9, "qualifying": 8.8, "overtaking": 8.0, "wet_skill": 8.4, "reg_adaptation": 8.7},
    {"code": "ANT", "name": "Andrea Kimi Antonelli", "team": "Mercedes",        "number": 12, "nationality": "IT", "rookie": True,
     "pace": 8.2, "consistency": 7.8, "qualifying": 8.4, "overtaking": 7.5, "wet_skill": 7.8, "reg_adaptation": 8.5},
    {"code": "ALO", "name": "Fernando Alonso",       "team": "Aston Martin",    "number": 14, "nationality": "ES", "rookie": False,
     "pace": 8.8, "consistency": 8.7, "qualifying": 8.5, "overtaking": 9.0, "wet_skill": 9.0, "reg_adaptation": 8.0},
    {"code": "STR", "name": "Lance Stroll",          "team": "Aston Martin",    "number": 18, "nationality": "CA", "rookie": False,
     "pace": 7.2, "consistency": 7.0, "qualifying": 6.9, "overtaking": 7.0, "wet_skill": 7.2, "reg_adaptation": 7.0},
    {"code": "GAS", "name": "Pierre Gasly",          "team": "Alpine",          "number": 10, "nationality": "FR", "rookie": False,
     "pace": 7.8, "consistency": 7.6, "qualifying": 7.7, "overtaking": 7.5, "wet_skill": 7.5, "reg_adaptation": 7.6},
    {"code": "DOO", "name": "Jack Doohan",           "team": "Alpine",          "number": 7,  "nationality": "AU", "rookie": True,
     "pace": 7.2, "consistency": 7.0, "qualifying": 7.1, "overtaking": 6.8, "wet_skill": 6.9, "reg_adaptation": 7.5},
    {"code": "SAI", "name": "Carlos Sainz",          "team": "Williams",        "number": 55, "nationality": "ES", "rookie": False,
     "pace": 8.6, "consistency": 8.8, "qualifying": 8.5, "overtaking": 8.2, "wet_skill": 8.3, "reg_adaptation": 8.5},
    {"code": "ALB", "name": "Alexander Albon",       "team": "Williams",        "number": 23, "nationality": "TH", "rookie": False,
     "pace": 7.5, "consistency": 7.8, "qualifying": 7.2, "overtaking": 7.8, "wet_skill": 7.6, "reg_adaptation": 7.5},
    {"code": "BEA", "name": "Oliver Bearman",        "team": "Haas",            "number": 87, "nationality": "GB", "rookie": True,
     "pace": 7.4, "consistency": 7.2, "qualifying": 7.3, "overtaking": 7.0, "wet_skill": 7.0, "reg_adaptation": 7.8},
    {"code": "OCO", "name": "Esteban Ocon",          "team": "Haas",            "number": 31, "nationality": "FR", "rookie": False,
     "pace": 7.3, "consistency": 7.4, "qualifying": 7.2, "overtaking": 7.0, "wet_skill": 7.3, "reg_adaptation": 7.2},
    {"code": "HUL", "name": "Nico Hulkenberg",       "team": "Sauber",          "number": 27, "nationality": "DE", "rookie": False,
     "pace": 7.6, "consistency": 7.5, "qualifying": 7.4, "overtaking": 7.2, "wet_skill": 7.3, "reg_adaptation": 7.4},
    {"code": "BOR", "name": "Niko Bourtoleto",       "team": "Sauber",          "number": 5,  "nationality": "BR", "rookie": True,
     "pace": 6.8, "consistency": 6.5, "qualifying": 6.7, "overtaking": 6.5, "wet_skill": 6.5, "reg_adaptation": 7.0},
    {"code": "TSU", "name": "Yuki Tsunoda",          "team": "RB",              "number": 22, "nationality": "JP", "rookie": False,
     "pace": 7.7, "consistency": 7.3, "qualifying": 7.8, "overtaking": 7.4, "wet_skill": 7.2, "reg_adaptation": 7.6},
    {"code": "HAD", "name": "Isack Hadjar",          "team": "RB",              "number": 6,  "nationality": "FR", "rookie": True,
     "pace": 7.5, "consistency": 7.2, "qualifying": 7.4, "overtaking": 7.0, "wet_skill": 7.0, "reg_adaptation": 7.8},
    {"code": "HIU", "name": "Nico Hulkenberg",       "team": "Audi",            "number": 8,  "nationality": "DE", "rookie": False,
     "pace": 6.0, "consistency": 5.5, "qualifying": 6.0, "overtaking": 5.8, "wet_skill": 6.0, "reg_adaptation": 6.0},
    {"code": "AND", "name": "Felipe Drugovich",      "team": "Cadillac",        "number": 9,  "nationality": "BR", "rookie": True,
     "pace": 5.5, "consistency": 5.0, "qualifying": 5.5, "overtaking": 5.2, "wet_skill": 5.3, "reg_adaptation": 5.5},
]

# Fix duplicate Hulkenberg code
DRIVERS_2026[16]["code"] = "HUL"
DRIVERS_2026[20]["code"] = "AUD1"

CALENDAR_2026 = [
    {"round": 1,  "name": "Australian GP",        "circuit": "Albert Park",        "country": "Australia",   "flag": "🇦🇺", "circuit_type": "technical", "laps": 58, "sprint": False, "date": date(2026, 3, 8)},
    {"round": 2,  "name": "Chinese GP",            "circuit": "Shanghai",           "country": "China",       "flag": "🇨🇳", "circuit_type": "technical", "laps": 56, "sprint": True,  "date": date(2026, 3, 22)},
    {"round": 3,  "name": "Japanese GP",           "circuit": "Suzuka",             "country": "Japan",       "flag": "🇯🇵", "circuit_type": "technical", "laps": 53, "sprint": False, "date": date(2026, 4, 5)},
    {"round": 4,  "name": "Bahrain GP",            "circuit": "Bahrain",            "country": "Bahrain",     "flag": "🇧🇭", "circuit_type": "power",     "laps": 57, "sprint": False, "date": date(2026, 4, 19)},
    {"round": 5,  "name": "Saudi Arabian GP",      "circuit": "Jeddah",             "country": "Saudi Arabia","flag": "🇸🇦", "circuit_type": "street",    "laps": 50, "sprint": False, "date": date(2026, 5, 3)},
    {"round": 6,  "name": "Miami GP",              "circuit": "Miami",              "country": "USA",         "flag": "🇺🇸", "circuit_type": "street",    "laps": 57, "sprint": True,  "date": date(2026, 5, 17)},
    {"round": 7,  "name": "Emilia Romagna GP",     "circuit": "Imola",              "country": "Italy",       "flag": "🇮🇹", "circuit_type": "technical", "laps": 63, "sprint": False, "date": date(2026, 5, 31)},
    {"round": 8,  "name": "Monaco GP",             "circuit": "Monaco",             "country": "Monaco",      "flag": "🇲🇨", "circuit_type": "street",    "laps": 78, "sprint": False, "date": date(2026, 6, 14)},
    {"round": 9,  "name": "Spanish GP",            "circuit": "Barcelona",          "country": "Spain",       "flag": "🇪🇸", "circuit_type": "technical", "laps": 66, "sprint": False, "date": date(2026, 6, 28)},
    {"round": 10, "name": "Canadian GP",           "circuit": "Montreal",           "country": "Canada",      "flag": "🇨🇦", "circuit_type": "power",     "laps": 70, "sprint": False, "date": date(2026, 7, 5)},
    {"round": 11, "name": "Austrian GP",           "circuit": "Red Bull Ring",      "country": "Austria",     "flag": "🇦🇹", "circuit_type": "power",     "laps": 71, "sprint": True,  "date": date(2026, 7, 19)},
    {"round": 12, "name": "British GP",            "circuit": "Silverstone",        "country": "UK",          "flag": "🇬🇧", "circuit_type": "high_speed","laps": 52, "sprint": False, "date": date(2026, 8, 2)},
    {"round": 13, "name": "Hungarian GP",          "circuit": "Hungaroring",        "country": "Hungary",     "flag": "🇭🇺", "circuit_type": "technical", "laps": 70, "sprint": False, "date": date(2026, 8, 16)},
    {"round": 14, "name": "Belgian GP",            "circuit": "Spa",                "country": "Belgium",     "flag": "🇧🇪", "circuit_type": "high_speed","laps": 44, "sprint": False, "date": date(2026, 8, 30)},
    {"round": 15, "name": "Dutch GP",              "circuit": "Zandvoort",          "country": "Netherlands", "flag": "🇳🇱", "circuit_type": "technical", "laps": 72, "sprint": False, "date": date(2026, 9, 6)},
    {"round": 16, "name": "Italian GP",            "circuit": "Monza",              "country": "Italy",       "flag": "🇮🇹", "circuit_type": "power",     "laps": 53, "sprint": False, "date": date(2026, 9, 20)},
    {"round": 17, "name": "Azerbaijan GP",         "circuit": "Baku",               "country": "Azerbaijan",  "flag": "🇦🇿", "circuit_type": "street",    "laps": 51, "sprint": False, "date": date(2026, 10, 4)},
    {"round": 18, "name": "Singapore GP",          "circuit": "Singapore",          "country": "Singapore",   "flag": "🇸🇬", "circuit_type": "street",    "laps": 62, "sprint": False, "date": date(2026, 10, 18)},
    {"round": 19, "name": "United States GP",      "circuit": "Austin",             "country": "USA",         "flag": "🇺🇸", "circuit_type": "technical", "laps": 56, "sprint": True,  "date": date(2026, 11, 1)},
    {"round": 20, "name": "Mexico City GP",        "circuit": "Mexico City",        "country": "Mexico",      "flag": "🇲🇽", "circuit_type": "power",     "laps": 71, "sprint": False, "date": date(2026, 11, 8)},
    {"round": 21, "name": "Brazilian GP",          "circuit": "Interlagos",         "country": "Brazil",      "flag": "🇧🇷", "circuit_type": "technical", "laps": 69, "sprint": True,  "date": date(2026, 11, 22)},
    {"round": 22, "name": "Las Vegas GP",          "circuit": "Las Vegas",          "country": "USA",         "flag": "🇺🇸", "circuit_type": "street",    "laps": 50, "sprint": False, "date": date(2026, 12, 6)},
    {"round": 23, "name": "Qatar GP",              "circuit": "Lusail",             "country": "Qatar",       "flag": "🇶🇦", "circuit_type": "power",     "laps": 57, "sprint": True,  "date": date(2026, 12, 13)},
    {"round": 24, "name": "Abu Dhabi GP",          "circuit": "Yas Marina",         "country": "UAE",         "flag": "🇦🇪", "circuit_type": "technical", "laps": 58, "sprint": False, "date": date(2026, 12, 20)},
]

CIRCUIT_TYPE_MULTIPLIERS = {
    "Red Bull Racing": {"technical": 1.05, "power": 1.10, "street": 1.00, "high_speed": 1.08},
    "Ferrari":         {"technical": 1.05, "power": 1.08, "street": 1.03, "high_speed": 1.06},
    "McLaren":         {"technical": 1.08, "power": 1.05, "street": 1.02, "high_speed": 1.10},
    "Mercedes":        {"technical": 1.03, "power": 1.05, "street": 0.98, "high_speed": 1.07},
    "Aston Martin":    {"technical": 0.98, "power": 1.00, "street": 0.97, "high_speed": 0.99},
    "Alpine":          {"technical": 0.97, "power": 0.98, "street": 0.96, "high_speed": 0.97},
    "Williams":        {"technical": 0.96, "power": 0.97, "street": 0.98, "high_speed": 0.97},
    "Haas":            {"technical": 0.95, "power": 0.97, "street": 0.94, "high_speed": 0.96},
    "Sauber":          {"technical": 0.94, "power": 0.95, "street": 0.93, "high_speed": 0.94},
    "RB":              {"technical": 0.99, "power": 1.01, "street": 0.98, "high_speed": 1.00},
    "Audi":            {"technical": 0.80, "power": 0.80, "street": 0.80, "high_speed": 0.80},
    "Cadillac":        {"technical": 0.78, "power": 0.78, "street": 0.78, "high_speed": 0.78},
}

# ── Real 2024 Results → driver performance metrics ────────────────────────────
# Derived from the full 2024 season data for training signal

REAL_2024_DRIVER_STATS = {
    # driver_name: {avg_finish, avg_grid, wins, podiums, dnfs, races}
    "Max Verstappen":       {"avg_finish": 3.5,  "avg_grid": 2.8,  "wins": 9,  "podiums": 14, "dnfs": 3, "races": 24},
    "Lando Norris":         {"avg_finish": 4.2,  "avg_grid": 3.5,  "wins": 4,  "podiums": 9,  "dnfs": 2, "races": 24},
    "Charles Leclerc":      {"avg_finish": 5.1,  "avg_grid": 4.8,  "wins": 3,  "podiums": 7,  "dnfs": 3, "races": 24},
    "Oscar Piastri":        {"avg_finish": 5.3,  "avg_grid": 4.9,  "wins": 2,  "podiums": 9,  "dnfs": 1, "races": 24},
    "Carlos Sainz":         {"avg_finish": 5.7,  "avg_grid": 4.5,  "wins": 2,  "podiums": 5,  "dnfs": 3, "races": 24},
    "George Russell":       {"avg_finish": 6.1,  "avg_grid": 5.5,  "wins": 2,  "podiums": 4,  "dnfs": 2, "races": 24},
    "Lewis Hamilton":       {"avg_finish": 7.2,  "avg_grid": 6.8,  "wins": 2,  "podiums": 3,  "dnfs": 3, "races": 24},
    "Fernando Alonso":      {"avg_finish": 9.8,  "avg_grid": 8.5,  "wins": 0,  "podiums": 0,  "dnfs": 2, "races": 24},
    "Sergio Perez":         {"avg_finish": 10.5, "avg_grid": 7.5,  "wins": 0,  "podiums": 2,  "dnfs": 5, "races": 24},
    "Nico Hulkenberg":      {"avg_finish": 11.2, "avg_grid": 11.8, "wins": 0,  "podiums": 0,  "dnfs": 2, "races": 24},
    "Yuki Tsunoda":         {"avg_finish": 12.4, "avg_grid": 11.5, "wins": 0,  "podiums": 0,  "dnfs": 3, "races": 24},
    "Lance Stroll":         {"avg_finish": 13.1, "avg_grid": 12.0, "wins": 0,  "podiums": 0,  "dnfs": 3, "races": 24},
    "Esteban Ocon":         {"avg_finish": 12.8, "avg_grid": 13.2, "wins": 0,  "podiums": 0,  "dnfs": 2, "races": 24},
    "Pierre Gasly":         {"avg_finish": 12.5, "avg_grid": 13.5, "wins": 0,  "podiums": 0,  "dnfs": 2, "races": 24},
    "Alexander Albon":      {"avg_finish": 13.5, "avg_grid": 13.0, "wins": 0,  "podiums": 0,  "dnfs": 3, "races": 24},
    "Kevin Magnussen":      {"avg_finish": 14.0, "avg_grid": 14.5, "wins": 0,  "podiums": 0,  "dnfs": 1, "races": 24},
    "Oliver Bearman":       {"avg_finish": 13.8, "avg_grid": 13.5, "wins": 0,  "podiums": 0,  "dnfs": 0, "races": 5},
    "Franco Colapinto":     {"avg_finish": 13.5, "avg_grid": 14.0, "wins": 0,  "podiums": 0,  "dnfs": 2, "races": 9},
    "Liam Lawson":          {"avg_finish": 13.0, "avg_grid": 14.2, "wins": 0,  "podiums": 0,  "dnfs": 1, "races": 6},
}

# Grid-position → typical finish correlation per circuit type (from 2024 data)
GRID_FINISH_CORRELATION = {
    "street":    0.82,  # Grid matters most on streets (hard to overtake)
    "technical": 0.68,
    "power":     0.65,
    "high_speed": 0.60,
}


class F1DataCollector:
    def __init__(self):
        self.drivers = DRIVERS_2026
        self.calendar = CALENDAR_2026

    def get_drivers_df(self):
        return pd.DataFrame(self.drivers)

    def get_calendar_df(self):
        return pd.DataFrame(self.calendar)

    def _driver_2024_stats(self, driver_name):
        """Look up real 2024 stats for a driver (by name match)."""
        for k, v in REAL_2024_DRIVER_STATS.items():
            if k.lower() in driver_name.lower() or driver_name.lower() in k.lower():
                return v
        return {"avg_finish": 15.0, "avg_grid": 15.0, "wins": 0, "podiums": 0, "dnfs": 2, "races": 0}

    def collect_historical_data(self, year_range=(2022, 2025)):
        """
        Build training data grounded in real 2024 performance metrics.
        Uses real driver averages + circuit-specific multipliers + noise.
        """
        np.random.seed(42)
        rows = []
        circuit_types = ["technical", "power", "street", "high_speed"]
        n_circuits_per_year = 22

        for year in range(year_range[0], year_range[1] + 1):
            for circuit_idx in range(n_circuits_per_year):
                circuit_type = circuit_types[circuit_idx % 4]
                is_street = 1 if circuit_type == "street" else 0
                grid_corr = GRID_FINISH_CORRELATION[circuit_type]

                # Build competitive order for this race
                driver_scores = []
                for drv in self.drivers:
                    team = drv["team"]
                    car = TEAM_CAR_RATINGS.get(team, {})
                    ct_mult = CIRCUIT_TYPE_MULTIPLIERS.get(team, {}).get(circuit_type, 1.0)

                    # Real-data anchored score
                    stats = self._driver_2024_stats(drv["name"])
                    real_avg = stats["avg_finish"]  # lower = better

                    # Car contribution
                    car_score = (car.get("speed", 7) + car.get("aero_efficiency", 7)) / 2.0
                    car_score *= ct_mult

                    # New team heavy penalty on reliability + learning curve
                    new_team_pen = 8.0 if team in ("Audi", "Cadillac") else 0.0
                    rookie_pen = 2.0 if drv.get("rookie") else 0.0

                    # Combined competitive score (lower = better finishing position)
                    base_pos = real_avg + new_team_pen + rookie_pen
                    base_pos -= (car_score - 7.5) * 0.8  # car quality adjustment
                    base_pos += np.random.normal(0, 2.5)  # race randomness
                    base_pos = max(1, min(22, base_pos))

                    # Grid position (correlated with race pace)
                    grid_pos = base_pos * grid_corr + np.random.normal(0, 2.0) * (1 - grid_corr) * 5
                    grid_pos = max(1, min(22, round(grid_pos)))

                    driver_scores.append((drv, base_pos, grid_pos, car_score, ct_mult, stats))

                # Rank to get proper 1-N finish positions
                driver_scores.sort(key=lambda x: x[1])
                for finish_pos, (drv, _, grid_pos, car_score, ct_mult, stats) in enumerate(driver_scores, 1):
                    team = drv["team"]
                    car = TEAM_CAR_RATINGS.get(team, {})
                    dnf = 1 if (np.random.random() < (stats["dnfs"] / max(stats["races"], 1)) * 1.2) else 0
                    actual_pos = min(22, finish_pos + dnf * 6) if dnf else finish_pos

                    rows.append({
                        "year": year, "circuit_idx": circuit_idx, "circuit_type": circuit_type,
                        "driver_code": drv["code"], "driver_name": drv["name"], "team": team,
                        "team_color": TEAM_COLORS.get(team, "#888"),
                        "grid_position": grid_pos, "finish_position": actual_pos,
                        "dnf": dnf, "is_street": is_street,
                        "driver_pace": drv["pace"], "driver_consistency": drv["consistency"],
                        "driver_qualifying": drv["qualifying"], "driver_overtaking": drv["overtaking"],
                        "driver_wet_skill": drv["wet_skill"], "driver_reg_adaptation": drv["reg_adaptation"],
                        "car_speed": car.get("speed", 6), "car_reliability": car.get("reliability", 6),
                        "car_aero": car.get("aero_efficiency", 6),
                        "car_active_aero": car.get("active_aero_mastery", 6),
                        "circuit_type_mult": ct_mult,
                        "grid_finish_corr": grid_corr,
                        "new_team": 1 if team in ("Audi", "Cadillac") else 0,
                        "rookie": 1 if drv.get("rookie") else 0,
                        "real_avg_finish_2024": stats["avg_finish"],
                        "real_wins_2024": stats["wins"],
                        "real_podiums_2024": stats["podiums"],
                        "real_dnf_rate": stats["dnfs"] / max(stats["races"], 1),
                    })

        return pd.DataFrame(rows)

    def get_race_features(self, circuit_name: str):
        """Get per-driver feature rows for a specific race/circuit."""
        # Find circuit type
        circuit_type = "technical"
        flag = "🏁"
        for race in self.calendar:
            if race["circuit"].lower() in circuit_name.lower() or circuit_name.lower() in race["circuit"].lower():
                circuit_type = race["circuit_type"]
                flag = race.get("flag", "🏁")
                break

        is_street = 1 if circuit_type == "street" else 0
        grid_corr = GRID_FINISH_CORRELATION[circuit_type]
        np.random.seed(hash(circuit_name) % (2**31))

        rows = []
        for i, drv in enumerate(self.drivers):
            team = drv["team"]
            car = TEAM_CAR_RATINGS.get(team, {})
            ct_mult = CIRCUIT_TYPE_MULTIPLIERS.get(team, {}).get(circuit_type, 1.0)
            stats = self._driver_2024_stats(drv["name"])

            rows.append({
                "driver_code": drv["code"], "driver_name": drv["name"], "team": team,
                "team_color": TEAM_COLORS.get(team, "#888"),
                "flag": flag,
                "grid_position": i + 1,  # will be re-ranked by model
                "finish_position": None,
                "circuit_type": circuit_type, "is_street": is_street,
                "driver_pace": drv["pace"], "driver_consistency": drv["consistency"],
                "driver_qualifying": drv["qualifying"], "driver_overtaking": drv["overtaking"],
                "driver_wet_skill": drv["wet_skill"], "driver_reg_adaptation": drv["reg_adaptation"],
                "car_speed": car.get("speed", 6), "car_reliability": car.get("reliability", 6),
                "car_aero": car.get("aero_efficiency", 6),
                "car_active_aero": car.get("active_aero_mastery", 6),
                "circuit_type_mult": ct_mult,
                "grid_finish_corr": grid_corr,
                "new_team": 1 if team in ("Audi", "Cadillac") else 0,
                "rookie": 1 if drv.get("rookie") else 0,
                "real_avg_finish_2024": stats["avg_finish"],
                "real_wins_2024": stats["wins"],
                "real_podiums_2024": stats["podiums"],
                "real_dnf_rate": stats["dnfs"] / max(stats["races"], 1),
            })

        return pd.DataFrame(rows)
