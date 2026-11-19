"""Shared paths, constants and feature lists for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "login_events.csv"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42
N_EVENTS = 20_000
TEST_SIZE = 0.25

NUMERIC_FEATURES = [
    "hour_of_day",
    "failed_attempts",
    "requests_per_min",
    "bytes_out_mb",
    "ip_reputation",
    "geo_distance_km",
    "new_device",
    "is_vpn",
    "session_minutes",
]
CATEGORICAL_FEATURES = ["country_group", "device_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"


def ensure_dirs() -> None:
    for d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
