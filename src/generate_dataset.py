"""Generate a synthetic dataset of login / network-session events.

20,000 events, ~11.9% suspicious, four attack types. To keep the problem
realistic:
  * ~22% of attacks are "low and slow" and look almost like normal users,
  * ~4% of normal events look odd (travel, new laptop, forgotten password),
  * 2-4% of values in four columns are removed (missing data).

Run:  python src/generate_dataset.py
"""
import numpy as np
import pandas as pd

from config import DATA_PATH, N_EVENTS, SEED, ensure_dirs

ATTACK_COUNTS = {
    "brute_force": 684,
    "phishing_login": 622,
    "credential_stuffing": 606,
    "data_exfiltration": 469,
}
LOW_AND_SLOW_SHARE = 0.22
NOISY_NORMAL_SHARE = 0.04
MISSING_RATES = {
    "requests_per_min": 0.03,
    "ip_reputation": 0.02,
    "session_minutes": 0.04,
    "device_type": 0.025,
}

COUNTRIES = ["domestic", "nearby", "international"]
DEVICES = ["laptop", "mobile", "tablet", "unknown"]

# Per attack type: failed-attempt rate, requests (shape, scale), bytes (shape, scale),
# ip reputation beta(a, b), geo distance mean, P(new device), P(vpn),
# session (shape, scale), country probabilities, device probabilities.
ATTACK_PROFILES = {
    "brute_force": dict(
        failed=6.0, rpm=(3, 25), bytes=(1.5, 2.5), iprep=(4, 3.5), geo=120,
        new_device=0.30, vpn=0.25, session=(1.5, 6),
        country=[0.65, 0.15, 0.20], device=[0.50, 0.25, 0.08, 0.17],
    ),
    "credential_stuffing": dict(
        failed=4.5, rpm=(3, 22), bytes=(1.5, 2.5), iprep=(3.5, 3.5), geo=250,
        new_device=0.40, vpn=0.40, session=(1.5, 7),
        country=[0.55, 0.20, 0.25], device=[0.45, 0.30, 0.08, 0.17],
    ),
    "data_exfiltration": dict(
        failed=1.0, rpm=(4, 7), bytes=(2.5, 12), iprep=(5, 3), geo=100,
        new_device=0.25, vpn=0.30, session=(3, 8),
        country=[0.70, 0.12, 0.18], device=[0.65, 0.20, 0.05, 0.10],
    ),
    "phishing_login": dict(
        failed=1.5, rpm=(4, 6), bytes=(2, 3), iprep=(4, 3), geo=400,
        new_device=0.65, vpn=0.30, session=(2, 7),
        country=[0.50, 0.20, 0.30], device=[0.40, 0.40, 0.05, 0.15],
    ),
}


def normal_events(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Ordinary users: office hours, few failures, good IPs, long sessions."""
    return pd.DataFrame(
        {
            "hour_of_day": np.clip(np.round(rng.normal(13.5, 3.5, n)), 0, 23),
            "failed_attempts": rng.poisson(0.5, n),
            "requests_per_min": rng.gamma(4, 3.7, n),
            "bytes_out_mb": rng.gamma(1.6, 1.9, n),
            "ip_reputation": 100 * rng.beta(7, 2, n),
            "geo_distance_km": rng.exponential(40, n),
            "new_device": (rng.random(n) < 0.07).astype(int),
            "is_vpn": (rng.random(n) < 0.14).astype(int),
            "session_minutes": rng.gamma(3, 11.7, n),
            "country_group": rng.choice(COUNTRIES, n, p=[0.80, 0.13, 0.07]),
            "device_type": rng.choice(DEVICES, n, p=[0.52, 0.34, 0.12, 0.02]),
        }
    )


def attack_events(rng: np.random.Generator, kind: str, n: int) -> pd.DataFrame:
    p = ATTACK_PROFILES[kind]
    # Attacks favour the night: 60% between 00:00 and 06:59, the rest any time.
    night = rng.random(n) < 0.6
    hour = np.where(night, rng.integers(0, 7, n), rng.integers(0, 24, n))
    return pd.DataFrame(
        {
            "hour_of_day": hour,
            "failed_attempts": rng.poisson(p["failed"], n),
            "requests_per_min": rng.gamma(*p["rpm"], n),
            "bytes_out_mb": rng.gamma(*p["bytes"], n),
            "ip_reputation": 100 * rng.beta(*p["iprep"], n),
            "geo_distance_km": rng.exponential(p["geo"], n),
            "new_device": (rng.random(n) < p["new_device"]).astype(int),
            "is_vpn": (rng.random(n) < p["vpn"]).astype(int),
            "session_minutes": rng.gamma(*p["session"], n),
            "country_group": rng.choice(COUNTRIES, n, p=p["country"]),
            "device_type": rng.choice(DEVICES, n, p=p["device"]),
        }
    )


def low_and_slow(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Stealthy attacks: normal-looking behaviour with only slight shifts."""
    df = normal_events(rng, n)
    df["failed_attempts"] += rng.poisson(0.7, n)
    df["ip_reputation"] -= rng.uniform(0, 15, n)
    df["new_device"] = np.maximum(df["new_device"], (rng.random(n) < 0.15).astype(int))
    return df


def make_noisy(rng: np.random.Generator, df: pd.DataFrame, idx: np.ndarray) -> None:
    """Honest users that look odd (travel, new device, forgotten password)."""
    n = len(idx)
    night = rng.random(n) < 0.5
    df.loc[idx, "hour_of_day"] = np.where(night, rng.integers(0, 7, n), rng.integers(0, 24, n))
    df.loc[idx, "failed_attempts"] = rng.poisson(3.5, n)
    df.loc[idx, "requests_per_min"] = rng.gamma(3, 15, n)
    df.loc[idx, "bytes_out_mb"] = rng.gamma(2, 4, n)
    df.loc[idx, "ip_reputation"] = 100 * rng.beta(4, 3.5, n)
    df.loc[idx, "geo_distance_km"] = rng.exponential(250, n)
    df.loc[idx, "session_minutes"] = rng.gamma(2, 8, n)
    df.loc[idx, "is_vpn"] = (rng.random(n) < 0.5).astype(int)
    df.loc[idx, "new_device"] = (rng.random(n) < 0.5).astype(int)
    df.loc[idx, "country_group"] = rng.choice(COUNTRIES, n, p=[0.45, 0.20, 0.35])


def generate(seed: int = SEED, n_events: int = N_EVENTS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_attacks = sum(ATTACK_COUNTS.values())

    normal = normal_events(rng, n_events - n_attacks)
    noisy_idx = rng.choice(len(normal), int(NOISY_NORMAL_SHARE * len(normal)), replace=False)
    make_noisy(rng, normal, noisy_idx)
    normal["attack_type"] = "none"
    normal["label"] = 0

    parts = [normal]
    for kind, count in ATTACK_COUNTS.items():
        n_stealth = int(round(LOW_AND_SLOW_SHARE * count))
        part = pd.concat(
            [attack_events(rng, kind, count - n_stealth), low_and_slow(rng, n_stealth)],
            ignore_index=True,
        )
        part["attack_type"] = kind
        part["label"] = 1
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Clip to realistic ranges and round.
    df["failed_attempts"] = df["failed_attempts"].clip(0, 15).astype(int)
    df["requests_per_min"] = df["requests_per_min"].clip(1, 130).round(1)
    df["bytes_out_mb"] = df["bytes_out_mb"].clip(0.01, 35).round(2)
    df["ip_reputation"] = df["ip_reputation"].clip(0, 100).round(1)
    df["geo_distance_km"] = df["geo_distance_km"].clip(0, 2500).round(1)
    df["session_minutes"] = df["session_minutes"].clip(0.5, 240).round(1)
    df["hour_of_day"] = df["hour_of_day"].astype(int)

    # Missing data.
    for col, rate in MISSING_RATES.items():
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan

    # Pseudonymised identifiers (no real personal data).
    df.insert(0, "event_id", [f"EVT-{i:05d}" for i in range(1, len(df) + 1)])
    df.insert(1, "user_id", [f"U{u:04d}" for u in rng.integers(1, 2500, len(df))])
    return df


def main() -> pd.DataFrame:
    ensure_dirs()
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df):,} events -> {DATA_PATH}")
    print(f"  suspicious share: {df['label'].mean():.1%}")
    print("  attack types:", df.loc[df.label == 1, "attack_type"].value_counts().to_dict())
    return df


if __name__ == "__main__":
    main()
