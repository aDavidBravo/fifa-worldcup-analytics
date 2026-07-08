"""Elo ratings for national teams from the full international match history.

Standard football Elo (eloratings.net methodology): K scaled by match
importance and goal margin, 100-point home advantage for non-neutral venues.
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

BASE = 1500.0
HOME_ADV = 100.0

K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50, "UEFA Euro": 50, "African Cup of Nations": 50,
    "AFC Asian Cup": 50, "Gold Cup": 50, "Oceania Nations Cup": 50,
    "Confederations Cup": 50, "UEFA Nations League": 40,
    "Friendly": 20,
}


def k_factor(tournament: str) -> float:
    if tournament in K_BY_TOURNAMENT:
        return K_BY_TOURNAMENT[tournament]
    if "qualification" in tournament:
        return 40
    return 30  # other competitive tournaments


def compute_ratings(as_of: str | None = None) -> dict[str, float]:
    df = pd.read_csv(RAW / "international_results.csv").dropna(subset=["home_score"])
    if as_of:
        df = df[df["date"] <= as_of]
    ratings: dict[str, float] = {}
    for r in df.itertuples():
        rh = ratings.get(r.home_team, BASE)
        ra = ratings.get(r.away_team, BASE)
        adv = 0.0 if r.neutral else HOME_ADV
        expected_home = 1.0 / (1.0 + 10 ** ((ra - (rh + adv)) / 400.0))
        if r.home_score > r.away_score:
            actual = 1.0
        elif r.home_score < r.away_score:
            actual = 0.0
        else:
            actual = 0.5
        margin = abs(int(r.home_score) - int(r.away_score))
        g = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        delta = k_factor(r.tournament) * g * (actual - expected_home)
        ratings[r.home_team] = rh + delta
        ratings[r.away_team] = ra - delta
    return ratings


def win_probability(elo_a: float, elo_b: float) -> float:
    """P(A beats B) on neutral ground, draws resolved (knockout)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


if __name__ == "__main__":
    ratings = compute_ratings()
    top30 = sorted(ratings.items(), key=lambda kv: -kv[1])[:30]
    out = [{"rank": i + 1, "team": t, "elo": round(e)} for i, (t, e) in enumerate(top30)]
    (OUT / "elo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote elo.json")
    for row in out[:12]:
        print(f"{row['rank']:>2} {row['team']:<15} {row['elo']}")
