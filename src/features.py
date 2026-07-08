"""Feature engineering for the match-outcome models.

Single chronological pass over the full international match history. Every
feature for a match is computed from information available *before* kickoff
(no leakage): rolling Elo, recent form, goal rates, rest days, venue and
tournament importance. Outputs data/processed/features.csv.

Target: home-team result -> 0 away win, 1 draw, 2 home win.
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from elo import BASE, HOME_ADV, k_factor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

FORM_WINDOW = 5
TRAIN_FROM = "2002-01-01"   # modern era: enough matches with warm rolling features

# coarse confederation map for teams that recur at World Cups (else OTH)
CONF = {
    "UEFA": ["Spain", "France", "England", "Germany", "West Germany", "Italy", "Netherlands",
             "Portugal", "Belgium", "Croatia", "Switzerland", "Norway", "Sweden", "Denmark",
             "Poland", "Austria", "Serbia", "Scotland", "Wales", "Ukraine", "Turkey", "Russia",
             "Czech Republic", "Czechoslovakia", "Hungary", "Bulgaria", "Greece", "Romania",
             "Bosnia and Herzegovina", "Yugoslavia", "Soviet Union"],
    "CONMEBOL": ["Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru", "Paraguay",
                 "Ecuador", "Bolivia", "Venezuela"],
    "CAF": ["Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon", "Egypt", "Algeria", "Tunisia",
            "Ivory Coast", "South Africa", "Cape Verde", "DR Congo"],
    "CONCACAF": ["United States", "Mexico", "Canada", "Costa Rica", "Panama", "Honduras",
                 "Jamaica", "Haiti", "Curaçao"],
    "AFC": ["Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar", "Iraq",
            "Uzbekistan", "Jordan"],
    "OFC": ["New Zealand"],
}
TEAM_CONF = {t: c for c, ts in CONF.items() for t in ts}


def result_target(hs: int, as_: int) -> int:
    return 2 if hs > as_ else (0 if hs < as_ else 1)


def build():
    df = pd.read_csv(RAW / "international_results.csv").dropna(subset=["home_score"])
    df = df.sort_values("date").reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    elo = defaultdict(lambda: BASE)
    last_date: dict[str, pd.Timestamp] = {}
    played: dict[str, int] = defaultdict(int)
    form_pts: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    form_gf: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    form_ga: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))

    rows = []
    for r in df.itertuples():
        h, a = r.home_team, r.away_team
        date = pd.Timestamp(r.date)
        neutral = 1 if (str(r.neutral).upper() == "TRUE") else 0

        eh, ea = elo[h], elo[a]
        adv = 0.0 if neutral else HOME_ADV

        def rest(t):
            return min((date - last_date[t]).days, 365) if t in last_date else 365

        def mean(dq, default):
            return float(np.mean(dq)) if len(dq) else default

        row = {
            "date": r.date,
            "home_team": h, "away_team": a,
            "tournament": r.tournament, "neutral": neutral,
            "elo_home": eh, "elo_away": ea,
            "elo_diff": (eh + adv) - ea,
            "form_home": mean(form_pts[h], 1.0), "form_away": mean(form_pts[a], 1.0),
            "gf_home": mean(form_gf[h], 1.2), "gf_away": mean(form_gf[a], 1.2),
            "ga_home": mean(form_ga[h], 1.2), "ga_away": mean(form_ga[a], 1.2),
            "rest_home": rest(h), "rest_away": rest(a),
            "exp_home": min(played[h], 300), "exp_away": min(played[a], 300),
            "importance": k_factor(r.tournament),
            "conf_home": TEAM_CONF.get(h, "OTH"), "conf_away": TEAM_CONF.get(a, "OTH"),
            "home_score": r.home_score, "away_score": r.away_score,
            "target": result_target(r.home_score, r.away_score),
        }
        rows.append(row)

        # --- update state AFTER recording the pre-match snapshot ---
        expected = 1.0 / (1.0 + 10 ** ((ea - (eh + adv)) / 400.0))
        actual = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
        margin = abs(r.home_score - r.away_score)
        g = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        delta = k_factor(r.tournament) * g * (actual - expected)
        elo[h] = eh + delta
        elo[a] = ea - delta

        hp = 3 if actual == 1.0 else (1 if actual == 0.5 else 0)
        form_pts[h].append(hp); form_pts[a].append(3 - hp if hp != 1 else 1)
        form_gf[h].append(r.home_score); form_ga[h].append(r.away_score)
        form_gf[a].append(r.away_score); form_ga[a].append(r.home_score)
        last_date[h] = last_date[a] = date
        played[h] += 1; played[a] += 1

    feat = pd.DataFrame(rows)
    feat = feat[feat["date"] >= TRAIN_FROM].reset_index(drop=True)
    feat.to_csv(OUT / "features.csv", index=False, encoding="utf-8")
    print(f"wrote features.csv: {len(feat)} matches, {feat['date'].min()} -> {feat['date'].max()}")
    print("class balance (away/draw/home):",
          feat["target"].value_counts(normalize=True).sort_index().round(3).tolist())
    return feat


if __name__ == "__main__":
    build()
