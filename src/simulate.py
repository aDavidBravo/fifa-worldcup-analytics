"""Monte Carlo simulation of the remaining 2026 World Cup bracket.

Simulates the confirmed bracket from the quarter-finals onward using
Elo-based win probabilities: SF1 = winner(FRA-MAR) vs winner(ESP-BEL),
SF2 = winner(ARG-SUI) vs winner(ENG-NOR).
"""
import json
import random
from pathlib import Path

from elo import compute_ratings, win_probability

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

N_SIMS = 10_000
SEED = 2026

# (home, away) as scheduled; venue is neutral for all remaining matches
QUARTER_FINALS = [
    ("France", "Morocco"),      # QF1 - Jul 9, Boston
    ("Spain", "Belgium"),       # QF2 - Jul 10, Los Angeles
    ("Argentina", "Switzerland"),  # QF3 - Jul 11, Kansas City
    ("England", "Norway"),      # QF4 - Jul 11, Miami
]
# semi-finals: (QF1 winner vs QF2 winner), (QF3 winner vs QF4 winner)


def play(a: str, b: str, ratings, rng) -> str:
    return a if rng.random() < win_probability(ratings[a], ratings[b]) else b


def simulate():
    ratings = compute_ratings()
    rng = random.Random(SEED)
    teams = sorted({t for qf in QUARTER_FINALS for t in qf})
    counts = {t: {"semi": 0, "final": 0, "champion": 0} for t in teams}

    for _ in range(N_SIMS):
        sf1_a = play(*QUARTER_FINALS[0], ratings, rng)
        sf1_b = play(*QUARTER_FINALS[1], ratings, rng)
        sf2_a = play(*QUARTER_FINALS[2], ratings, rng)
        sf2_b = play(*QUARTER_FINALS[3], ratings, rng)
        for t in (sf1_a, sf1_b, sf2_a, sf2_b):
            counts[t]["semi"] += 1
        f1 = play(sf1_a, sf1_b, ratings, rng)
        f2 = play(sf2_a, sf2_b, ratings, rng)
        counts[f1]["final"] += 1
        counts[f2]["final"] += 1
        counts[play(f1, f2, ratings, rng)]["champion"] += 1

    probs = [
        {
            "team": t,
            "elo": round(ratings[t]),
            "p_semi": round(100 * c["semi"] / N_SIMS, 1),
            "p_final": round(100 * c["final"] / N_SIMS, 1),
            "p_champion": round(100 * c["champion"] / N_SIMS, 1),
        }
        for t, c in counts.items()
    ]
    probs.sort(key=lambda d: -d["p_champion"])

    qf_probs = [
        {"home": h, "away": a,
         "p_home": round(100 * win_probability(ratings[h], ratings[a]), 1)}
        for h, a in QUARTER_FINALS
    ]

    out = {"n_sims": N_SIMS, "as_of": "2026-07-08",
           "method": "Elo (full 1872-2026 history) + Monte Carlo",
           "champion_probs": probs, "qf_probs": qf_probs}
    (OUT / "predictions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote predictions.json")
    for p in probs:
        print(f"{p['team']:<13} elo {p['elo']}  champion {p['p_champion']:>5}%  final {p['p_final']:>5}%")


if __name__ == "__main__":
    simulate()
