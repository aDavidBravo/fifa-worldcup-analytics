"""Build processed datasets for the World Cup dashboard.

Reads the Fjelstul World Cup Database (1930-2022) and the martj42
international results dataset (includes the ongoing 2026 World Cup),
and writes aggregated JSON files to data/processed/.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# 2026 knockout stage date windows (48-team format)
WC2026_STAGES = [
    ("Group stage", "2026-06-11", "2026-06-27"),
    ("Round of 32", "2026-06-28", "2026-07-03"),
    ("Round of 16", "2026-07-04", "2026-07-07"),
    ("Quarter-finals", "2026-07-09", "2026-07-11"),
    ("Semi-finals", "2026-07-14", "2026-07-15"),
    ("Third place", "2026-07-18", "2026-07-18"),
    ("Final", "2026-07-19", "2026-07-19"),
]


def save(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {name}.json")


def build_historical():
    tournaments = pd.read_csv(RAW / "tournaments.csv")
    matches = pd.read_csv(RAW / "matches.csv")
    goals = pd.read_csv(RAW / "goals.csv")
    awards = pd.read_csv(RAW / "award_winners.csv")
    standings = pd.read_csv(RAW / "tournament_standings.csv")
    teams = pd.read_csv(RAW / "teams.csv")

    # the Fjelstul database also ships FIFA Women's World Cups; this project
    # covers the men's tournament only
    tournaments = tournaments[tournaments["tournament_name"].str.contains("Men's")]
    mens_ids = set(tournaments["tournament_id"])
    matches = matches[matches["tournament_id"].isin(mens_ids)]
    goals = goals[goals["tournament_id"].isin(mens_ids)]
    awards = awards[awards["tournament_id"].isin(mens_ids)]
    standings = standings[standings["tournament_id"].isin(mens_ids)]
    print(f"men's tournaments: {len(tournaments)} ({tournaments['year'].min()}-{tournaments['year'].max()})")

    conf_of = dict(zip(teams["team_name"], teams["confederation_code"]))

    # --- per-tournament summary ---
    m_agg = matches.groupby("tournament_id").agg(
        matches=("match_id", "nunique"),
        goals_home=("home_team_score", "sum"),
        goals_away=("away_team_score", "sum"),
    )
    m_agg["goals"] = m_agg["goals_home"] + m_agg["goals_away"]
    t = tournaments.set_index("tournament_id").join(m_agg)
    per_tournament = [
        {
            "year": int(r.year),
            "host": r.host_country,
            "winner": r.winner,
            "teams": int(r.count_teams),
            "matches": int(r.matches),
            "goals": int(r.goals),
            "goals_per_match": round(r.goals / r.matches, 2),
        }
        for r in t.itertuples()
    ]

    # --- titles and podiums by country ---
    top4 = standings[standings["position"] <= 4]
    titles = {}
    for r in top4.itertuples():
        d = titles.setdefault(
            r.team_name,
            {"team": r.team_name, "confederation": conf_of.get(r.team_name, "?"),
             "titles": 0, "runner_up": 0, "top4": 0},
        )
        d["top4"] += 1
        if r.position == 1:
            d["titles"] += 1
        elif r.position == 2:
            d["runner_up"] += 1
    titles = sorted(titles.values(), key=lambda d: (-d["titles"], -d["runner_up"], -d["top4"]))

    # --- team all-time records ---
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        part = matches[[f"{side}_team_name", f"{side}_team_score", f"{opp}_team_score",
                        f"{side}_team_win", "draw", "tournament_id"]].copy()
        part.columns = ["team", "gf", "ga", "win", "draw", "tournament_id"]
        rows.append(part)
    long = pd.concat(rows)
    team_rec = long.groupby("team").agg(
        appearances=("tournament_id", "nunique"),
        played=("gf", "size"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
    ).reset_index()
    team_rec["losses"] = team_rec["played"] - team_rec["wins"] - team_rec["draws"]
    title_count = {d["team"]: d["titles"] for d in titles}
    team_rec["titles"] = team_rec["team"].map(title_count).fillna(0).astype(int)
    team_rec["confederation"] = team_rec["team"].map(conf_of).fillna("?")
    team_records = (
        team_rec.sort_values(["wins", "played"], ascending=False)
        .head(25)
        .astype({"wins": int, "draws": int, "losses": int, "gf": int, "ga": int})
        .to_dict("records")
    )

    # --- confederation summary ---
    long["confederation"] = long["team"].map(conf_of).fillna("?")
    conf = long.groupby("confederation").agg(
        played=("gf", "size"), wins=("win", "sum"), gf=("gf", "sum")
    ).reset_index()
    conf_titles = {}
    conf_top4 = {}
    for d in titles:
        conf_titles[d["confederation"]] = conf_titles.get(d["confederation"], 0) + d["titles"]
        conf_top4[d["confederation"]] = conf_top4.get(d["confederation"], 0) + d["top4"]
    confederations = [
        {
            "confederation": r.confederation,
            "played": int(r.played),
            "win_pct": round(100 * r.wins / r.played, 1),
            "goals": int(r.gf),
            "titles": conf_titles.get(r.confederation, 0),
            "top4": conf_top4.get(r.confederation, 0),
        }
        for r in conf.itertuples() if r.confederation != "?"
    ]
    confederations.sort(key=lambda d: -d["titles"])

    # --- top scorers all-time (own goals excluded) ---
    g = goals[goals["own_goal"] == 0].copy()
    g["player"] = (g["given_name"].replace("not applicable", "") + " " + g["family_name"]).str.strip()
    sc = g.groupby(["player_id", "player"]).agg(
        goals=("goal_id", "count"),
        team=("player_team_name", "first"),
        tournaments=("tournament_id", "nunique"),
    ).reset_index().sort_values("goals", ascending=False).head(15)
    top_scorers = sc[["player", "team", "goals", "tournaments"]].to_dict("records")

    # --- award winners (Golden Ball / Golden Boot) ---
    aw = awards[awards["award_name"].isin(["Golden Ball", "Golden Boot"])].copy()
    aw["player"] = (aw["given_name"].replace("not applicable", "") + " " + aw["family_name"]).str.strip()
    aw["year"] = aw["tournament_name"].str.extract(r"(\d{4})").astype(int)
    award_winners = aw[["year", "award_name", "player", "team_name"]].rename(
        columns={"award_name": "award", "team_name": "team"}
    ).sort_values("year").to_dict("records")

    kpis = {
        "tournaments": int(len(tournaments)),
        "matches": int(matches["match_id"].nunique()),
        "goals": int(t["goals"].sum()),
        "teams_ever": int(matches["home_team_name"].nunique()),
        "span": "1930-2022",
    }

    save("historical", {
        "kpis": kpis,
        "per_tournament": per_tournament,
        "titles": titles,
        "team_records": team_records,
        "confederations": confederations,
        "top_scorers": top_scorers,
        "award_winners": award_winners,
    })
    return conf_of


def build_wc2026(conf_of):
    res = pd.read_csv(RAW / "international_results.csv")
    shoot = pd.read_csv(RAW / "shootouts.csv")
    wc = res[(res["tournament"] == "FIFA World Cup") & (res["date"] >= "2026-01-01")].copy()
    print(f"2026 WC rows: {len(wc)}")

    def stage_of(date):
        for name, lo, hi in WC2026_STAGES:
            if lo <= date <= hi:
                return name
        return "?"

    wc["stage"] = wc["date"].map(stage_of)
    shoot_winner = {
        (r.date, r.home_team, r.away_team): r.winner
        for r in shoot[shoot["date"] >= "2026-01-01"].itertuples()
    }

    played = wc.dropna(subset=["home_score"]).copy()
    upcoming = wc[wc["home_score"].isna()].copy()

    match_list = []
    for r in played.itertuples():
        winner = None
        if r.home_score > r.away_score:
            winner = r.home_team
        elif r.away_score > r.home_score:
            winner = r.away_team
        else:
            winner = shoot_winner.get((r.date, r.home_team, r.away_team))
        match_list.append({
            "date": r.date, "stage": r.stage,
            "home": r.home_team, "away": r.away_team,
            "score": f"{int(r.home_score)}-{int(r.away_score)}",
            "winner": winner,
            "shootout": (r.date, r.home_team, r.away_team) in shoot_winner,
            "city": r.city,
        })

    # per-team record in the 2026 tournament
    teams_2026 = {}
    for m in match_list:
        for side, other in (("home", "away"), ("away", "home")):
            name = m[side]
            gf, ga = map(int, m["score"].split("-"))
            if side == "away":
                gf, ga = ga, gf
            d = teams_2026.setdefault(name, {
                "team": name, "confederation": conf_of.get(name, "?"),
                "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0,
            })
            d["played"] += 1
            d["gf"] += gf
            d["ga"] += ga
            if m["winner"] is None or m["shootout"]:
                d["draws"] += 1
            elif m["winner"] == name:
                d["wins"] += 1
            else:
                d["losses"] += 1

    alive = sorted({t for m in match_list if m["stage"] == "Round of 16" and m["winner"]
                    for t in [m["winner"]]})
    qf = [{"date": r.date, "home": r.home_team, "away": r.away_team, "city": r.city}
          for r in upcoming.itertuples() if r.stage == "Quarter-finals"]

    total_goals = int(played["home_score"].sum() + played["away_score"].sum())
    stage_counts = played.groupby("stage")["date"].count().to_dict()
    print("stage counts:", stage_counts)

    save("wc2026", {
        "kpis": {
            "matches_played": int(len(played)),
            "goals": total_goals,
            "goals_per_match": round(total_goals / len(played), 2),
            "quarterfinalists": alive,
        },
        "matches": match_list,
        "teams": sorted(teams_2026.values(), key=lambda d: (-d["wins"], -d["gf"])),
        "quarter_finals": qf,
    })


if __name__ == "__main__":
    conf_of = build_historical()
    build_wc2026(conf_of)
