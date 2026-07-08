# ⚽ FIFA World Cup Analytics — 1930 → 2026

**Interactive analytics dashboard + Elo-based Monte Carlo forecast of the ongoing 2026 FIFA World Cup.**

96 years of World Cup history — every match, every goal, every champion — combined with
live data from the first 48-team World Cup being played right now in the United States,
Canada and Mexico, and a probabilistic forecast of who lifts the trophy on July 19.

> 🔮 **Forecast as of July 8, 2026 (quarter-finals set):**
> Argentina 28.9% · Spain 27.4% · France 19.7% · England 12.6%

## What's inside

| Section | Content |
|---|---|
| **2026 Forecast** | Championship probabilities from 10,000 Monte Carlo simulations, tie-by-tie bracket odds, current world Elo top 12 |
| **2026 Tournament** | Golden Boot race, quarter-finalists' form across all 96 matches played |
| **Historical Analysis** | Goals-per-match evolution since 1930, tournament growth (13 → 48 teams) |
| **Teams & Confederations** | Titles and finals by country, confederation scoreboard (UEFA 12 vs CONMEBOL 10) |
| **Players** | All-time World Cup top scorers, Golden Ball winners |

## Data sources

- **[Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup)** — all 22 men's
  tournaments 1930–2022: matches, goals, squads, standings, awards (curated, peer-reviewed).
- **[martj42/international_results](https://github.com/martj42/international_results)** —
  49,000+ international matches since 1872, updated daily; includes the ongoing 2026 World Cup.
- **FIFA.com** — 2026 Golden Boot standings.

## Methodology

1. **ETL** (`src/build_dataset.py`) — pandas pipeline that filters, aggregates and validates
   the raw CSVs into analysis-ready JSON (men's tournaments only; stage inference for 2026
   validated against the official match counts: 72 group + 16 + 8).
2. **Ratings** (`src/elo.py`) — football Elo over the full 1872–2026 history:
   K = 60 (World Cup) → 20 (friendlies), goal-margin multiplier, +100 home advantage
   on non-neutral venues.
3. **Forecast** (`src/simulate.py`) — the confirmed quarter-final bracket simulated 10,000
   times with the logistic Elo win expectancy `P(A) = 1 / (1 + 10^(−ΔElo/400))`; seeded RNG
   for reproducibility.
4. **Dashboard** (`dashboard/`) — Apache ECharts on a dark analytics theme; static site,
   no backend required.

## Reproduce

```bash
pip install -r requirements.txt
python src/build_dataset.py   # aggregates -> data/processed/
python src/elo.py             # Elo ratings -> elo.json
python src/simulate.py        # Monte Carlo  -> predictions.json
python src/export_dashboard.py# bundle       -> dashboard/data/data.js
python -m http.server 8123 --directory dashboard
```

Then open http://localhost:8123.

To refresh the forecast as the tournament progresses: re-download
`international_results.csv`, update the bracket in `simulate.py`, and re-run the pipeline.

## Stack

Python 3.13 · pandas · Apache ECharts · vanilla JS/CSS (zero build step)

## License & attribution

Data belongs to its original maintainers (see sources above). Analysis and dashboard code
free to reuse with attribution. Probabilities are model estimates, not betting advice.
