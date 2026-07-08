# ⚽ FIFA World Cup Analytics — 1930 → 2026

**Interactive analytics dashboard + a forecast of the ongoing 2026 FIFA World Cup, benchmarked
against three machine-learning models on a temporal hold-out.**

96 years of World Cup history — every match, every goal, every champion — combined with
live data from the first 48-team World Cup being played right now in the United States,
Canada and Mexico, an Elo/Monte Carlo forecast of who lifts the trophy on July 19, and a
**Model Lab** that puts that forecast up against Dixon-Coles Poisson, XGBoost and a neural
network to prove it's the right call — not just the simplest one.

> 🔮 **Forecast as of July 8, 2026 (quarter-finals set):**
> Argentina 28.9% · Spain 27.4% · France 19.7% · England 12.6%
>
> 🧪 **Model Lab verdict:** on 3,694 held-out matches, XGBoost and Dixon-Coles Poisson beat
> the Elo baseline by **under 1% in log-loss** — and Elo keeps the best raw accuracy (60.7%).
> The interpretable model stays as the headline forecast because the evidence says so.

## What's inside

| Section | Content |
|---|---|
| **2026 Forecast** | Championship probabilities from 10,000 Monte Carlo simulations, tie-by-tie bracket odds, current world Elo top 12 |
| **2026 Tournament** | Golden Boot race, quarter-finalists' form across all 96 matches played |
| **Historical Analysis** | Goals-per-match evolution since 1930, tournament growth (13 → 48 teams) |
| **Teams & Confederations** | Titles and finals by country, confederation scoreboard (UEFA 12 vs CONMEBOL 10) |
| **Players** | All-time World Cup top scorers, Golden Ball winners |
| **Model Lab** | Elo vs. Dixon-Coles Poisson vs. XGBoost vs. neural net — log-loss, Brier, accuracy, calibration curves, feature importance, and each model's own 2026 champion pick |

Every chart is flag- and country-color coded (57 self-hosted flag assets, ~65 KB total,
zero external image requests) so 60+ national teams read at a glance.

## Data sources

- **[Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup)** — all 22 men's
  tournaments 1930–2022: matches, goals, squads, standings, awards (curated, peer-reviewed).
- **[martj42/international_results](https://github.com/martj42/international_results)** —
  49,000+ international matches since 1872, updated daily; includes the ongoing 2026 World Cup.
- **FIFA.com** — 2026 Golden Boot standings.

## Model Lab — does advanced ML actually beat a good Elo model?

A common portfolio failure mode is reaching for a neural network because it *sounds*
senior. Here the opposite question is asked and answered with numbers: **is the extra
complexity earning its keep?**

**Setup.** One leak-free feature set — pre-match Elo gap, rolling form, goal rates, rest
days, venue, tournament importance — built with a single chronological pass over 23,367
internationals (2002–2026), so no feature ever sees a result before it happened. Trained on
19,673 matches (2002–2022), evaluated on **3,694 matches the models never saw** (2023–2026,
including this World Cup) — a temporal split, not a random shuffle, because random
splits leak future form into the past.

| Model | Log-loss ↓ | Brier ↓ | Accuracy ↑ |
|---|---|---|---|
| **Dixon-Coles Poisson** | **0.8572** | **0.5036** | 60.6% |
| XGBoost | 0.8573 | 0.5048 | 60.5% |
| Neural net (MLP) | 0.8605 | 0.5069 | 60.1% |
| Elo baseline (logistic) | 0.8657 | 0.5092 | **60.7%** |

*3,694 held-out matches, 2023–2026. Lower log-loss/Brier is better; higher accuracy is better.*

**Reading it like a data scientist, not a leaderboard:** the gap between the best model and
the simplest one is **0.0085 log-loss** — noise-level, and Elo actually keeps the best
accuracy. With only ~64 World-Cup matches of real signal, a gradient-boosted tree or a
neural net has nothing extra to learn that a well-tuned Elo rating hasn't already captured.
XGBoost's own feature importances confirm it — `elo_diff` alone accounts for **21.5%** of
total gain, more than the next five features combined.

**So the forecast you see stays Elo + Monte Carlo** — not by default, but because it was
measured against three legitimate alternatives and won on the metric that matters
(log-loss) essentially in a tie while staying fully interpretable. That comparison, not
the model itself, is the deliverable.

### Feature engineering — no leakage, one pass, four state stores

```python
# src/features.py — every feature is the pre-match snapshot; state updates AFTER logging it
for r in df.itertuples():
    h, a = r.home_team, r.away_team
    eh, ea = elo[h], elo[a]                       # rating BEFORE this match
    adv = 0.0 if neutral else HOME_ADV

    row = {
        "elo_diff": (eh + adv) - ea,
        "form_home": mean(form_pts[h], 1.0),      # last-5 rolling points, home team
        "gf_home": mean(form_gf[h], 1.2),          # last-5 rolling goals-for
        "rest_home": min((date - last_date[h]).days, 365),
        "importance": k_factor(r.tournament),      # World Cup=60 ... friendly=20
        "target": result_target(r.home_score, r.away_score),
    }
    rows.append(row)

    # Elo, form and rest are updated ONLY here — after the row is already recorded
    elo[h], elo[a] = elo[h] + delta, elo[a] - delta
    form_pts[h].append(home_points); last_date[h] = last_date[a] = date
```

### Dixon-Coles Poisson — maximum likelihood from scratch (SciPy, no black box)

The winning model isn't a library call — it's two Poisson GLMs for expected goals plus the
[Dixon & Coles (1997)](https://www.jstor.org/stable/2986290) low-score correlation
correction, with `ρ` fit by direct log-likelihood scan:

```python
# src/train_models.py
def dc_tau(i, j, lh, la, rho):
    """Dixon-Coles correction for the four low-scoring cells (0-0, 1-0, 0-1, 1-1),
    where two independent Poissons underestimate real match correlation."""
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0

def score_matrix_probs(lh, la, rho):
    ph, pa = poisson.pmf(np.arange(MAXG + 1), lh), poisson.pmf(np.arange(MAXG + 1), la)
    M = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= dc_tau(i, j, lh, la, rho)
    M /= M.sum()
    return np.array([np.triu(M, 1).sum(), np.trace(M), np.tril(M, -1).sum()])  # away, draw, home

# rho fit by scanning training log-likelihood — not gradient descent, fully auditable
for rho in np.linspace(-0.2, 0.2, 21):
    probs = np.clip([score_matrix_probs(a, b, rho) for a, b in zip(lh, la)], 1e-9, 1)
    ll = np.log(probs[np.arange(len(probs)), y]).sum()
    if ll > best: best, brho = ll, rho
```

### Validation — the metrics that actually matter for probabilistic forecasts

Accuracy alone is a poor way to judge a forecaster — a model that says "60% home win"
and is wrong 40% of the time should be penalized less than one that said "95%." That's
what log-loss and calibration measure and accuracy doesn't:

```python
# src/train_models.py
def multiclass_brier(y, proba):
    """Brier score: mean squared error between predicted probabilities and the
    one-hot true outcome. Rewards well-calibrated confidence, not just correct picks."""
    oh = np.zeros_like(proba)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))

def calibration_curve(y_home, p_home, bins=10):
    """Reliability check: among matches where the model said '70% home win',
    did the home team actually win ~70% of the time?"""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p_home >= edges[i]) & (p_home < edges[i + 1])
        if m.sum() >= 20:
            out.append({"p_pred": p_home[m].mean(), "p_true": y_home[m].mean(), "n": int(m.sum())})
    return out
```

All four models' calibration curves sit almost exactly on the diagonal (see the dashboard's
**Model Lab** section) — none of them is overconfident, which is the real risk with small
football datasets.

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
4. **Features** (`src/features.py`) — one leak-free chronological pass builds 23,367 rows of
   pre-match features (Elo gap, rolling form, goal rates, rest days, venue, importance).
5. **Model Lab** (`src/train_models.py`) — Elo-logistic baseline, Dixon-Coles Poisson
   (SciPy MLE), XGBoost and a scikit-learn MLP, trained on 2002–2022 and scored on a
   2023–2026 temporal hold-out with log-loss, Brier, accuracy and calibration.
6. **Dashboard** (`docs/`) — Apache ECharts on a dark analytics theme; static site,
   no backend required (served by GitHub Pages). Flags and per-country colors are
   self-hosted (`docs/flags/`, ~65 KB for 57 countries) — no runtime calls to a flag API.

## Live dashboard

**https://adavidbravo.github.io/fifa-worldcup-analytics/**

## Reproduce

```bash
pip install -r requirements.txt

# historical data + live 2026 forecast
python src/build_dataset.py     # aggregates            -> data/processed/historical.json, wc2026.json
python src/elo.py               # Elo ratings            -> data/processed/elo.json
python src/simulate.py          # Monte Carlo forecast   -> data/processed/predictions.json

# Model Lab: feature engineering + 4-model comparison
python src/features.py          # leak-free features     -> data/processed/features.csv
python src/train_models.py      # train + validate 4 models -> data/processed/models.json

# bundle everything for the static dashboard
python src/export_dashboard.py  # bundle                 -> docs/data/data.js
python -m http.server 8123 --directory docs
```

Then open http://localhost:8123.

To refresh the forecast as the tournament progresses: re-download
`international_results.csv` + `shootouts.csv`, update the bracket in `simulate.py` and
`train_models.py`, and re-run the pipeline above.

## Stack

Python 3.13 · pandas / NumPy · scikit-learn · XGBoost · SciPy (Dixon-Coles MLE) ·
Apache ECharts · vanilla JS/CSS (zero build step)

## License & attribution

Data belongs to its original maintainers (see sources above). Analysis and dashboard code
free to reuse with attribution. Probabilities are model estimates, not betting advice.
