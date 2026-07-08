"""Train and compare four match-outcome models on a temporal hold-out.

Baseline (Elo logistic) vs. Dixon-Coles Poisson vs. XGBoost vs. neural net
(MLP). Everything is validated on matches the models never saw during
training (temporal split), and scored with the metrics that actually matter
for probabilistic forecasts: multiclass log-loss, Brier score, accuracy and
calibration. Writes data/processed/models.json for the comparison dashboard.

Target classes: 0 away win · 1 draw · 2 home win.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from elo import compute_ratings, win_probability

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
SPLIT_DATE = "2023-01-01"
SEED = 2026
MAXG = 10  # goal cap for the Poisson score matrix

NUM = ["elo_diff", "elo_home", "elo_away", "form_home", "form_away",
       "gf_home", "gf_away", "ga_home", "ga_away", "rest_home", "rest_away",
       "exp_home", "exp_away", "importance", "neutral"]
CAT = ["conf_home", "conf_away"]
CLASSES = [0, 1, 2]


def multiclass_brier(y, proba):
    oh = np.zeros_like(proba)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))


def calibration_curve(y_home, p_home, bins=10):
    """Reliability of the P(home win) forecast."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p_home >= lo) & (p_home < hi) if i < bins - 1 else (p_home >= lo) & (p_home <= hi)
        if m.sum() >= 20:
            out.append({"p_pred": round(float(p_home[m].mean()), 3),
                        "p_true": round(float(y_home[m].mean()), 3),
                        "n": int(m.sum())})
    return out


# ---------- Dixon-Coles Poisson ----------
def dc_tau(i, j, lh, la, rho):
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


def score_matrix_probs(lh, la, rho):
    ph = poisson.pmf(np.arange(MAXG + 1), lh)
    pa = poisson.pmf(np.arange(MAXG + 1), la)
    M = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] *= dc_tau(i, j, lh, la, rho)
    M /= M.sum()
    home = np.tril(M, -1).sum()   # i > j
    draw = np.trace(M)
    away = np.triu(M, 1).sum()    # i < j
    return np.array([away, draw, home])


class PoissonModel:
    """Two Poisson GLMs for goals + Dixon-Coles low-score correction."""
    def __init__(self):
        self.pre = ColumnTransformer([("n", StandardScaler(), NUM),
                                      ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
        self.mh = PoissonRegressor(alpha=1e-3, max_iter=500)
        self.ma = PoissonRegressor(alpha=1e-3, max_iter=500)
        self.rho = 0.0

    def fit(self, df):
        X = self.pre.fit_transform(df)
        self.mh.fit(X, df["home_score"]); self.ma.fit(X, df["away_score"])
        lh = self.mh.predict(X); la = self.ma.predict(X)
        # fit rho by 1-D scan on training log-likelihood of realised results
        y = df["target"].to_numpy()
        best, brho = -1e18, 0.0
        for rho in np.linspace(-0.2, 0.2, 21):
            ll = 0.0
            probs = np.array([score_matrix_probs(a, b, rho) for a, b in zip(lh[:4000], la[:4000])])
            probs = np.clip(probs, 1e-9, 1)
            ll = np.log(probs[np.arange(len(probs)), y[:4000]]).sum()
            if ll > best: best, brho = ll, rho
        self.rho = float(brho)
        return self

    def predict_proba(self, df):
        X = self.pre.transform(df)
        lh = np.clip(self.mh.predict(X), 0.05, 8); la = np.clip(self.ma.predict(X), 0.05, 8)
        return np.array([score_matrix_probs(a, b, self.rho) for a, b in zip(lh, la)])


def main():
    df = pd.read_csv(OUT / "features.csv")
    tr = df[df["date"] < SPLIT_DATE].reset_index(drop=True)
    te = df[df["date"] >= SPLIT_DATE].reset_index(drop=True)
    ytr, yte = tr["target"].to_numpy(), te["target"].to_numpy()
    yte_home = (yte == 2).astype(int)
    print(f"train {len(tr)}  test {len(te)}  ({te['date'].min()} .. {te['date'].max()})")

    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])

    models = {}
    # 1. Elo baseline: multinomial logistic on elo_diff + neutral only
    models["Elo baseline"] = Pipeline([
        ("pre", ColumnTransformer([("n", StandardScaler(), ["elo_diff", "neutral"])])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    # 2. Dixon-Coles Poisson
    models["Dixon-Coles Poisson"] = PoissonModel()
    # 3. XGBoost
    models["XGBoost"] = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.05,
                              subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
                              objective="multi:softprob", num_class=3,
                              eval_metric="mlogloss", tree_method="hist", random_state=SEED)),
    ])
    # 4. Neural network (MLP)
    models["Neural net (MLP)"] = Pipeline([
        ("pre", pre),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=600,
                              early_stopping=True, random_state=SEED)),
    ])

    metrics, calib, fitted = [], {}, {}
    for name, m in models.items():
        if isinstance(m, PoissonModel):
            m.fit(tr); proba = m.predict_proba(te)
        else:
            m.fit(tr, ytr); proba = m.predict_proba(te)
        proba = np.clip(proba, 1e-9, 1); proba /= proba.sum(axis=1, keepdims=True)
        pred = proba.argmax(1)
        metrics.append({
            "model": name,
            "log_loss": round(float(log_loss(yte, proba, labels=CLASSES)), 4),
            "brier": round(multiclass_brier(yte, proba), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
        })
        calib[name] = calibration_curve(yte_home, proba[:, 2])
        fitted[name] = m
        print(f"  {name:22} logloss={metrics[-1]['log_loss']}  "
              f"brier={metrics[-1]['brier']}  acc={metrics[-1]['accuracy']}")

    metrics.sort(key=lambda d: d["log_loss"])

    # feature importance from XGBoost (gain)
    xgb = fitted["XGBoost"].named_steps["clf"]
    feat_names = NUM + list(fitted["XGBoost"].named_steps["pre"]
                            .named_transformers_["c"].get_feature_names_out(CAT))
    imp = sorted(zip(feat_names, xgb.feature_importances_), key=lambda t: -t[1])[:12]
    feature_importance = [{"feature": f, "importance": round(float(v), 4)} for f, v in imp]

    # ---- each model's 2026 champion forecast (same bracket, model win-probs) ----
    champ = model_champion_forecasts(df, fitted)

    out = {
        "test_period": f"{te['date'].min()} to {te['date'].max()}",
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "metrics": metrics,
        "calibration": calib,
        "feature_importance": feature_importance,
        "champion_by_model": champ,
    }
    (OUT / "models.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote models.json — best model:", metrics[0]["model"])


def model_champion_forecasts(df, fitted):
    """Run the QF bracket Monte Carlo under each model's match probabilities."""
    import random
    QF = [("France", "Morocco"), ("Spain", "Belgium"),
          ("Argentina", "Switzerland"), ("England", "Norway")]
    teams = sorted({t for p in QF for t in p})
    elo = compute_ratings()

    # latest feature snapshot per team (from its most recent appearance)
    state = {}
    for t in teams:
        sub = df[(df.home_team == t) | (df.away_team == t)]
        r = sub.iloc[-1]
        home = r.home_team == t
        state[t] = {
            "form": r.form_home if home else r.form_away,
            "gf": r.gf_home if home else r.gf_away,
            "ga": r.ga_home if home else r.ga_away,
            "exp": r.exp_home if home else r.exp_away,
            "conf": r.conf_home if home else r.conf_away,
        }

    def row(a, b):
        return pd.DataFrame([{
            "elo_diff": elo[a] - elo[b], "elo_home": elo[a], "elo_away": elo[b],
            "form_home": state[a]["form"], "form_away": state[b]["form"],
            "gf_home": state[a]["gf"], "gf_away": state[b]["gf"],
            "ga_home": state[a]["ga"], "ga_away": state[b]["ga"],
            "rest_home": 4, "rest_away": 4,
            "exp_home": state[a]["exp"], "exp_away": state[b]["exp"],
            "importance": 60, "neutral": 1,
            "conf_home": state[a]["conf"], "conf_away": state[b]["conf"],
        }])

    def p_advance(model, a, b):
        # symmetrise over slot to remove residual home bias on neutral ground
        def adv(x, y):
            if isinstance(model, PoissonModel):
                pr = model.predict_proba(row(x, y))[0]
            else:
                pr = model.predict_proba(row(x, y))[0]
            return pr[2] + 0.5 * pr[1]  # P(x win) + half draws
        return 0.5 * (adv(a, b) + (1 - adv(b, a)))

    out = {}
    N = 10000
    for name, model in fitted.items():
        rng = random.Random(SEED)
        pcache = {}
        def pw(a, b):
            key = (a, b)
            if key not in pcache:
                if name == "Elo baseline":
                    pcache[key] = win_probability(elo[a], elo[b])  # analytic, matches dashboard
                else:
                    pcache[key] = float(np.clip(p_advance(model, a, b), 0.01, 0.99))
            return pcache[key]
        counts = {t: 0 for t in teams}
        for _ in range(N):
            s1 = QF[0][0] if rng.random() < pw(*QF[0]) else QF[0][1]
            s2 = QF[1][0] if rng.random() < pw(*QF[1]) else QF[1][1]
            s3 = QF[2][0] if rng.random() < pw(*QF[2]) else QF[2][1]
            s4 = QF[3][0] if rng.random() < pw(*QF[3]) else QF[3][1]
            f1 = s1 if rng.random() < pw(s1, s2) else s2
            f2 = s3 if rng.random() < pw(s3, s4) else s4
            champ = f1 if rng.random() < pw(f1, f2) else f2
            counts[champ] += 1
        out[name] = sorted(
            [{"team": t, "p_champion": round(100 * c / N, 1)} for t, c in counts.items()],
            key=lambda d: -d["p_champion"])
    return out


if __name__ == "__main__":
    main()
