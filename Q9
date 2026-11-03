# ==============================================================
# 09_expected_shortfall.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 9 (bonus): Expected Shortfall of the returns, plotted with returns and VaR
#   ES Case 1: GARCH with Gaussian innovations
#   ES Case 5: GARCH + GPD tail of the standardized losses
# Input : outputs_q3/returns_with_port.csv, outputs_q4/forecast_*_normal.csv, outputs_q4/VaR_case1_*.csv,
#         outputs_q6/evt_quantiles_gpd.csv, outputs_q7/VaR_case5_gpd_*.csv
# Output: outputs_q9/
# Requires: pandas numpy scipy matplotlib
# ==============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs_q9")
os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.95, 0.99]
FOCUS = ["PETR4", "PORT"]

def rd(*p):
    return pd.read_csv(os.path.join(BASE_DIR, *p), index_col=0, parse_dates=True)

rets = rd("outputs_q3", "returns_with_port.csv")
mu = rd("outputs_q4", "forecast_mu_normal.csv")
sg = rd("outputs_q4", "forecast_sigma_normal.csv")
gpd = pd.read_csv(os.path.join(BASE_DIR, "outputs_q6", "evt_quantiles_gpd.csv"), index_col=0)
cols = list(mu.columns)
real = rets.loc[mu.index, cols]
n = len(real)

# ---------- 1. Expected Shortfall of the returns ---------------------------
# r_t = mu_t + sigma_t Z_t  =>  loss L_t = -mu_t + sigma_t (-Z_t), hence
#     ES_t(alpha) = -mu_t + sigma_t * ES_Z(alpha)
# with ES_Z the expected shortfall of the standardized loss -Z.
#  Case 1 (Normal): ES_Z(alpha) = phi(Phi^{-1}(alpha)) / (1 - alpha)
#  Case 5 (GPD)   : ES_Z(alpha) = VaR_Z/(1-xi) + (beta - xi u)/(1-xi)   (computed in question 6)
def es_normal(a):
    return stats.norm.pdf(stats.norm.ppf(a)) / (1 - a)

ES, VAR = {}, {}
for a in ALPHAS:
    k = int(a * 100)
    ES[("case1", a)] = pd.DataFrame({c: -mu[c] + sg[c] * es_normal(a) for c in cols}, index=mu.index)
    ES[("case5", a)] = pd.DataFrame({c: -mu[c] + sg[c] * gpd.loc[c, f"es_gpd_{k}"] for c in cols}, index=mu.index)
    VAR[("case1", a)] = rd("outputs_q4", f"VaR_case1_normal_{k}.csv")[cols]
    VAR[("case5", a)] = rd("outputs_q7", f"VaR_case5_gpd_{k}.csv")[cols]
    ES[("case1", a)].to_csv(os.path.join(OUT, f"ES_case1_normal_{k}.csv"), date_format="%Y-%m-%d")
    ES[("case5", a)].to_csv(os.path.join(OUT, f"ES_case5_gpd_{k}.csv"), date_format="%Y-%m-%d")
LABEL = {"case1": "Case 1: Normal", "case5": "Case 5: GPD"}
COLOR = {"case1": "C0", "case5": "C3"}

# ---------- 2. Simple backtest of the ES -----------------------------------
# On the days when the loss exceeds the VaR, the realised loss should on average equal the
# predicted ES. McNeil & Frey (2000): the standardized excess (L_t - ES_t)/sigma_t has mean zero.
rows = []
for a in ALPHAS:
    for k in ["case1", "case5"]:
        for c in cols:
            L = -real[c]
            viol = (L > VAR[(k, a)][c]).values
            nv = int(viol.sum())
            es_v = ES[(k, a)][c][viol]
            row = {"alpha": a, "Case": k, "Asset": c, "Violations": nv,
                   "Mean ES (%)": 100 * ES[(k, a)][c].mean(),
                   "Mean VaR (%)": 100 * VAR[(k, a)][c].mean(),
                   "ES/VaR": ES[(k, a)][c].mean() / VAR[(k, a)][c].mean()}
            if nv >= 1:
                row["Mean realised loss on violation days (%)"] = 100 * L[viol].mean()
                row["Mean predicted ES on those days (%)"] = 100 * es_v.mean()
                row["Days with loss > ES"] = int((L.values > ES[(k, a)][c].values).sum())
            if nv >= 3:                                   # McNeil-Frey test needs a few violations
                r = ((L[viol] - es_v) / sg[c][viol]).values
                t = stats.ttest_1samp(r, 0.0, alternative="greater")   # H1: ES underestimates the tail
                row["McNeil-Frey mean"] = r.mean(); row["McNeil-Frey p (one-sided)"] = t.pvalue
            rows.append(row)
es_tab = pd.DataFrame(rows)
es_tab.to_csv(os.path.join(OUT, "table_es_backtest_all.csv"), index=False, float_format="%.4g")
for a in ALPHAS:
    s = es_tab[es_tab.alpha == a]
    foc = s[s.Asset.isin(FOCUS)].drop(columns="alpha")
    foc.to_csv(os.path.join(OUT, f"table_es_focus_{int(a*100)}.csv"), index=False, float_format="%.4g")
    print(f"\n=== {int(a*100)}%: ES for the selected series ===\n", foc.round(3).to_string(index=False))
    g = s.groupby("Case")
    summ = pd.DataFrame({"Avg VaR (%)": g["Mean VaR (%)"].mean(), "Avg ES (%)": g["Mean ES (%)"].mean(),
                         "Avg ES/VaR": g["ES/VaR"].mean(), "Total violations": g["Violations"].sum(),
                         "Mean loss on viol. days (%)": g["Mean realised loss on violation days (%)"].mean(),
                         "Mean ES on viol. days (%)": g["Mean predicted ES on those days (%)"].mean()})
    summ.to_csv(os.path.join(OUT, f"table_es_summary_{int(a*100)}.csv"), float_format="%.4g")
    print(f"\n=== {int(a*100)}%: summary over the {len(cols)} series ===\n", summ.round(3).to_string())

# ---------- 3. Figures: returns, VaR and ES --------------------------------
for a in ALPHAS:
    fig, ax = plt.subplots(len(FOCUS), 1, figsize=(12, 4.3 * len(FOCUS)), sharex=True)
    for axx, c in zip(np.atleast_1d(ax), FOCUS):
        axx.plot(real.index, real[c] * 100, color="grey", lw=0.7, label="return")
        for k in ["case1", "case5"]:
            axx.plot(real.index, -VAR[(k, a)][c] * 100, color=COLOR[k], lw=1.2, label=f"VaR {LABEL[k]}")
            axx.plot(real.index, -ES[(k, a)][c] * 100, color=COLOR[k], lw=1.2, ls="--", label=f"ES {LABEL[k]}")
        v = real[c] < -VAR[("case5", a)][c]
        axx.scatter(real.index[v], real[c][v] * 100, color="k", s=30, zorder=5, label="return below VaR (Case 5)")
        axx.set_ylabel(f"{c} (%)")
        axx.set_title(f"{c}: returns, {int(a*100)}% VaR (solid) and {int(a*100)}% Expected Shortfall (dashed)")
    np.atleast_1d(ax)[0].legend(ncol=3, fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_es_var_{int(a*100)}.png"), dpi=150); plt.close()

# ES/VaR ratio over the assets (heavy tails -> larger ratio)
piv = es_tab[es_tab.alpha == 0.99].pivot(index="Asset", columns="Case", values="ES/VaR").loc[cols]
fig, ax = plt.subplots(figsize=(11, 3.8)); xs = np.arange(len(cols)); w = 0.4
ax.bar(xs - w / 2, piv["case1"], w, color=COLOR["case1"], label=LABEL["case1"])
ax.bar(xs + w / 2, piv["case5"], w, color=COLOR["case5"], label=LABEL["case5"])
ax.set_xticks(xs); ax.set_xticklabels(cols, rotation=60); ax.set_ylabel("mean ES / mean VaR (99%)"); ax.legend()
ax.set_title("ES-to-VaR ratio at 99%")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_es_var_ratio_99.png"), dpi=150); plt.close()

print(f"\nDone. Files saved in: {OUT}")
