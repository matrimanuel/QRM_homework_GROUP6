# ==============================================================
# 07_var_evt.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 7: VaR of the returns = GARCH forecasts + EVT quantiles of Z
#   Case 3: GEV, independence      Case 4: GEV with extremal index      Case 5: GPD
# Input : outputs_q3/returns_with_port.csv
#         outputs_q4/forecast_mu_normal.csv, forecast_sigma_normal.csv (Gaussian GARCH forecasts)
#         outputs_q5/evt_quantiles_gev.csv, outputs_q6/evt_quantiles_gpd.csv (in-sample EVT quantiles)
#         outputs_q4/VaR_case1_*.csv, VaR_case2_*.csv (optional, for the comparison table)
# Output: outputs_q7/
# Requires: pandas numpy matplotlib
# ==============================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs_q7")
os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.95, 0.99]

def rd(*p):
    return pd.read_csv(os.path.join(BASE_DIR, *p), index_col=0, parse_dates=True)

rets = rd("outputs_q3", "returns_with_port.csv")
mu = rd("outputs_q4", "forecast_mu_normal.csv")          # one-step-ahead mean forecast (decimal units)
sg = rd("outputs_q4", "forecast_sigma_normal.csv")       # one-step-ahead volatility forecast (decimal units)
gev = pd.read_csv(os.path.join(BASE_DIR, "outputs_q5", "evt_quantiles_gev.csv"), index_col=0)
gpd = pd.read_csv(os.path.join(BASE_DIR, "outputs_q6", "evt_quantiles_gpd.csv"), index_col=0)
cols = list(mu.columns)
real = rets.loc[mu.index, cols]
n_oos = len(real)
print(f"Out-of-sample: {mu.index[0].date()} -> {mu.index[-1].date()} (n={n_oos}), series={len(cols)}")

# ---------- 1. VaR from GARCH forecasts and the EVT quantile of Z ----------
# Model: r_t = mu_t + sigma_t * Z_t. Loss L_t = -r_t = -mu_t + sigma_t * (-Z_t),
# and the EVT quantile q(alpha) is the alpha-quantile of the loss -Z. Hence
#     VaR_t(alpha) = -mu_t + sigma_t * q(alpha)         (positive number = loss, return units)
# A violation occurs when the return is below -VaR.
def var_evt(q):
    return pd.DataFrame({c: -mu[c] + sg[c] * q[c] for c in cols}, index=mu.index)

cases = {"case3_gev_indep": ("GEV, independence", "q_indep_{}", gev),
         "case4_gev_theta": ("GEV, extremal index", "q_theta_{}", gev),
         "case5_gpd": ("GPD, independence", "q_gpd_{}", gpd)}
VaR = {}
for key, (label, pat, src) in cases.items():
    for a in ALPHAS:
        q = src[pat.format(int(a * 100))]
        VaR[(key, a)] = var_evt(q)
        VaR[(key, a)].to_csv(os.path.join(OUT, f"VaR_{key}_{int(a*100)}.csv"), date_format="%Y-%m-%d")

# Cases 1 and 2 from question 4 (if available), to get one comparison table
for f in glob.glob(os.path.join(BASE_DIR, "outputs_q4", "VaR_case[12]_*.csv")):
    name = os.path.basename(f)[:-4]                           # e.g. VaR_case2_t_99
    parts = name.split("_")
    key = "case1_normal" if parts[1] == "case1" else f"case2_{parts[2]}"
    a = int(parts[-1]) / 100
    VaR[(key, a)] = pd.read_csv(f, index_col=0, parse_dates=True)[cols]

# ---------- 2. Summary: VaR level and violations ---------------------------
rows = []
for (key, a), v in sorted(VaR.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    for c in cols:
        rows.append({"Case": key, "alpha": a, "Asset": c,
                     "Mean VaR (%)": v[c].mean() * 100,
                     "Violations": int((real[c] < -v[c]).sum()),
                     "Expected": (1 - a) * n_oos})
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "table_var_evt_summary_long.csv"), index=False, float_format="%.4g")

for a in ALPHAS:
    s = summ[summ.alpha == a]
    wide_v = s.pivot(index="Asset", columns="Case", values="Violations").loc[cols]
    wide_m = s.pivot(index="Asset", columns="Case", values="Mean VaR (%)").loc[cols]
    wide_v.to_csv(os.path.join(OUT, f"table_violations_{int(a*100)}.csv"))
    wide_m.to_csv(os.path.join(OUT, f"table_mean_var_{int(a*100)}.csv"), float_format="%.4g")
    print(f"\nVIOLATIONS at {int(a*100)}% (expected {(1-a)*n_oos:.1f} of {n_oos} days)\n", wide_v.to_string())
    print(f"\nMEAN VaR (%) at {int(a*100)}%\n", wide_m.round(3).to_string())
    print("Total violations over the 16 series:", wide_v.sum().to_dict())

# ---------- 3. Figures -----------------------------------------------------
styles = {"case1_normal": ("C0", "Case 1 Normal"), "case3_gev_indep": ("C1", "Case 3 GEV indep."),
          "case4_gev_theta": ("C2", "Case 4 GEV ext. index"), "case5_gpd": ("C3", "Case 5 GPD")}
for c in [x for x in ["PETR4", "PORT"] if x in cols]:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(real.index, real[c] * 100, lw=0.7, color="grey", label="return")
    for key, (col, lab) in styles.items():
        if (key, 0.99) in VaR:
            ax.plot(real.index, -VaR[(key, 0.99)][c] * 100, lw=1.2, color=col, label=lab)
    ax.set_ylabel("%"); ax.set_title(f"{c}: returns and 99% VaR (GARCH + EVT)"); ax.legend(ncol=3, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_var_evt_{c}.png"), dpi=150); plt.close()

# average 99% VaR over the assets, by case
keys = [k for k in ["case1_normal", "case3_gev_indep", "case4_gev_theta", "case5_gpd"] if (k, 0.99) in VaR]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([styles[k][1] for k in keys], [VaR[(k, 0.99)].mean().mean() * 100 for k in keys],
       color=[styles[k][0] for k in keys])
ax.set_ylabel("average 99% VaR (%)"); ax.set_title("Average VaR over assets and days")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_mean_var_by_case.png"), dpi=150); plt.close()

print(f"\nDone. Files saved in: {OUT}")
