# ==============================================================
# 04_var_garch.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 4: one-step-ahead GARCH forecasts for the last year,
#             VaR Case 1 (Normal), normality test of in-sample Z,
#             alternative density, VaR Case 2
# Input : outputs_q3/returns_with_port.csv, outputs_q3/insample_std_residuals.csv
#         (produced by 03_garch_portfolio.py)
# Output: outputs_q4/
# Requires: pandas numpy scipy matplotlib arch
# ==============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model

warnings.filterwarnings("ignore")

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
Q3 = os.path.join(BASE_DIR, "outputs_q3")
OUT = os.path.join(BASE_DIR, "outputs_q4")
os.makedirs(OUT, exist_ok=True)

SPLIT = "2025-09-30"            # last in-sample day
SCALE = 100                     # GARCH estimated on returns in %
ALPHAS = [0.95, 0.99]           # VaR confidence levels
REFIT_EVERY = 20                # re-estimate the GARCH every h days (set to 10**6 for fixed parameters)
CANDIDATES = ["normal", "t", "skewt", "ged"]   # densities compared for the innovations

rets = pd.read_csv(os.path.join(Q3, "returns_with_port.csv"), index_col=0, parse_dates=True)
Z = pd.read_csv(os.path.join(Q3, "insample_std_residuals.csv"), index_col=0, parse_dates=True)
cols = list(rets.columns)                      # 15 assets + PORT
n_ins = int((rets.index <= SPLIT).sum())
n_oos = len(rets) - n_ins
oos_idx = rets.index[n_ins:]
print(f"In-sample n={n_ins}, out-of-sample n={n_oos} ({oos_idx[0].date()} -> {oos_idx[-1].date()})")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()

# ---------- 1. Is Z normal? (in-sample standardized residuals) -------------
rows = {}
for c in cols:
    z = Z[c].dropna()
    ad = stats.anderson(z, dist="norm")
    nu = stats.t.fit(z, floc=0)[0]                       # ML estimate of the t degrees of freedom on Z
    rows[c] = {"Skewness": stats.skew(z), "Excess kurtosis": stats.kurtosis(z),
               "JB p-value": stats.jarque_bera(z).pvalue,
               "Anderson-Darling": ad.statistic, "AD 1% crit.": ad.critical_values[-1],
               "KS p-value (vs N(0,1))": stats.kstest(z, "norm").pvalue,
               "t df fitted on Z": nu}
normZ = pd.DataFrame(rows).T
normZ.to_csv(os.path.join(OUT, "table_normality_Z.csv"), float_format="%.4g")
print("\nNORMALITY OF IN-SAMPLE Z\n", normZ.round(3).to_string())
print(f"Normality rejected (JB, 5%): {(normZ['JB p-value'] < 0.05).sum()} of {len(normZ)}")

# Q-Q plots of Z against the normal
fig, ax = plt.subplots(4, 4, figsize=(14, 12))
for a, c in zip(ax.ravel(), cols):
    stats.probplot(Z[c].dropna(), dist="norm", plot=a)
    a.set_title(c); a.set_xlabel(""); a.set_ylabel("")
for a in ax.ravel()[len(cols):]:
    a.axis("off")
plt.suptitle("Q-Q plots of in-sample standardized residuals Z vs Normal", y=1.0)
save_fig("fig_qq_Z_normal.png")

# ---------- 2. Choice of the alternative density ---------------------------
# Re-estimate the in-sample GARCH(1,1) with each candidate density, compare AIC/BIC
def fit(series, dist):
    am = arch_model(series * SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    return am.fit(disp="off")

aic, bic, dpar = {}, {}, {}
for c in cols:
    aic[c], bic[c], dpar[c] = {}, {}, {}
    for d in CANDIDATES:
        res = fit(rets[c].iloc[:n_ins], d)
        aic[c][d], bic[c][d] = res.aic, res.bic
        nd = len(res.model.distribution.parameter_names())
        dpar[c][d] = list(res.params.values[-nd:]) if nd else []
aic, bic = pd.DataFrame(aic).T, pd.DataFrame(bic).T
aic.to_csv(os.path.join(OUT, "table_aic_by_density.csv"), float_format="%.2f")
bic.to_csv(os.path.join(OUT, "table_bic_by_density.csv"), float_format="%.2f")
best = bic.idxmin(axis=1)
print("\nBIC by density\n", bic.round(1).to_string())
print("\nDensity with the lowest BIC, count over assets:\n", best.value_counts().to_string())
print("Density with the lowest AIC, count over assets:\n", aic.idxmin(axis=1).value_counts().to_string())

# Decision rule: the density that minimises BIC for most series (ties -> fewer parameters)
ALT = best.value_counts().index[0]
print(f"\n==> Alternative density chosen: {ALT}")
pd.DataFrame({c: dpar[c][ALT] for c in cols}).T.to_csv(
    os.path.join(OUT, f"table_params_{ALT}_insample.csv"), float_format="%.4g")

# ---------- 3. Rolling one-step-ahead forecasts ----------------------------
def rolling_forecast(series, dist):
    """Return mu_t, sigma_t (in %) and the lower-tail standardized quantiles for the OOS days.
    Parameters are re-estimated every REFIT_EVERY days on an expanding window;
    between refits the variance recursion is updated with the realised returns."""
    r = series * SCALE
    mu_f, s2_f = np.zeros(n_oos), np.zeros(n_oos)
    q_f = {a: np.zeros(n_oos) for a in ALPHAS}
    for s in range(n_ins, len(r), REFIT_EVERY):
        am = arch_model(r.iloc[:s], mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
        res = am.fit(disp="off")
        mu, om, al, be = (res.params[k] for k in ["mu", "omega", "alpha[1]", "beta[1]"])
        nd = len(am.distribution.parameter_names())
        dp = res.params.values[-nd:] if nd else None
        for a in ALPHAS:                                   # lower-tail quantile of the standardized density
            qa = am.distribution.ppf(np.array([1 - a]), dp)
            for t in range(s, min(s + REFIT_EVERY, len(r))):
                q_f[a][t - n_ins] = float(np.ravel(qa)[0])
        # variance for day s uses the last in-sample variance and residual
        sig2 = om + al * res.resid.iloc[-1] ** 2 + be * res.conditional_volatility.iloc[-1] ** 2
        for t in range(s, min(s + REFIT_EVERY, len(r))):
            mu_f[t - n_ins], s2_f[t - n_ins] = mu, sig2
            sig2 = om + al * (r.iloc[t] - mu) ** 2 + be * sig2   # update with realised return
    return mu_f, np.sqrt(s2_f), q_f

store = {d: {} for d in ["normal", ALT]}
for d in store:
    for c in cols:
        store[d][c] = rolling_forecast(rets[c], d)
    print(f"Forecasts done for density: {d}")

# GARCH forecasts (Normal model) saved for the EVT-based VaR of Q7
mu_n = pd.DataFrame({c: store["normal"][c][0] for c in cols}, index=oos_idx) / SCALE
sg_n = pd.DataFrame({c: store["normal"][c][1] for c in cols}, index=oos_idx) / SCALE
mu_n.to_csv(os.path.join(OUT, "forecast_mu_normal.csv"), date_format="%Y-%m-%d")
sg_n.to_csv(os.path.join(OUT, "forecast_sigma_normal.csv"), date_format="%Y-%m-%d")

# ---------- 4. VaR (loss, positive number, in return units) ----------------
# VaR_t = -(mu_t + sigma_t * q_{1-alpha}),  q = lower-tail quantile of the standardized density
def var_table(d, a):
    return pd.DataFrame({c: -(store[d][c][0] + store[d][c][1] * store[d][c][2][a]) / SCALE
                         for c in cols}, index=oos_idx)

VaR1 = {a: var_table("normal", a) for a in ALPHAS}     # Case 1: Normal
VaR2 = {a: var_table(ALT, a) for a in ALPHAS}          # Case 2: alternative density
for a in ALPHAS:
    VaR1[a].to_csv(os.path.join(OUT, f"VaR_case1_normal_{int(a*100)}.csv"), date_format="%Y-%m-%d")
    VaR2[a].to_csv(os.path.join(OUT, f"VaR_case2_{ALT}_{int(a*100)}.csv"), date_format="%Y-%m-%d")

# ---------- 5. Summary: level of VaR and number of violations --------------
real = rets.iloc[n_ins:]
summ = []
for a in ALPHAS:
    for c in cols:
        v1, v2 = VaR1[a][c], VaR2[a][c]
        summ.append({"alpha": a, "Asset": c, "Expected violations": (1 - a) * n_oos,
                     "Violations Case 1 (Normal)": int((real[c] < -v1).sum()),
                     f"Violations Case 2 ({ALT})": int((real[c] < -v2).sum()),
                     "Mean VaR Case 1 (%)": v1.mean() * 100, "Mean VaR Case 2 (%)": v2.mean() * 100,
                     "Ratio VaR2/VaR1": v2.mean() / v1.mean()})
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(OUT, "table_var_cases_summary.csv"), index=False, float_format="%.4g")
for a in ALPHAS:
    print(f"\nVaR at {int(a*100)}%  (a violation = return below -VaR)\n",
          summ[summ.alpha == a].drop(columns="alpha").round(3).to_string(index=False))

# ---------- 6. Figures -----------------------------------------------------
for c in [x for x in ["PETR4", "PORT"] if x in cols]:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(real.index, real[c] * 100, lw=0.7, color="grey", label="return")
    ax.plot(real.index, -VaR1[0.99][c] * 100, color="C0", lw=1.3, label="VaR 99% Case 1 (Normal)")
    ax.plot(real.index, -VaR2[0.99][c] * 100, color="C3", lw=1.3, label=f"VaR 99% Case 2 ({ALT})")
    v = real[c] < -VaR1[0.99][c]
    ax.scatter(real.index[v], real[c][v] * 100, color="C0", s=25, zorder=3)
    ax.set_ylabel("%"); ax.set_title(f"{c}: out-of-sample returns and 99% VaR"); ax.legend(ncol=3)
    save_fig(f"fig_var_{c}.png")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["Normal", ALT], [VaR1[0.99].mean().mean() * 100, VaR2[0.99].mean().mean() * 100], color=["C0", "C3"])
ax.set_ylabel("average 99% VaR over assets (%)"); ax.set_title("Effect of the density on the VaR")
save_fig("fig_var_level_by_density.png")

print(f"\nDone. Files saved in: {OUT}")
