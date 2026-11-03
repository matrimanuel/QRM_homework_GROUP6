# ==============================================================
# 03_garch_portfolio.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 3: Gaussian GARCH(1,1) on each asset, random portfolio weights,
#             portfolio PORT, GARCH on PORT, conditional variance plots
# Input : logreturns_brazil.csv
# Output: outputs_q3/ (tables, figures, and files used by later questions)
# Requires: pandas numpy scipy matplotlib arch
# ==============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model

warnings.filterwarnings("ignore")

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs_q3")
os.makedirs(OUT, exist_ok=True)

for cand in [os.path.join(BASE_DIR, "logreturns_brazil.csv"),
             os.path.join(BASE_DIR, "data_clean", "logreturns_brazil.csv")]:
    if os.path.exists(cand):
        rets = pd.read_csv(cand, index_col=0, parse_dates=True)
        break
else:
    raise FileNotFoundError("logreturns_brazil.csv not found")
rets.columns = [c.replace(".SA", "") for c in rets.columns]

SPLIT = "2025-09-30"                      # last in-sample day; test = 2025-10-01 ... 2026-09-30
SEED = 123                                # fixed seed -> weights never change
SCALE = 100                               # GARCH is estimated on returns in % (numerical stability)
N = rets.shape[1]

# ---------- 1. Portfolio weights from a positive-support density ----------
# Draw N values from a log-normal (support (0, inf)) and normalize to sum to 1.
rng = np.random.default_rng(SEED)
raw = rng.lognormal(mean=0.0, sigma=1.0, size=N)
weights = pd.Series(raw / raw.sum(), index=rets.columns, name="weight")
assert abs(weights.sum() - 1) < 1e-12 and (weights > 0).all()
weights.to_csv(os.path.join(OUT, "table_weights.csv"), header=True)
print("PORTFOLIO WEIGHTS (seed = %d)\n%s\nsum = %.6f" % (SEED, weights.round(4).to_string(), weights.sum()))

# ---------- 2. Portfolio return PORT --------------------------------------
# Daily weighted average of the asset returns (as required by the assignment)
rets["PORT"] = rets[weights.index].dot(weights)

# In-sample / out-of-sample split
ins = rets.loc[:SPLIT]
oos = rets.loc[SPLIT:].iloc[1:]
print(f"\nIn-sample : {ins.index[0].date()} -> {ins.index[-1].date()}  (n={len(ins)})")
print(f"Out-sample: {oos.index[0].date()} -> {oos.index[-1].date()}  (n={len(oos)})")
rets.to_csv(os.path.join(OUT, "returns_with_port.csv"), date_format="%Y-%m-%d")

# ---------- 3. Gaussian GARCH(1,1) on every series (in-sample) -------------
# Model: r_t = mu + e_t, e_t = sigma_t z_t, z_t ~ N(0,1),
#        sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2
def fit_garch(series):
    am = arch_model(series * SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    return am.fit(disp="off")

fits, rows = {}, []
for c in rets.columns:
    res = fit_garch(ins[c])
    fits[c] = res
    p, se, pv = res.params, res.std_err, res.pvalues
    rows.append({"Asset": c,
                 "mu": p["mu"], "mu p": pv["mu"],
                 "omega": p["omega"], "omega p": pv["omega"],
                 "alpha": p["alpha[1]"], "alpha se": se["alpha[1]"], "alpha p": pv["alpha[1]"],
                 "beta": p["beta[1]"], "beta se": se["beta[1]"], "beta p": pv["beta[1]"],
                 "alpha+beta": p["alpha[1]"] + p["beta[1]"],
                 "Uncond. ann. vol (%)": np.sqrt(p["omega"] / (1 - p["alpha[1]"] - p["beta[1]"]) * 250)
                                         if p["alpha[1]"] + p["beta[1]"] < 1 else np.nan,
                 "Log-lik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
                 "Converged": res.convergence_flag == 0})
par = pd.DataFrame(rows).set_index("Asset")
par.to_csv(os.path.join(OUT, "table_garch_parameters.csv"), float_format="%.6g")
print("\nGARCH(1,1) GAUSSIAN PARAMETERS (returns in %)\n", par.round(4).to_string())
sig = par[["mu p", "omega p", "alpha p", "beta p"]] < 0.05
print(f"\nParameters significant at 5% (count out of {len(par)}):", sig.sum().to_dict())

# Full summary of the portfolio model
print("\n", fits["PORT"].summary())

# ---------- 4. Save objects needed in later questions ----------------------
# conditional volatility (%) and standardized residuals Z (in-sample)
condvol = pd.DataFrame({c: fits[c].conditional_volatility for c in rets.columns})
Z = pd.DataFrame({c: fits[c].std_resid for c in rets.columns})
condvol.to_csv(os.path.join(OUT, "insample_conditional_vol_pct.csv"), date_format="%Y-%m-%d")
Z.to_csv(os.path.join(OUT, "insample_std_residuals.csv"), date_format="%Y-%m-%d")
par[["mu", "omega", "alpha", "beta"]].to_csv(os.path.join(OUT, "garch_params_for_forecast.csv"))

# ---------- 5. Plot conditional variance -----------------------------------
show = [c for c in ["PETR4", "ITUB4", "VALE3", "PORT"] if c in rets.columns]
fig, ax = plt.subplots(len(show), 1, figsize=(11, 2.6 * len(show)), sharex=True)
for a, c in zip(ax, show):
    a.plot(condvol.index, condvol[c] ** 2, lw=0.9, color="C3" if c == "PORT" else "C0")
    a.set_ylabel(f"{c}\ncond. variance (%$^2$)")
ax[0].set_title("Conditional variance from Gaussian GARCH(1,1), in-sample")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_conditional_variance.png"), dpi=150)
plt.close()

# Portfolio vs average of asset conditional variances (diversification effect)
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(condvol.index, (condvol.drop(columns="PORT") ** 2).mean(axis=1), lw=0.9, label="average asset")
ax.plot(condvol.index, condvol["PORT"] ** 2, lw=0.9, label="PORT")
ax.set_title("Conditional variance: average asset vs portfolio"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_condvar_port_vs_assets.png"), dpi=150)
plt.close()

print(f"\nDone. Files saved in: {OUT}")
