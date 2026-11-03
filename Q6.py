# ==============================================================
# 06_gpd.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 6: GPD (peaks over threshold) on the in-sample standardized losses,
#             fit quality, VaR of Z (and ES of Z, reused in the bonus question)
# Input : outputs_q3/insample_std_residuals.csv ; outputs_q5/table_var_Z.csv (optional, for comparison)
# Output: outputs_q6/
# Requires: pandas numpy scipy matplotlib
# ==============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs_q6")
os.makedirs(OUT, exist_ok=True)

THRESH_Q = 0.90              # threshold u = 90% quantile of the losses (both 95% and 99% VaR lie beyond it)
SENS_Q = [0.85, 0.90, 0.925, 0.95]   # thresholds for the sensitivity analysis
ALPHAS = [0.95, 0.99]
B_BOOT = 200
SEED = 123
rng = np.random.default_rng(SEED)

Z = pd.read_csv(os.path.join(BASE_DIR, "outputs_q3", "insample_std_residuals.csv"),
                index_col=0, parse_dates=True)
cols = list(Z.columns)
L = -Z                                       # losses
n = len(L)
print(f"In-sample losses: n={n}, series={len(cols)}, threshold quantile={THRESH_Q}")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()

# ---------- 1. Helper functions -------------------------------------------
def fit_gpd(exc):
    """ML fit of the GPD(xi, beta) to the exceedances (location fixed at 0)."""
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    return xi, beta

def gpd_negll(p, x):
    xi, beta = p
    if beta <= 0:
        return np.inf
    return -np.sum(stats.genpareto.logpdf(x, c=xi, loc=0, scale=beta))

def num_hessian(f, p, h=1e-4):
    p = np.asarray(p, float); k = len(p); H = np.zeros((k, k))
    step = h * np.maximum(np.abs(p), 1.0)
    for i in range(k):
        for j in range(k):
            pp = p.copy(); pm = p.copy(); mp = p.copy(); mm = p.copy()
            pp[i] += step[i]; pp[j] += step[j]; pm[i] += step[i]; pm[j] -= step[j]
            mp[i] -= step[i]; mp[j] += step[j]; mm[i] -= step[i]; mm[j] -= step[j]
            H[i, j] = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * step[i] * step[j])
    return H

def pot_var(u, xi, beta, nu, n, alpha):
    """VaR_alpha = u + beta/xi * ( (n/Nu * (1-alpha))^(-xi) - 1 )   (tail estimator of McNeil-Frey)."""
    r = (n / nu) * (1 - alpha)
    return u + beta * np.log(1 / r) if abs(xi) < 1e-8 else u + beta / xi * (r ** (-xi) - 1)

def pot_es(var, u, xi, beta):
    """ES_alpha = VaR/(1-xi) + (beta - xi*u)/(1-xi), valid for xi < 1."""
    return var / (1 - xi) + (beta - xi * u) / (1 - xi) if xi < 1 else np.nan

# ---------- 2. GPD fit at the main threshold -------------------------------
fits, rows = {}, []
for c in cols:
    x = L[c].values
    u = np.quantile(x, THRESH_Q)
    exc = x[x > u] - u
    nu = len(exc)
    xi, beta = fit_gpd(exc)
    H = num_hessian(lambda p: gpd_negll(p, exc), [xi, beta])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    ll = -gpd_negll([xi, beta], exc)
    ll_exp = np.sum(stats.expon.logpdf(exc, scale=exc.mean()))       # restricted model xi = 0
    lr = 2 * (ll - ll_exp)
    fits[c] = (u, xi, beta, nu)
    rows.append({"Asset": c, "u": u, "N exceed.": nu, "xi": xi, "se(xi)": se[0],
                 "beta": beta, "se(beta)": se[1], "Log-lik": ll,
                 "LR test xi=0 (p)": stats.chi2.sf(max(lr, 0), 1)})
gpd = pd.DataFrame(rows).set_index("Asset")
gpd.to_csv(os.path.join(OUT, "table_gpd_parameters.csv"), float_format="%.4g")
print("\nGPD PARAMETERS (exceedances of the standardized losses)\n", gpd.round(3).to_string())

# ---------- 3. Quality of the fit ------------------------------------------
def ks_ad(exc, xi, beta):
    u_ = np.sort(stats.genpareto.cdf(exc, c=xi, loc=0, scale=beta))
    u_ = np.clip(u_, 1e-12, 1 - 1e-12); k = len(u_); i = np.arange(1, k + 1)
    ks = max(np.max(i / k - u_), np.max(u_ - (i - 1) / k))
    ad = -k - np.mean((2 * i - 1) * (np.log(u_) + np.log(1 - u_[::-1])))
    return ks, ad

rows = []
for c in cols:
    u, xi, beta, nu = fits[c]
    exc = L[c].values[L[c].values > u] - u
    ks0, ad0 = ks_ad(exc, xi, beta)
    ks_b, ad_b = [], []
    for _ in range(B_BOOT):                                   # parametric bootstrap (parameters are estimated)
        xb = stats.genpareto.rvs(c=xi, loc=0, scale=beta, size=nu, random_state=rng)
        try:
            xb_, bb_ = fit_gpd(xb)
        except Exception:
            continue
        a, b = ks_ad(xb, xb_, bb_); ks_b.append(a); ad_b.append(b)
    rows.append({"Asset": c, "KS stat": ks0, "KS p (bootstrap)": np.mean(np.array(ks_b) >= ks0),
                 "AD stat": ad0, "AD p (bootstrap)": np.mean(np.array(ad_b) >= ad0)})
fitq = pd.DataFrame(rows).set_index("Asset")
fitq.to_csv(os.path.join(OUT, "table_gpd_fit_quality.csv"), float_format="%.4g")
print("\nQUALITY OF THE GPD FIT\n", fitq.round(3).to_string())
print("Rejections at 5% (KS, AD):", int((fitq["KS p (bootstrap)"] < .05).sum()),
      int((fitq["AD p (bootstrap)"] < .05).sum()), "of", len(fitq))

# Q-Q plots: empirical exceedances vs fitted GPD quantiles
fig, ax = plt.subplots(4, 4, figsize=(14, 12))
for a, c in zip(ax.ravel(), cols):
    u, xi, beta, nu = fits[c]
    exc = np.sort(L[c].values[L[c].values > u] - u)
    p = (np.arange(1, nu + 1) - 0.5) / nu
    q = stats.genpareto.ppf(p, c=xi, loc=0, scale=beta)
    a.scatter(q, exc, s=8)
    a.plot([0, max(q.max(), exc.max())], [0, max(q.max(), exc.max())], "r", lw=1)
    a.set_title(f"{c}  (xi={xi:.2f})")
for a in ax.ravel()[len(cols):]:
    a.axis("off")
plt.suptitle("GPD Q-Q plots of exceedances: fitted quantiles (x) vs empirical (y)", y=1.0)
save_fig("fig_gpd_qq.png")

# ---------- 4. Threshold choice: mean-excess plot and stability of xi -------
sel = [c for c in ["PETR4", "VALE3", "ITUB4", "PORT"] if c in cols]
fig, ax = plt.subplots(2, len(sel), figsize=(14, 6))
qs = np.arange(0.80, 0.985, 0.005)
for j, c in enumerate(sel):
    x = L[c].values
    us = np.quantile(x, qs)
    ax[0, j].plot(us, [np.mean(x[x > u_] - u_) for u_ in us])               # mean excess function e(u)
    ax[0, j].axvline(fits[c][0], color="r", ls="--")
    ax[0, j].set_title(f"{c}: mean excess e(u)")
    xis = []
    for u_ in us:
        try:
            xis.append(fit_gpd(x[x > u_] - u_)[0])
        except Exception:
            xis.append(np.nan)
    ax[1, j].plot(qs, xis); ax[1, j].axvline(THRESH_Q, color="r", ls="--")
    ax[1, j].set_title(f"{c}: xi vs threshold quantile")
save_fig("fig_gpd_threshold_choice.png")

# Sensitivity of xi and the 99% VaR to the threshold
rows = []
for c in cols:
    x = L[c].values; row = {"Asset": c}
    for q in SENS_Q:
        u_ = np.quantile(x, q); e_ = x[x > u_] - u_
        xi_, be_ = fit_gpd(e_)
        row[f"xi q={q}"] = xi_
        row[f"VaR99 q={q}"] = pot_var(u_, xi_, be_, len(e_), n, 0.99)
    rows.append(row)
sens = pd.DataFrame(rows).set_index("Asset")
sens.to_csv(os.path.join(OUT, "table_gpd_threshold_sensitivity.csv"), float_format="%.4f")
print("\nSENSITIVITY TO THE THRESHOLD\n", sens.round(3).to_string())

# ---------- 5. VaR and ES of Z ---------------------------------------------
# Clustering (extremal index) does not change the POT tail estimator of the marginal
# quantile: it only affects the uncertainty of the estimates. So VaR under independence is the VaR.
rows = []
for c in cols:
    u, xi, beta, nu = fits[c]
    row = {"Asset": c}
    for a in ALPHAS:
        v = pot_var(u, xi, beta, nu, n, a)
        row[f"VaR GPD {a}"] = v
        row[f"ES GPD {a}"] = pot_es(v, u, xi, beta)
        row[f"Empirical {a}"] = np.quantile(L[c].values, a)
        row[f"Normal {a}"] = stats.norm.ppf(a)
    rows.append(row)
vz = pd.DataFrame(rows).set_index("Asset")

# add the GEV results of question 5 if available (comparison)
f5 = os.path.join(BASE_DIR, "outputs_q5", "table_var_Z.csv")
if os.path.exists(f5):
    g = pd.read_csv(f5, index_col=0)
    for a in ALPHAS:
        vz[f"GEV indep {a}"] = g[f"GEV indep {a}"]
        vz[f"GEV theta {a}"] = g[f"GEV theta {a}"]
vz.to_csv(os.path.join(OUT, "table_var_Z_gpd.csv"), float_format="%.4f")
print("\nVaR / ES OF Z (standardized loss)\n", vz.round(3).to_string())

# quantiles used in question 7 (Case 5) and the bonus question
evt = pd.DataFrame({f"q_gpd_{int(a*100)}": vz[f"VaR GPD {a}"] for a in ALPHAS})
for a in ALPHAS:
    evt[f"es_gpd_{int(a*100)}"] = vz[f"ES GPD {a}"]
evt.to_csv(os.path.join(OUT, "evt_quantiles_gpd.csv"), float_format="%.6f")

# Figure: tail of the loss distribution with the fitted GPD (selected series)
fig, ax = plt.subplots(1, len(sel), figsize=(14, 3.5))
for a, c in zip(ax, sel):
    u, xi, beta, nu = fits[c]
    x = np.sort(L[c].values); x = x[x > u]
    xs = np.linspace(u, x.max() * 1.15, 200)
    a.semilogy(x, np.array([(x >= v).sum() / n for v in x]), "o", ms=3, label="empirical")
    a.semilogy(xs, nu / n * stats.genpareto.sf(xs - u, c=xi, loc=0, scale=beta), "r", label="GPD tail")
    a.set_title(c); a.set_xlabel("loss of Z"); a.legend()
save_fig("fig_gpd_tail.png")

# Figure: 99% VaR of Z by method
meth = [("Normal 0.99", "Normal"), ("Empirical 0.99", "Empirical"), ("VaR GPD 0.99", "GPD")]
if "GEV indep 0.99" in vz:
    meth.insert(2, ("GEV indep 0.99", "GEV"))
fig, ax = plt.subplots(figsize=(12, 4))
w = 0.8 / len(meth); xs = np.arange(len(cols))
for k, (col, lab) in enumerate(meth):
    ax.bar(xs + (k - (len(meth) - 1) / 2) * w, vz[col], w, label=lab)
ax.set_xticks(xs); ax.set_xticklabels(cols, rotation=60); ax.set_ylabel("99% VaR of Z"); ax.legend(ncol=4)
ax.set_title("99% VaR of the standardized losses by method")
save_fig("fig_var_Z_99_methods.png")

print(f"\nDone. Files saved in: {OUT}")
