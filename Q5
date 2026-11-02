# ==============================================================
# 05_gev.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 5: GEV (block maxima) on the in-sample standardized losses,
#             fit quality, VaR of Z without / with the extremal index
# Input : outputs_q3/insample_std_residuals.csv  (standardized residuals Z)
# Output: outputs_q5/
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
OUT = os.path.join(BASE_DIR, "outputs_q5")
os.makedirs(OUT, exist_ok=True)

M = 21                       # block length in days (about one month)
ALPHAS = [0.95, 0.99]        # VaR levels
THETA_Q = 0.95               # threshold quantile for the extremal index
B_BOOT = 200                 # parametric bootstrap replications for the KS p-value
SEED = 123
rng = np.random.default_rng(SEED)

Z = pd.read_csv(os.path.join(BASE_DIR, "outputs_q3", "insample_std_residuals.csv"),
                index_col=0, parse_dates=True)
cols = list(Z.columns)                        # 15 assets + PORT
L = -Z                                        # losses: positive values = bad days
n = len(L)
print(f"In-sample losses: n={n}, series={len(cols)}, block length m={M}")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()

# ---------- 1. Block maxima -----------------------------------------------
def block_maxima(x, m):
    k = len(x) // m                           # complete blocks only (drop the first incomplete part)
    x = np.asarray(x)[len(x) - k * m:]
    return x.reshape(k, m).max(axis=1)

BM = {c: block_maxima(L[c].values, M) for c in cols}
nb = len(BM[cols[0]])
print(f"Number of block maxima per series: {nb}")

# ---------- 2. GEV fit by maximum likelihood (independence) ----------------
# scipy parametrisation: shape c = -xi.   xi > 0: heavy tail (Frechet), xi = 0: Gumbel
def gev_negll(p, x):
    xi, mu, sg = p
    if sg <= 0:
        return np.inf
    return -np.sum(stats.genextreme.logpdf(x, c=-xi, loc=mu, scale=sg))

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

def fit_gev(x):
    c, loc, sc = stats.genextreme.fit(x)
    return -c, loc, sc                        # (xi, mu, sigma)

rows, fits = [], {}
for c in cols:
    x = BM[c]
    xi, mu, sg = fit_gev(x)
    fits[c] = (xi, mu, sg)
    H = num_hessian(lambda p: gev_negll(p, x), [xi, mu, sg])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    ll = -gev_negll([xi, mu, sg], x)
    loc_g, sc_g = stats.gumbel_r.fit(x)       # restricted model xi = 0
    ll_g = np.sum(stats.gumbel_r.logpdf(x, loc_g, sc_g))
    lr = 2 * (ll - ll_g)
    rows.append({"Asset": c, "xi": xi, "se(xi)": se[0], "mu": mu, "se(mu)": se[1],
                 "sigma": sg, "se(sigma)": se[2], "Log-lik": ll, "AIC": -2 * ll + 6,
                 "LR test xi=0 (p)": stats.chi2.sf(max(lr, 0), 1)})
gev = pd.DataFrame(rows).set_index("Asset")
gev.to_csv(os.path.join(OUT, "table_gev_parameters.csv"), float_format="%.4g")
print("\nGEV PARAMETERS (block maxima of the standardized losses)\n", gev.round(3).to_string())

# ---------- 3. Quality of the fit -----------------------------------------
# KS statistic with a parametric-bootstrap p-value (parameters are estimated, so the
# standard KS p-value would be too large); Anderson-Darling-type statistic as well
def ks_ad(x, par):
    xi, mu, sg = par
    u = np.sort(stats.genextreme.cdf(x, c=-xi, loc=mu, scale=sg))
    u = np.clip(u, 1e-12, 1 - 1e-12); k = len(u)
    i = np.arange(1, k + 1)
    ks = max(np.max(i / k - u), np.max(u - (i - 1) / k))
    ad = -k - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
    return ks, ad

rows = []
for c in cols:
    x, par = BM[c], fits[c]
    ks0, ad0 = ks_ad(x, par)
    ks_b, ad_b = [], []
    for _ in range(B_BOOT):
        xb = stats.genextreme.rvs(c=-par[0], loc=par[1], scale=par[2], size=len(x), random_state=rng)
        try:
            pb = fit_gev(xb)
        except Exception:
            continue
        a, b = ks_ad(xb, pb); ks_b.append(a); ad_b.append(b)
    rows.append({"Asset": c, "KS stat": ks0, "KS p (bootstrap)": np.mean(np.array(ks_b) >= ks0),
                 "AD stat": ad0, "AD p (bootstrap)": np.mean(np.array(ad_b) >= ad0)})
fitq = pd.DataFrame(rows).set_index("Asset")
fitq.to_csv(os.path.join(OUT, "table_gev_fit_quality.csv"), float_format="%.4g")
print("\nQUALITY OF THE GEV FIT\n", fitq.round(3).to_string())
print("Rejections at 5% (KS, AD):", int((fitq['KS p (bootstrap)'] < .05).sum()),
      int((fitq['AD p (bootstrap)'] < .05).sum()), "of", len(fitq))

# Q-Q plots: empirical block maxima vs fitted GEV quantiles
fig, ax = plt.subplots(4, 4, figsize=(14, 12))
for a, c in zip(ax.ravel(), cols):
    x = np.sort(BM[c]); p = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    xi, mu, sg = fits[c]
    q = stats.genextreme.ppf(p, c=-xi, loc=mu, scale=sg)
    a.scatter(q, x, s=10)
    lo, hi = min(q.min(), x.min()), max(q.max(), x.max())
    a.plot([lo, hi], [lo, hi], "r", lw=1)
    a.set_title(f"{c}  (xi={xi:.2f})")
for a in ax.ravel()[len(cols):]:
    a.axis("off")
plt.suptitle(f"GEV Q-Q plots of block maxima (m={M}): fitted quantiles (x) vs empirical (y)", y=1.0)
save_fig("fig_gev_qq.png")

# Histogram of block maxima with the fitted GEV density (selected series)
sel = [c for c in ["PETR4", "VALE3", "ITUB4", "PORT"] if c in cols]
fig, ax = plt.subplots(1, len(sel), figsize=(14, 3.3))
for a, c in zip(ax, sel):
    xi, mu, sg = fits[c]
    xs = np.linspace(BM[c].min() * 0.8, BM[c].max() * 1.1, 300)
    a.hist(BM[c], bins=20, density=True, alpha=0.6)
    a.plot(xs, stats.genextreme.pdf(xs, c=-xi, loc=mu, scale=sg), "r")
    a.set_title(c)
save_fig("fig_gev_density.png")

# ---------- 4. Extremal index ---------------------------------------------
# Threshold u = THETA_Q-quantile of the losses; exceedance times T_1 < ... < T_N
def theta_intervals(x, u):
    """Ferro-Segers (2003) intervals estimator."""
    T = np.where(x > u)[0]
    N = len(T)
    if N < 3:
        return np.nan
    S = np.diff(T).astype(float)
    if S.max() <= 2:
        th = 2 * S.sum() ** 2 / ((N - 1) * np.sum(S ** 2))
    else:
        th = 2 * np.sum(S - 1) ** 2 / ((N - 1) * np.sum((S - 1) * (S - 2)))
    return min(1.0, th)

def theta_runs(x, u, r=5):
    """Runs estimator: clusters are separated by at least r consecutive non-exceedances."""
    T = np.where(x > u)[0]
    if len(T) == 0:
        return np.nan
    clusters = 1 + np.sum(np.diff(T) > r)
    return clusters / len(T)

th_tab = {}
for c in cols:
    x = L[c].values; row = {}
    for q in [0.90, 0.95, 0.975]:
        u = np.quantile(x, q)
        row[f"intervals q={q}"] = theta_intervals(x, u)
        row[f"runs q={q}"] = theta_runs(x, u)
    th_tab[c] = row
theta_df = pd.DataFrame(th_tab).T
theta_df.to_csv(os.path.join(OUT, "table_extremal_index.csv"), float_format="%.3f")
print("\nEXTREMAL INDEX (sensitivity to estimator and threshold)\n", theta_df.round(2).to_string())
theta = pd.Series({c: theta_intervals(L[c].values, np.quantile(L[c].values, THETA_Q)) for c in cols})
print(f"\nTheta used in the VaR (intervals estimator, threshold q={THETA_Q}):\n", theta.round(3).to_string())

# ---------- 5. VaR of Z (loss quantile) -----------------------------------
# Block maxima df: H(x) = F(x)^(m*theta). Solving F(x)=alpha gives  x = H^{-1}(alpha^(m*theta)).
#  - independence : theta = 1  -> p = alpha^m
#  - extremal index theta < 1 -> p = alpha^(m*theta)
def gev_var(par, alpha, th):
    xi, mu, sg = par
    return stats.genextreme.ppf(alpha ** (M * th), c=-xi, loc=mu, scale=sg)

rows = []
for c in cols:
    row = {"Asset": c}
    for a in ALPHAS:
        row[f"Normal {a}"] = stats.norm.ppf(a)
        row[f"Empirical {a}"] = np.quantile(L[c].values, a)
        row[f"GEV indep {a}"] = gev_var(fits[c], a, 1.0)
        row[f"GEV theta {a}"] = gev_var(fits[c], a, theta[c])
    rows.append(row)
vz = pd.DataFrame(rows).set_index("Asset")
vz["theta"] = theta
vz.to_csv(os.path.join(OUT, "table_var_Z.csv"), float_format="%.4f")
print("\nVaR OF Z (standardized loss quantile)\n", vz.round(3).to_string())

# Quantiles for the VaR of the returns in question 7
evt = pd.DataFrame({f"q_indep_{int(a*100)}": [gev_var(fits[c], a, 1.0) for c in cols] for a in ALPHAS}, index=cols)
for a in ALPHAS:
    evt[f"q_theta_{int(a*100)}"] = [gev_var(fits[c], a, theta[c]) for c in cols]
evt["theta"] = theta
evt.to_csv(os.path.join(OUT, "evt_quantiles_gev.csv"), float_format="%.6f")

# Figure: comparison of the 99% quantile of Z across methods
fig, ax = plt.subplots(figsize=(12, 4))
w = 0.2; xs = np.arange(len(cols))
ax.bar(xs - 1.5 * w, vz["Normal 0.99"], w, label="Normal")
ax.bar(xs - 0.5 * w, vz["Empirical 0.99"], w, label="Empirical")
ax.bar(xs + 0.5 * w, vz["GEV indep 0.99"], w, label="GEV, independence")
ax.bar(xs + 1.5 * w, vz["GEV theta 0.99"], w, label="GEV, extremal index")
ax.set_xticks(xs); ax.set_xticklabels(cols, rotation=60); ax.set_ylabel("99% VaR of Z")
ax.legend(ncol=4); ax.set_title("99% VaR of the standardized losses")
save_fig("fig_var_Z_99.png")

print(f"\nDone. Files saved in: {OUT}")
