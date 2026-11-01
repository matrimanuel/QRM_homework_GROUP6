# ==============================================================
# 02_descriptive.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 2: descriptive analysis of the log-returns (full sample)
# Input : logreturns_brazil.csv   (produced by 01_clean_data.py)
# Output: folder outputs_q2/ with all tables (CSV) and figures (PNG)
# ==============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save figures to files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore", category=FutureWarning)   # hide library deprecation notices

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs_q2")
os.makedirs(OUT, exist_ok=True)

# look for the returns file next to the script or in data_clean/
for cand in [os.path.join(BASE_DIR, "logreturns_brazil.csv"),
             os.path.join(BASE_DIR, "data_clean", "logreturns_brazil.csv")]:
    if os.path.exists(cand):
        rets = pd.read_csv(cand, index_col=0, parse_dates=True)
        break
else:
    raise FileNotFoundError("logreturns_brazil.csv not found")

rets.columns = [c.replace(".SA", "") for c in rets.columns]
n, k = rets.shape
W = 250                                      # rolling window (about 1 trading year)
print(f"Sample: {rets.index[0].date()} -> {rets.index[-1].date()}, n={n}, assets={k}")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()

# ---------- 1. Summary statistics + normality tests (TABLE 1) -------------
tab = pd.DataFrame(index=rets.columns)
tab["Mean (daily %)"] = rets.mean() * 100
tab["Std (daily %)"] = rets.std() * 100
tab["Ann. vol (%)"] = rets.std() * np.sqrt(250) * 100
tab["Min (%)"] = rets.min() * 100
tab["Max (%)"] = rets.max() * 100
tab["Skewness"] = rets.skew()
tab["Excess kurtosis"] = rets.kurt()                      # 0 for the normal
jb = [stats.jarque_bera(rets[c]) for c in rets]           # H0: skew=0, excess kurt=0
tab["Jarque-Bera"] = [j.statistic for j in jb]
tab["JB p-value"] = [j.pvalue for j in jb]
ad = [stats.anderson(rets[c], dist="norm") for c in rets] # Anderson-Darling
tab["Anderson-Darling"] = [a.statistic for a in ad]
tab["AD 1% crit."] = [a.critical_values[-1] for a in ad]
tab["Zero returns (%)"] = (rets == 0).mean() * 100        # stale-price indicator
tab.round(4).to_csv(os.path.join(OUT, "table1_summary_normality.csv"))
print("\nTABLE 1 - summary statistics and normality tests\n", tab.round(3).to_string())

# ---------- 2. Extreme events (TABLE 2) -----------------------------------
# Number of days beyond +-3 standard deviations vs. what a normal would give
z = (rets - rets.mean()) / rets.std()
exp3 = 2 * (1 - stats.norm.cdf(3)) * n
ext = pd.DataFrame({"Days |z|>3": (z.abs() > 3).sum(),
                    "Expected if normal": exp3,
                    "Days z<-3 (losses)": (z < -3).sum(),
                    "Days z>3 (gains)": (z > 3).sum(),
                    "Days |z|>5": (z.abs() > 5).sum()})
ext["Ratio obs/exp"] = ext["Days |z|>3"] / exp3
ext.round(2).to_csv(os.path.join(OUT, "table2_extreme_events.csv"))
print("\nTABLE 2 - extreme events\n", ext.round(2).to_string())

# Market-wide worst days: equally weighted average return and number of assets beyond -3 sd
mkt = rets.mean(axis=1)
worst = pd.DataFrame({"Avg return (%)": mkt * 100,
                      "Assets with z<-3": (z < -3).sum(axis=1)}).sort_values("Avg return (%)").head(10)
worst.round(2).to_csv(os.path.join(OUT, "table3_worst_market_days.csv"))
print("\nTABLE 3 - ten worst market days\n", worst.round(2).to_string())

# ---------- 3. Serial dependence (TABLE 4) --------------------------------
# Ljung-Box on returns, absolute returns and squared returns; ARCH-LM test
LAGS = [10, 20]
rows = {}
for c in rets:
    r = rets[c]
    row = {}
    for L in LAGS:
        row[f"LB r p(lag{L})"] = acorr_ljungbox(r, lags=[L])["lb_pvalue"].iloc[0]
        row[f"LB |r| p(lag{L})"] = acorr_ljungbox(r.abs(), lags=[L])["lb_pvalue"].iloc[0]
        row[f"LB r2 p(lag{L})"] = acorr_ljungbox(r ** 2, lags=[L])["lb_pvalue"].iloc[0]
    row["ARCH-LM p(10)"] = het_arch(r, nlags=10)[1]
    rows[c] = row
ser = pd.DataFrame(rows).T
ser.to_csv(os.path.join(OUT, "table4_serial_dependence_pvalues.csv"), float_format="%.4g")
summ = pd.DataFrame({"Reject H0 at 5%": (ser < 0.05).sum(), "Out of": k})
summ.to_csv(os.path.join(OUT, "table4b_serial_dependence_summary.csv"))
print("\nTABLE 4 - p-values (number of assets rejecting 'no serial dependence' at 5%)\n",
      summ.to_string())

# ---------- 4. Figures ----------------------------------------------------
sns.set_style("whitegrid")

# FIG 1: return series of four representative assets (volatility clustering)
sel = [c for c in ["PETR4", "VALE3", "ITUB4", "WEGE3"] if c in rets]
fig, ax = plt.subplots(len(sel), 1, figsize=(11, 8), sharex=True)
for a, c in zip(ax, sel):
    a.plot(rets.index, rets[c], lw=0.6)
    a.set_ylabel(c)
ax[0].set_title("Daily log-returns: volatility clustering")
save_fig("fig1_returns_selected.png")

# FIG 2: sample ACF of r and |r| (average across assets and for 3 assets)
nl = 30
acf_r = pd.DataFrame({c: acf(rets[c], nlags=nl, fft=True) for c in rets})
acf_a = pd.DataFrame({c: acf(rets[c].abs(), nlags=nl, fft=True) for c in rets})
band = 1.96 / np.sqrt(n)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, d, t in zip(ax, [acf_r, acf_a], ["ACF of returns", "ACF of absolute returns"]):
    a.plot(range(1, nl + 1), d.iloc[1:], color="grey", alpha=0.35, lw=0.8)
    a.plot(range(1, nl + 1), d.iloc[1:].mean(axis=1), color="C3", lw=2, label="average over assets")
    a.axhline(band, ls="--", c="k"); a.axhline(-band, ls="--", c="k")
    a.set_title(t); a.set_xlabel("lag"); a.legend()
save_fig("fig2_acf_returns_vs_abs.png")

# FIG 3: Q-Q plots against the normal, all assets
cols = 4
rws = int(np.ceil(k / cols))
fig, ax = plt.subplots(rws, cols, figsize=(14, 3 * rws))
for a, c in zip(ax.ravel(), rets):
    stats.probplot(z[c], dist="norm", plot=a)
    a.set_title(c); a.set_xlabel(""); a.set_ylabel("")
for a in ax.ravel()[k:]:
    a.axis("off")
plt.suptitle("Normal Q-Q plots of standardized returns", y=1.0)
save_fig("fig3_qqplots.png")

# FIG 4: histogram vs normal density (four assets)
fig, ax = plt.subplots(1, len(sel), figsize=(14, 3.2))
for a, c in zip(ax, sel):
    a.hist(z[c], bins=80, density=True, alpha=0.6)
    xs = np.linspace(-8, 8, 400)
    a.plot(xs, stats.norm.pdf(xs), "r", lw=1.5)
    a.set_title(c); a.set_xlim(-8, 8)
save_fig("fig4_hist_vs_normal.png")

# FIG 5: correlation heatmap + stats on dependence
C = rets.corr()
C.round(3).to_csv(os.path.join(OUT, "table5_correlation_matrix.csv"))
plt.figure(figsize=(9, 7))
sns.heatmap(C, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=0, vmax=1, annot_kws={"size": 7})
plt.title("Correlation matrix of daily log-returns")
save_fig("fig5_correlation_heatmap.png")
off = C.values[np.triu_indices(k, 1)]
eig = np.linalg.eigvalsh(C.values)[::-1]
print(f"\nCorrelations: mean={off.mean():.3f}, min={off.min():.3f}, max={off.max():.3f}; "
      f"first eigenvalue explains {eig[0]/k*100:.1f}% of total variance")
pairs = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().dropna().sort_values()
print("Lowest pairs:\n", pairs.head(3).round(3).to_string())
print("Highest pairs:\n", pairs.tail(3).round(3).to_string())

# FIG 6: is the mean / variance / skewness / kurtosis constant? rolling statistics
roll = {"Mean (daily %)": rets.rolling(W).mean() * 100,
        "Volatility (ann. %)": rets.rolling(W).std() * np.sqrt(250) * 100,
        "Skewness": rets.rolling(W).skew(),
        "Excess kurtosis": rets.rolling(W).kurt()}
fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for a, (t, d) in zip(ax.ravel(), roll.items()):
    a.plot(d.index, d, color="grey", alpha=0.3, lw=0.7)
    a.plot(d.index, d.mean(axis=1), color="C3", lw=2)
    a.set_title(f"Rolling {W}-day {t}  (red = average of the {k} assets)")
save_fig("fig6_rolling_moments.png")

# FIG 7: rolling average pairwise correlation and selected pairs
def avg_corr(w):
    m = w.corr().values
    return m[np.triu_indices(k, 1)].mean()
avgc = pd.Series({rets.index[i]: avg_corr(rets.iloc[i - W:i]) for i in range(W, n + 1, 5)})
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(avgc.index, avgc, lw=2, label="average pairwise correlation")
for p in [("ITUB4", "BBDC4"), ("PETR4", "VALE3"), ("ABEV3", "EMBJ3")]:
    if p[0] in rets and p[1] in rets:
        ax.plot(rets[p[0]].rolling(W).corr(rets[p[1]]), lw=0.9, label=f"{p[0]}-{p[1]}")
ax.set_title(f"Rolling {W}-day correlations"); ax.legend(ncol=2)
save_fig("fig7_rolling_correlation.png")

# FIG 8: clustering of extremes - waiting times between the 5% largest losses vs Exp
fig, ax = plt.subplots(1, len(sel), figsize=(14, 3.2))
for a, c in zip(ax, sel):
    thr = rets[c].quantile(0.05)
    idx = np.where(rets[c].values < thr)[0]
    wait = np.diff(idx)
    stats.probplot(wait, dist="expon", plot=a)
    a.set_title(c); a.set_xlabel(""); a.set_ylabel("")
plt.suptitle("Q-Q plot of waiting times between 5% worst losses vs exponential (iid benchmark)", y=1.02)
save_fig("fig8_waiting_times.png")

# ---------- 5. Constancy over time: yearly table + formal tests (TABLE 6) --
yr = rets.groupby(rets.index.year)
yearly = pd.DataFrame({"Avg mean (%)": yr.mean().mean(axis=1) * 100,
                       "Avg ann. vol (%)": yr.std().mean(axis=1) * np.sqrt(250) * 100,
                       "Avg skew": yr.skew().mean(axis=1),
                       "Avg exc. kurt": yr.apply(lambda d: d.kurt()).mean(axis=1),
                       "Avg pair. corr": yr.apply(lambda d: d.corr().values[np.triu_indices(k, 1)].mean())})
yearly.round(3).to_csv(os.path.join(OUT, "table6_yearly_moments.csv"))
print("\nTABLE 6 - yearly averages across assets\n", yearly.round(3).to_string())

# Formal test: variance equal in first vs second half (Levene), mean equal (Welch t-test)
h = n // 2
test = pd.DataFrame({
    "Levene p (var, 1st vs 2nd half)": [stats.levene(rets[c][:h], rets[c][h:]).pvalue for c in rets],
    "Welch t p (mean)": [stats.ttest_ind(rets[c][:h], rets[c][h:], equal_var=False).pvalue for c in rets]},
    index=rets.columns)
test.to_csv(os.path.join(OUT, "table7_constancy_tests.csv"), float_format="%.4g")
print("\nTABLE 7 - subsample tests (reject if p<0.05):",
      (test < 0.05).sum().to_dict())

print(f"\nDone. Tables and figures saved in: {OUT}")
