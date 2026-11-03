# ==============================================================
# 08_var_comparison.py  -  QRM Homework 1, Group 6 (Brazil)
# Question 8: compare the VaR sequences (Cases 1-5) out of sample:
#             number of violations, backtests, level and stability of VaR, graphs
# Input : outputs_q3/returns_with_port.csv, outputs_q4/VaR_case[12]_*.csv, outputs_q7/VaR_case[345]_*.csv
# Output: outputs_q8/
# Requires: pandas numpy scipy matplotlib
# ==============================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------- 0. Setup ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs_q8")
os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.95, 0.99]
FOCUS = ["PETR4", "PORT"]                     # series shown in the graphs

rets = pd.read_csv(os.path.join(BASE_DIR, "outputs_q3", "returns_with_port.csv"), index_col=0, parse_dates=True)

# ---------- 1. Load the five VaR cases -------------------------------------
LABEL = {"case1": "Case 1: Normal", "case2": "Case 2: alt. density", "case3": "Case 3: GEV indep.",
         "case4": "Case 4: GEV ext. index", "case5": "Case 5: GPD"}
COLOR = {"case1": "C0", "case2": "C4", "case3": "C1", "case4": "C2", "case5": "C3"}
VaR = {}
for f in glob.glob(os.path.join(BASE_DIR, "outputs_q[47]", "VaR_case*_*.csv")):
    parts = os.path.basename(f)[:-4].split("_")           # e.g. ['VaR', 'case3', 'gev', 'indep', '99']
    VaR[(parts[1], int(parts[-1]) / 100)] = pd.read_csv(f, index_col=0, parse_dates=True)
cases = sorted({k for k, _ in VaR})
cols = list(VaR[(cases[0], 0.99)].columns)
idx = VaR[(cases[0], 0.99)].index
real = rets.loc[idx, cols]
n = len(real)
print(f"Cases: {cases}; out-of-sample n={n}; series={len(cols)}")

# ---------- 2. Backtest functions ------------------------------------------
def kupiec(x, n, p):
    """Unconditional coverage LR test (Kupiec 1995). H0: violation probability = p."""
    ll0 = (n - x) * np.log(1 - p) + (x * np.log(p) if x > 0 else 0)
    ph = x / n
    ll1 = (n - x) * np.log(1 - ph) + (x * np.log(ph) if x > 0 else 0)
    lr = max(2 * (ll1 - ll0), 0)
    return lr, stats.chi2.sf(lr, 1)

def christoffersen_ind(v):
    """Independence LR test (Christoffersen 1998): are violations clustered? (first-order Markov chain)"""
    v = np.asarray(v).astype(int)
    a, b = v[:-1], v[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    if n01 + n11 == 0:
        return np.nan, np.nan                              # no violations: test not defined
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    def ll(pr, k1, k0):
        return (k1 * np.log(pr) if k1 > 0 else 0) + (k0 * np.log(1 - pr) if k0 > 0 else 0)
    lr = max(2 * (ll(p01, n01, n00) + ll(p11, n11, n10) - ll(p, n01 + n11, n00 + n10)), 0)
    return lr, stats.chi2.sf(lr, 1)

def check_loss(L, q, a):
    """Quantile (tick) loss of the VaR forecast q for the loss L at level a: lower is better."""
    u = L - q
    return u * (a - (u < 0))

# ---------- 3. Compute all statistics --------------------------------------
rows = []
for a in ALPHAS:
    for k in cases:
        for c in cols:
            v = VaR[(k, a)][c]
            viol = (real[c] < -v).values
            x = int(viol.sum()); p = 1 - a
            lr_uc, p_uc = kupiec(x, n, p)
            lr_ind, p_ind = christoffersen_ind(viol)
            lr_cc = lr_uc + lr_ind if not np.isnan(lr_ind) else np.nan
            L = -real[c]
            rows.append({"alpha": a, "Case": k, "Asset": c, "Violations": x, "Expected": p * n,
                         "Violation rate (%)": 100 * x / n,
                         "Binomial p": stats.binomtest(x, n, p).pvalue,
                         "Kupiec p": p_uc, "Christoffersen ind p": p_ind,
                         "Cond. coverage p": stats.chi2.sf(lr_cc, 2) if not np.isnan(lr_cc) else np.nan,
                         "Mean VaR (%)": 100 * v.mean(), "Std VaR (%)": 100 * v.std(),
                         "Mean excess when violated (%)": 100 * (L[viol] - v[viol]).mean() if x > 0 else np.nan,
                         "Max loss / VaR": (L / v).max(),
                         "Check loss (x100)": 100 * check_loss(L, v, a).mean()})
bt = pd.DataFrame(rows)
bt.to_csv(os.path.join(OUT, "table_backtest_all.csv"), index=False, float_format="%.4g")

for a in ALPHAS:
    s = bt[bt.alpha == a]
    # (i) selected series, all cases
    sel = s[s.Asset.isin(FOCUS)].drop(columns="alpha")
    sel.to_csv(os.path.join(OUT, f"table_backtest_focus_{int(a*100)}.csv"), index=False, float_format="%.4g")
    print(f"\n=== {int(a*100)}% VaR: selected series ===\n", sel.round(3).to_string(index=False))
    # (ii) summary over all 16 series, by case
    g = s.groupby("Case")
    summ = pd.DataFrame({"Total violations": g["Violations"].sum(), "Expected": g["Expected"].sum(),
                         "Series rejected by Kupiec (5%)": g["Kupiec p"].apply(lambda z: int((z < .05).sum())),
                         "Series rejected by cond. coverage (5%)": g["Cond. coverage p"].apply(lambda z: int((z < .05).sum())),
                         "Avg VaR (%)": g["Mean VaR (%)"].mean(), "Avg check loss (x100)": g["Check loss (x100)"].mean()})
    summ.to_csv(os.path.join(OUT, f"table_backtest_summary_{int(a*100)}.csv"), float_format="%.4g")
    print(f"\n=== {int(a*100)}% VaR: summary over the {len(cols)} series ===\n", summ.round(3).to_string())
    # (iii) violations: all series x all cases
    s.pivot(index="Asset", columns="Case", values="Violations").loc[cols].to_csv(
        os.path.join(OUT, f"table_violations_{int(a*100)}.csv"))

# ---------- 4. Figures -----------------------------------------------------
# FIG 1: returns and VaR sequences (and violations) for the focus series
for a in ALPHAS:
    fig, ax = plt.subplots(len(FOCUS), 1, figsize=(12, 4.2 * len(FOCUS)), sharex=True)
    for axx, c in zip(np.atleast_1d(ax), FOCUS):
        axx.plot(real.index, real[c] * 100, color="grey", lw=0.7, label="return")
        for k in cases:
            axx.plot(real.index, -VaR[(k, a)][c] * 100, lw=1.1, color=COLOR[k], label=LABEL[k])
        v = real[c] < -VaR[("case1", a)][c]
        axx.scatter(real.index[v], real[c][v] * 100, s=28, color="k", zorder=5, label="violation (Case 1)")
        axx.set_ylabel(f"{c} (%)"); axx.set_title(f"{c}: out-of-sample returns and {int(a*100)}% VaR")
    np.atleast_1d(ax)[0].legend(ncol=3, fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_var_sequences_{int(a*100)}.png"), dpi=150); plt.close()

# FIG 2: violation timeline (one row per case)
for a in ALPHAS:
    fig, ax = plt.subplots(len(FOCUS), 1, figsize=(12, 2.6 * len(FOCUS)), sharex=True)
    for axx, c in zip(np.atleast_1d(ax), FOCUS):
        for j, k in enumerate(cases):
            v = real[c] < -VaR[(k, a)][c]
            axx.scatter(real.index[v], np.full(v.sum(), j), color=COLOR[k], s=35)
        axx.set_yticks(range(len(cases))); axx.set_yticklabels([LABEL[k] for k in cases], fontsize=7)
        axx.set_title(f"{c}: days with return below -VaR ({int(a*100)}%)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_violation_timeline_{int(a*100)}.png"), dpi=150); plt.close()

# FIG 3: cumulative violations vs expected number
fig, ax = plt.subplots(len(FOCUS), 2, figsize=(12, 3.6 * len(FOCUS)), sharex=True)
for i, c in enumerate(FOCUS):
    for j, a in enumerate(ALPHAS):
        for k in cases:
            ax[i, j].step(real.index, (real[c] < -VaR[(k, a)][c]).cumsum(), color=COLOR[k], label=LABEL[k], lw=1.2)
        ax[i, j].plot(real.index, (1 - a) * np.arange(1, n + 1), "k--", label="expected")
        ax[i, j].set_title(f"{c}, {int(a*100)}%: cumulative violations")
ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_cumulative_violations.png"), dpi=150); plt.close()

# FIG 4: violations by case for all series (99%), against the expected value
for a in ALPHAS:
    piv = bt[bt.alpha == a].pivot(index="Asset", columns="Case", values="Violations").loc[cols]
    fig, ax = plt.subplots(figsize=(13, 4))
    w = 0.8 / len(cases); xs = np.arange(len(cols))
    for j, k in enumerate(cases):
        ax.bar(xs + (j - (len(cases) - 1) / 2) * w, piv[k], w, color=COLOR[k], label=LABEL[k])
    ax.axhline((1 - a) * n, color="k", ls="--", label="expected")
    ax.set_xticks(xs); ax.set_xticklabels(cols, rotation=60); ax.set_ylabel("violations")
    ax.set_title(f"Number of days with return below -VaR ({int(a*100)}%), 249 days"); ax.legend(ncol=3, fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, f"fig_violations_all_series_{int(a*100)}.png"), dpi=150); plt.close()

# FIG 5: distribution of the VaR over time for the focus series (level and stability)
fig, ax = plt.subplots(len(FOCUS), 2, figsize=(12, 3.6 * len(FOCUS)))
for i, c in enumerate(FOCUS):
    for j, a in enumerate(ALPHAS):
        ax[i, j].boxplot([VaR[(k, a)][c] * 100 for k in cases], showfliers=True)
        ax[i, j].set_xticklabels(cases)
        ax[i, j].set_title(f"{c}: distribution of the {int(a*100)}% VaR over the year (%)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_var_distribution.png"), dpi=150); plt.close()

# FIG 6: difference between each VaR and the Normal VaR (focus series, 99%)
fig, ax = plt.subplots(len(FOCUS), 1, figsize=(12, 3.6 * len(FOCUS)), sharex=True)
for axx, c in zip(np.atleast_1d(ax), FOCUS):
    for k in [x for x in cases if x != "case1"]:
        axx.plot(real.index, (VaR[(k, 0.99)][c] - VaR[("case1", 0.99)][c]) * 100, color=COLOR[k], lw=1, label=LABEL[k])
    axx.axhline(0, color="k", lw=0.6); axx.set_ylabel("% points")
    axx.set_title(f"{c}: 99% VaR minus Case 1 (Normal) VaR")
np.atleast_1d(ax)[0].legend(ncol=2, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_var_difference_to_normal.png"), dpi=150); plt.close()

print(f"\nDone. Files saved in: {OUT}")
