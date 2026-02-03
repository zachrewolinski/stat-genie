import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv("reading.csv")

    # Focus on participants with dyslexia (binary flag)
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Drop missing speed or reader_view
    dys = dys.dropna(subset=["speed", "reader_view"])

    # Basic group stats
    g0 = dys[dys["reader_view"] == 0]["speed"]
    g1 = dys[dys["reader_view"] == 1]["speed"]

    mean0 = g0.mean()
    mean1 = g1.mean()
    n0 = g0.shape[0]
    n1 = g1.shape[0]

    t_stat, p_val, _ = ttest_ind(g1, g0, usevar="unequal")
    d = cohen_d(g1, g0)

    # Regression with controls (log speed to reduce skew)
    dys = dys.copy()
    dys["log_speed"] = np.log(dys["speed"])

    # Build formula with categorical controls
    # Keep key content variables that influence reading speed
    formula = (
        "log_speed ~ reader_view + num_words + Flesch_Kincaid + "
        "C(page_id) + C(device) + retake_trial + age + C(gender) + C(english_native)"
    )

    model = smf.ols(formula=formula, data=dys).fit(cov_type="HC3")

    rv_coef = model.params.get("reader_view", np.nan)
    rv_p = model.pvalues.get("reader_view", np.nan)

    # Write a compact analysis report for downstream use
    with open("analysis_results.txt", "w", encoding="utf-8") as f:
        f.write("Dyslexia-only sample\n")
        f.write(f"n reader_view=0: {n0}\n")
        f.write(f"n reader_view=1: {n1}\n")
        f.write(f"mean speed (rv=0): {mean0:.3f}\n")
        f.write(f"mean speed (rv=1): {mean1:.3f}\n")
        f.write(f"mean diff (rv=1 - rv=0): {mean1 - mean0:.3f}\n")
        f.write(f"Welch t-test p-value: {p_val:.6f}\n")
        f.write(f"Cohen's d: {d:.3f}\n")
        f.write("\nRegression (log speed) with controls\n")
        f.write(f"reader_view coef: {rv_coef:.6f}\n")
        f.write(f"reader_view p-value: {rv_p:.6f}\n")


if __name__ == "__main__":
    main()
