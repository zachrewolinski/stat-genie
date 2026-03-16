import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "reading.csv"


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return np.nan
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def main():
    df = pd.read_csv(DATA_PATH)

    # Determine dyslexia indicator
    if "dyslexia_bin" in df.columns and df["dyslexia_bin"].notna().any():
        dys = df["dyslexia_bin"] == 1
        dys_label = "dyslexia_bin==1"
    else:
        dys = df["dyslexia"].fillna(0) > 0
        dys_label = "dyslexia>0"

    df = df[dys].copy()

    # Keep relevant columns
    df = df[df["speed"].notna() & df["reader_view"].notna()]
    df = df[df["speed"] > 0]

    df["log_speed"] = np.log(df["speed"])

    # Group stats
    grp = df.groupby("reader_view")
    stats_table = grp["speed"].agg(["count", "mean", "median", "std"])

    # Two-sample tests
    s0 = df[df["reader_view"] == 0]["speed"].values
    s1 = df[df["reader_view"] == 1]["speed"].values

    ttest_speed = stats.ttest_ind(s1, s0, equal_var=False, nan_policy="omit")

    l0 = df[df["reader_view"] == 0]["log_speed"].values
    l1 = df[df["reader_view"] == 1]["log_speed"].values

    ttest_log = stats.ttest_ind(l1, l0, equal_var=False, nan_policy="omit")

    try:
        mwu = stats.mannwhitneyu(s1, s0, alternative="two-sided")
    except Exception:
        mwu = None

    d_speed = cohen_d(s1, s0)
    d_log = cohen_d(l1, l0)

    # Regression with cluster-robust SE by participant
    # Use page_id as categorical control if available
    formula = "log_speed ~ reader_view"
    if "page_id" in df.columns:
        formula += " + C(page_id)"

    model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["uuid"]})

    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)
    # Convert log effect to percent change
    pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

    # Build conclusion
    # Decision logic
    response = 50
    if np.isfinite(pval):
        if pval < 0.01 and pct_change > 0:
            response = 80
        elif pval < 0.05 and pct_change > 0:
            response = 70
        elif pval < 0.10 and pct_change > 0:
            response = 60
        elif pval < 0.10 and pct_change <= 0:
            response = 40
        else:
            response = 30
    else:
        response = 50

    # Prepare explanation
    n_total = len(df)
    n_rv1 = int((df["reader_view"] == 1).sum())
    n_rv0 = int((df["reader_view"] == 0).sum())

    exp_parts = []
    exp_parts.append(
        f"Analysis restricted to participants with dyslexia ({dys_label}); n={n_total} rows (reader_view=1: {n_rv1}, reader_view=0: {n_rv0})."
    )
    if 0 in stats_table.index and 1 in stats_table.index:
        mean0 = stats_table.loc[0, "mean"]
        mean1 = stats_table.loc[1, "mean"]
        med0 = stats_table.loc[0, "median"]
        med1 = stats_table.loc[1, "median"]
        exp_parts.append(
            f"Mean speed without reader view: {mean0:.2f}; with reader view: {mean1:.2f}. Median speed without reader view: {med0:.2f}; with reader view: {med1:.2f}."
        )
    exp_parts.append(
        f"Welch t-test on raw speed: t={ttest_speed.statistic:.3f}, p={ttest_speed.pvalue:.4f}; on log(speed): t={ttest_log.statistic:.3f}, p={ttest_log.pvalue:.4f}."
    )
    if mwu is not None:
        exp_parts.append(
            f"Mann-Whitney U test on speed: U={mwu.statistic:.1f}, p={mwu.pvalue:.4f}."
        )
    exp_parts.append(
        f"Effect size (Cohen's d) raw speed: {d_speed:.3f}; log(speed): {d_log:.3f}."
    )
    exp_parts.append(
        f"Cluster-robust OLS on log(speed) with page controls: coef(reader_view)={coef:.4f}, p={pval:.4f}, implied percent change={pct_change:.2f}%."
    )

    explanation = " ".join(exp_parts)

    output = {"response": int(round(response)), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
