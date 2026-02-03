import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind


def main():
    df = pd.read_csv("reading.csv")

    # Basic cleaning
    df = df.copy()
    df = df[df["speed"].notna()]
    df = df[df["speed"] > 0]

    # Focus on dyslexia participants (including severe)
    dys = df[df["dyslexia_bin"] == 1]

    # Group statistics
    group_stats = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).rename(index={0: "No Reader View", 1: "Reader View"})

    # Two-sample t-test (Welch)
    rv = dys[dys["reader_view"] == 1]["speed"].values
    nrv = dys[dys["reader_view"] == 0]["speed"].values
    t_stat, p_val, _ = ttest_ind(rv, nrv, usevar="unequal")

    # Log-speed regression with controls
    dys = dys.copy()
    dys["log_speed"] = np.log(dys["speed"])

    # Build model with reasonable controls
    formula = (
        "log_speed ~ reader_view + num_words + Flesch_Kincaid + age + C(device) + C(page_id) "
        "+ C(english_native) + retake_trial + C(gender)"
    )

    model = smf.ols(formula=formula, data=dys).fit()

    # Extract reader_view coefficient
    coef = model.params.get("reader_view", np.nan)
    pval = model.pvalues.get("reader_view", np.nan)

    # Back-transform to percent change
    pct_change = (np.exp(coef) - 1) * 100 if np.isfinite(coef) else np.nan

    print("Dyslexia-only sample size:", len(dys))
    print("\nSpeed by Reader View (dyslexia only):")
    print(group_stats)
    print("\nWelch t-test (speed): t=%.3f, p=%.4g" % (t_stat, p_val))
    print("\nLog-speed regression (dyslexia only):")
    print("reader_view coef=%.4f, p=%.4g, approx %% change=%.2f" % (coef, pval, pct_change))


if __name__ == "__main__":
    main()
