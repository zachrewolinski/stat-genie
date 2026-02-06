import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "reading.csv"
OUTPUT_PATH = "conclusion.txt"


def main():
    df = pd.read_csv(DATA_PATH)

    # Focus on individuals with dyslexia
    df_dys = df[df["dyslexia_bin"] == 1].copy()
    df_dys = df_dys[df_dys["speed"] > 0].copy()

    if df_dys.empty or df_dys["reader_view"].nunique() < 2:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("No\nInsufficient data for dyslexic readers with both reader view conditions.")
        return

    df_dys["log_speed"] = np.log(df_dys["speed"])

    # Descriptive stats
    group_means = df_dys.groupby("reader_view")["speed"].mean()
    mean_rv0 = float(group_means.get(0, np.nan))
    mean_rv1 = float(group_means.get(1, np.nan))
    pct_diff = ((mean_rv1 - mean_rv0) / mean_rv0 * 100.0) if mean_rv0 > 0 else np.nan

    # Paired test on per-participant means (if possible)
    per_user = df_dys.groupby(["uuid", "reader_view"])["speed"].mean().unstack()
    paired = per_user.dropna()
    paired_p = np.nan
    paired_diff = np.nan
    if len(paired) >= 5:
        paired_diff = float((paired[1] - paired[0]).mean())
        tstat, paired_p = stats.ttest_rel(paired[1], paired[0])

    # Mixed effects model (random intercept per user), controlling for page
    model_used = "mixedlm"
    beta = np.nan
    p_value = np.nan
    try:
        mixed = smf.mixedlm("log_speed ~ reader_view + C(page_id)", df_dys, groups=df_dys["uuid"])
        mixed_res = mixed.fit(reml=False, method="lbfgs", maxiter=200)
        beta = float(mixed_res.params.get("reader_view", np.nan))
        p_value = float(mixed_res.pvalues.get("reader_view", np.nan))
    except Exception:
        # Fallback to OLS with cluster-robust SEs by user
        model_used = "ols_cluster"
        ols = smf.ols("log_speed ~ reader_view + C(page_id)", df_dys).fit(
            cov_type="cluster", cov_kwds={"groups": df_dys["uuid"]}
        )
        beta = float(ols.params.get("reader_view", np.nan))
        p_value = float(ols.pvalues.get("reader_view", np.nan))

    # Convert log coefficient to percent change
    pct_change_model = (math.exp(beta) - 1.0) * 100.0 if np.isfinite(beta) else np.nan

    # Decision rule
    improves = bool(np.isfinite(beta) and np.isfinite(p_value) and (beta > 0) and (p_value < 0.05))

    # Write conclusion
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("Yes\n" if improves else "No\n")
        sentence_parts = []
        sentence_parts.append(
            f"Among dyslexic readers, mean speed was {mean_rv0:.1f} wpm without reader view and {mean_rv1:.1f} wpm with reader view"
        )
        if np.isfinite(pct_diff):
            sentence_parts.append(f"(difference {pct_diff:.1f}%)")
        sentence = " ".join(sentence_parts) + "."

        detail_parts = []
        if np.isfinite(pct_change_model) and np.isfinite(p_value):
            detail_parts.append(
                f"A {model_used} model on log speed (controlling for page) estimates a {pct_change_model:.1f}% change with reader view (p={p_value:.3g})"
            )
        if np.isfinite(paired_p):
            detail_parts.append(
                f"paired within-reader comparison shows mean change {paired_diff:.1f} wpm (p={paired_p:.3g})"
            )
        detail = "; ".join(detail_parts) + "." if detail_parts else ""

        if detail:
            f.write(sentence + " " + detail)
        else:
            f.write(sentence)


if __name__ == "__main__":
    main()
