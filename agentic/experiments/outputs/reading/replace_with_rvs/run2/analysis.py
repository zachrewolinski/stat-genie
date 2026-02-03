import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("reading.csv")

    # The dyslexia_bin column appears to be a noisy/continuous proxy for dyslexia status.
    # Use a threshold at 0.5 (midpoint) to define the dyslexic group.
    dys = df[df["dyslexia_bin"] >= 0.5].copy()

    # Descriptive stats by reader view
    grouped = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

    # Welch's t-test for difference in means
    rv1 = dys[dys["reader_view"] == 1]["speed"]
    rv0 = dys[dys["reader_view"] == 0]["speed"]
    t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)
    diff = rv1.mean() - rv0.mean()

    # Regression with basic controls
    formula = "speed ~ reader_view + num_words + Flesch_Kincaid + age + correct_rate + retake_trial"
    model = smf.ols(formula, data=dys).fit()
    coef = float(model.params["reader_view"])
    coef_p = float(model.pvalues["reader_view"])

    print("Dyslexic group definition: dyslexia_bin >= 0.5")
    print(grouped.to_string(index=False))
    print(f"Welch t-test: t={t_stat:.3f}, p={p_val:.4f}, mean diff (rv1-rv0)={diff:.2f}")
    print(f"Regression reader_view coef={coef:.2f}, p={coef_p:.4f}")

    # Conclusion logic: improvement requires positive effect and statistical significance.
    improved = (diff > 0 and p_val < 0.05) or (coef > 0 and coef_p < 0.05)

    if improved:
        first_line = "Yes"
        rationale = (
            "Among participants classified as dyslexic (dyslexia_bin ≥ 0.5), reader view shows a statistically "
            "significant positive association with reading speed."
        )
    else:
        first_line = "No"
        rationale = (
            "Among participants classified as dyslexic (dyslexia_bin ≥ 0.5), reader view does not show a "
            "statistically significant improvement in reading speed in either the mean comparison or a controlled regression."
        )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(first_line + "\n")
        f.write(rationale + "\n")


if __name__ == "__main__":
    main()
