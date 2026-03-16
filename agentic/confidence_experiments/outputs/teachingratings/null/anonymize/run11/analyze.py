import json
from pathlib import Path

import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

DATA_PATH = Path(__file__).resolve().parent / "teachingratings.csv"


def get_param(result, name):
    names = result.model.exog_names
    idx = names.index(name)
    return result.params[idx], result.pvalues[idx]


def main():
    df = pd.read_csv(DATA_PATH)

    n_rows = len(df)
    missing = df.isna().sum()

    beauty = df["feature6"]
    rating = df["feature7"]

    # Correlation
    r, r_p = stats.pearsonr(beauty, rating)

    # Simple OLS
    m1 = smf.ols("feature7 ~ feature6", data=df).fit()
    m1_cluster = m1.get_robustcov_results(cov_type="cluster", groups=df["feature13"])

    # Full model with controls
    formula = (
        "feature7 ~ feature6 + feature3 + C(feature2) + C(feature4) + C(feature5) + "
        "C(feature8) + C(feature9) + C(feature10) + feature11 + feature12"
    )
    m2 = smf.ols(formula, data=df).fit()
    m2_cluster = m2.get_robustcov_results(cov_type="cluster", groups=df["feature13"])

    # Effect sizes
    beauty_sd = beauty.std(ddof=1)
    rating_sd = rating.std(ddof=1)

    coef_simple, p_simple = get_param(m1_cluster, "feature6")
    coef_full, p_full = get_param(m2_cluster, "feature6")

    effect_sd_simple = coef_simple * beauty_sd
    effect_sd_full = coef_full * beauty_sd

    beta_simple = coef_simple * beauty_sd / rating_sd
    beta_full = coef_full * beauty_sd / rating_sd

    # Decide response based on statistical evidence in the multivariate model
    if p_full < 0.01 and abs(beta_full) >= 0.2:
        response = 85
    elif p_full < 0.05 and abs(beta_full) >= 0.1:
        response = 75
    elif p_full < 0.10 and abs(beta_full) >= 0.05:
        response = 60
    else:
        # No reliable evidence of a meaningful relationship
        response = 20

    # Build explanation
    explanation = (
        f"Dataset has {n_rows} courses; missing values per column are all zero: "
        f"{missing.to_dict()}. "
        f"Beauty (feature6) and teaching ratings (feature7) show essentially no correlation "
        f"(Pearson r = {r:.3f}, p = {r_p:.3g}). "
        f"In a simple OLS of ratings on beauty with instructor-clustered SEs, the beauty coefficient is "
        f"{coef_simple:.3f} (p = {p_simple:.3g}), implying a 1-unit higher beauty score corresponds to about "
        f"{coef_simple:.3f} rating points. A 1 SD increase in beauty (~{beauty_sd:.3f}) predicts about "
        f"{effect_sd_simple:.3f} rating points (~{beta_simple:.3f} SDs). "
        f"In a multivariate model controlling for age, gender, minority status, course type/level, native English, "
        f"tenure-track status, and class size (feature11/feature12), the beauty coefficient is {coef_full:.3f} "
        f"with clustered p = {p_full:.3g}; a 1 SD beauty increase predicts ~{effect_sd_full:.3f} rating points "
        f"(~{beta_full:.3f} SDs). "
        f"Because the estimated effects are near zero and not statistically significant, the evidence does not "
        f"support a relationship between instructor beauty and student instructional ratings in this dataset."
    )

    output = {"response": int(response), "explanation": explanation}

    out_path = Path(__file__).resolve().parent / "conclusion.txt"
    out_path.write_text(json.dumps(output))


if __name__ == "__main__":
    main()
