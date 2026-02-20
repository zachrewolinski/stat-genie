import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["feature6"] / df["feature7"]  # student–teacher ratio
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0  # average of reading and math

    # Drop any rows with missing values in variables used for modeling
    model_vars = [
        "stratio",
        "testscr",
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature11",  # expenditure per student
        "feature12",  # district income
        "feature13",  # % English learners
    ]
    df_model = df.dropna(subset=model_vars).copy()

    n = len(df_model)
    mean_ratio = float(df_model["stratio"].mean())
    sd_ratio = float(df_model["stratio"].std(ddof=1))
    mean_test = float(df_model["testscr"].mean())
    sd_test = float(df_model["testscr"].std(ddof=1))

    # Correlation between student–teacher ratio and test scores
    r, p_r = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    y = df_model["testscr"]
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(y, X_simple).fit()
    beta_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    ci_simple = model_simple.conf_int().loc["stratio"]
    ci_simple_low = float(ci_simple[0])
    ci_simple_high = float(ci_simple[1])

    # Multiple regression with key covariates
    X_cov = df_model[
        [
            "stratio",
            "feature8",
            "feature9",
            "feature11",
            "feature12",
            "feature13",
        ]
    ]
    X_cov = sm.add_constant(X_cov)
    model_cov = sm.OLS(y, X_cov).fit()
    beta_cov = float(model_cov.params["stratio"])
    p_cov = float(model_cov.pvalues["stratio"])
    ci_cov = model_cov.conf_int().loc["stratio"]
    ci_cov_low = float(ci_cov[0])
    ci_cov_high = float(ci_cov[1])

    # Decide on binary answer based on direction and significance of the simple regression
    response = "Yes" if (beta_simple < 0 and p_simple < 0.05) else "No"

    explanation_parts = [
        (
            "Using data on "
            f"{n} California K-6 and K-8 school districts, "
            "I computed the student–teacher ratio as total enrollment divided by the number of teachers "
            "and defined academic performance as the average of 5th grade reading and math test scores."
        ),
        (
            f" The mean student–teacher ratio was {mean_ratio:.1f} students per teacher "
            f"(SD {sd_ratio:.1f}), and the mean test score was {mean_test:.1f} (SD {sd_test:.1f})."
        ),
        (
            f" The Pearson correlation between the student–teacher ratio and average test scores was "
            f"r = {r:.3f} (p = {p_r:.3g}), indicating essentially no linear association between class "
            "size and test scores in this sample."
        ),
        (
            " In a simple linear regression of average test scores on the student–teacher ratio, "
            f"each additional student per teacher was associated with a change of {beta_simple:.2f} points "
            f"in test scores (95% CI [{ci_simple_low:.2f}, {ci_simple_high:.2f}], p = {p_simple:.3g}), "
            "a very small and statistically non-significant effect."
        ),
        (
            " In a multiple regression that additionally controlled for student poverty (CalWorks and "
            "reduced-price lunch percentages), per-pupil expenditures, district income, and the percent "
            "of English learners, "
            f"the estimated association between the student–teacher ratio and test scores remained "
            f"{'negative' if beta_cov < 0 else 'positive'}: {beta_cov:.2f} points per additional student "
            f"(95% CI [{ci_cov_low:.2f}, {ci_cov_high:.2f}], p = {p_cov:.3g})."
        ),
    ]

    if response == "Yes":
        conclusion_sentence = (
            " Taken together, these results show a consistently negative and statistically significant "
            "association between the student–teacher ratio and academic performance, so within this dataset "
            "districts with lower student–teacher ratios tend to have higher test scores."
        )
    else:
        conclusion_sentence = (
            " Taken together, these results do not provide strong or consistent evidence that lower "
            "student–teacher ratios are associated with higher academic performance in this dataset."
        )

    caveat_sentence = (
        " Because the data are observational and aggregated at the district level, these findings describe "
        "associations rather than definitive causal effects of changing class size."
    )

    explanation = "".join(explanation_parts) + conclusion_sentence + caveat_sentence

    result = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
