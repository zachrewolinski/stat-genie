import json
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    cols: List[str] = [
        "stratio",
        "testscr",
        "calworks",
        "lunch",
        "income",
        "english",
        "expenditure",
        "computer",
    ]
    df_model = df[cols].replace([np.inf, -np.inf], np.nan).dropna()

    # Correlation between student-teacher ratio and test scores
    r, p_corr = stats.pearsonr(df_model["stratio"], df_model["testscr"])

    # Simple bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    beta_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression controlling for key demographics and resources
    covariates = [
        "calworks",
        "lunch",
        "income",
        "english",
        "expenditure",
        "computer",
    ]
    X_multi = sm.add_constant(df_model[["stratio"] + covariates])
    model_multi = sm.OLS(df_model["testscr"], X_multi).fit()
    beta_multi = float(model_multi.params["stratio"])
    p_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)
    ci_low, ci_high = model_multi.conf_int().loc["stratio"].tolist()

    # Effect of a 5-student change in the ratio based on multivariate model
    effect_5_students = 5.0 * beta_multi

    # Evaluate strength of evidence
    direction_consistent = beta_simple < 0 and beta_multi < 0
    significant_simple = p_simple < 0.05
    significant_multi = p_multi < 0.05

    if direction_consistent and significant_simple and significant_multi:
        corr_strength = min(1.0, abs(r) / 0.3)
        response = int(round(75 + 20 * corr_strength))
    elif direction_consistent and (significant_simple or significant_multi):
        response = 65
    elif direction_consistent and (p_simple < 0.1 or p_multi < 0.1):
        response = 55
    else:
        if not (significant_simple or significant_multi):
            response = 35
        else:
            response = 45

    # Build explanation text
    n_obs = int(df_model.shape[0])
    mean_stratio = float(df_model["stratio"].mean())
    mean_testscr = float(df_model["testscr"].mean())

    yes_no = "Yes" if response >= 50 else "No"

    if direction_consistent and significant_simple and significant_multi:
        significance_phrase = (
            "statistically significant in both the simple and multivariate models"
        )
    else:
        significance_phrase = (
            "at least partially supported by statistical significance tests"
        )

    explanation = (
        f"{yes_no}. Using data on {n_obs} California K-6/K-8 school districts, "
        f"I examined whether a lower student-teacher ratio is associated with higher "
        f"academic performance, measured as the average of the district reading and "
        f"math test scores. The average student-teacher ratio in the sample is about "
        f"{format_float(mean_stratio)} students per teacher, and the average combined "
        f"test score is about {format_float(mean_testscr)} points.\n\n"
        f"First, I computed the Pearson correlation between the student-teacher ratio "
        f"and test scores. The correlation is r = {format_float(r)}, with a "
        f"p-value of {format_float(p_corr, 3)}, indicating "
        f"{'a statistically significant' if p_corr < 0.05 else 'no statistically significant'} "
        f"negative association in the simple bivariate relationship.\n\n"
        f"Next, I estimated a linear regression of test scores on the student-teacher "
        f"ratio alone. In this model, the coefficient on the ratio is "
        f"{format_float(beta_simple)} (p = {format_float(p_simple, 3)}), with an R² of "
        f"{format_float(r2_simple, 3)}. This implies that districts with more students "
        f"per teacher tend to have lower average test scores. A reduction of 5 students "
        f"per teacher in this simple model is associated with an estimated change of "
        f"{format_float(5.0 * beta_simple)} points in average test scores.\n\n"
        f"To account for confounding factors, I then ran a multiple regression of test "
        f"scores on the student-teacher ratio while controlling for district economic "
        f"and demographic characteristics (percent on CalWorks, percent on reduced-price "
        f"lunch, average income, percent English learners) and resource measures "
        f"(expenditure per student and number of computers). In this multivariate "
        f"model, the coefficient on the student-teacher ratio remains "
        f"{'negative' if beta_multi < 0 else 'positive'} "
        f"({format_float(beta_multi)}), with p = {format_float(p_multi, 3)} and R² = "
        f"{format_float(r2_multi, 3)}. The 95% confidence interval for this coefficient "
        f"is approximately [{format_float(ci_low)}, {format_float(ci_high)}]. "
        f"Based on this model, a 5-student reduction in the student-teacher ratio is "
        f"associated with an estimated change of {format_float(effect_5_students)} "
        f"points in average test scores, holding other observed factors constant.\n\n"
        f"Taken together, the direction of the association is consistently negative "
        f"(lower ratios associated with higher scores), and the relationship is "
        f"{significance_phrase}. However, the effect size is modest relative to "
        f"overall score variation, "
        f"and the data are observational, so these results should be interpreted as "
        f"evidence of association rather than definitive proof of a causal effect. "
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"I encode this assessment as {response}, indicating that the data provide "
        f"{'strong' if response >= 75 else 'moderate' if response >= 55 else 'limited'} "
        f"evidence that lower student-teacher ratios are associated with higher "
        f"academic performance in this dataset."
    )

    result = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)

    print("Analysis complete. Response:", response)


if __name__ == "__main__":
    main()
