import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_path = Path(__file__).resolve().parent
    info_path = base_path / "info.json"
    data_path = base_path / "caschools.csv"

    with info_path.open() as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Construct key variables
    df = df.copy()
    # Student–teacher ratio: students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    # Overall test score as the average of reading and math
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in variables of interest, if any
    df_model = df[
        [
            "testscr",
            "stratio",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
        ]
    ].dropna()

    # Simple correlation between student–teacher ratio and test scores
    corr = df_model["testscr"].corr(df_model["stratio"])

    # Bivariate regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    beta_str_simple = model_simple.params["stratio"]
    pval_str_simple = model_simple.pvalues["stratio"]

    # Multiple regression controlling for key demographics and resources
    X_controls = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_full = sm.OLS(df_model["testscr"], X_controls).fit()
    beta_str_full = model_full.params["stratio"]
    pval_str_full = model_full.pvalues["stratio"]

    # Decide direction of association based on full model coefficient
    # Lower student–teacher ratio is associated with higher scores
    # if the coefficient on "stratio" is significantly negative.
    alpha = 0.05
    has_stat_sig = pval_str_full < alpha
    association_negative = beta_str_full < 0

    if has_stat_sig and association_negative:
        response = "Yes"
        strength = 80
        confidence = 80
        explanation = (
            "Using the California school districts data, I constructed a student–teacher "
            "ratio (students per teacher) and an overall achievement measure (average of "
            "reading and math test scores). In a regression of test scores on the "
            "student–teacher ratio controlling for income, English-learner share, lunch "
            "subsidy share, CalWorks share, and expenditure per student, the coefficient "
            "on the student–teacher ratio is negative and statistically significant at the "
            "5% level. This means districts with fewer students per teacher tend to have "
            "higher test scores, even after accounting for key demographic and funding "
            "differences. The simple correlation between the student–teacher ratio and "
            "test scores is also negative, which is consistent with this pattern. Taken "
            "together, the evidence indicates that a lower student–teacher ratio is "
            "associated with higher academic performance."
        )
    elif has_stat_sig and not association_negative:
        # Statistically significant but in the opposite direction
        response = "No"
        strength = 85
        confidence = 80
        explanation = (
            "I constructed a student–teacher ratio (students per teacher) and an overall "
            "achievement measure (average of reading and math test scores). In a "
            "regression of test scores on the student–teacher ratio controlling for "
            "income, English-learner share, lunch subsidy share, CalWorks share, and "
            "expenditure per student, the coefficient on the student–teacher ratio is "
            "positive and statistically significant at the 5% level. This implies that "
            "districts with fewer students per teacher do not have higher test scores; if "
            "anything, higher student–teacher ratios are associated with higher scores. "
            "While the relationship is statistically clear in the data, its direction "
            "contradicts the idea that smaller classes are linked to better performance."
        )
    else:
        # Not statistically significant at conventional levels
        response = "No"
        strength = 65
        confidence = 70
        explanation = (
            "Using the California school districts data, I constructed a student–teacher "
            "ratio (students per teacher) and an overall achievement measure (average of "
            "reading and math test scores). In a regression of test scores on the "
            "student–teacher ratio controlling for income, English-learner share, lunch "
            "subsidy share, CalWorks share, and expenditure per student, the coefficient "
            "on the student–teacher ratio is not statistically different from zero at "
            "conventional significance levels. The simple correlation between the "
            "student–teacher ratio and test scores is also small in magnitude. These "
            "results suggest that within this dataset, districts with lower student–teacher "
            "ratios do not systematically have higher test scores, once we account for "
            "observable demographic and funding differences."
        )

    conclusion = {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": explanation,
        "details": {
            "research_question": research_question,
            "n_obs": int(df_model.shape[0]),
            "corr_testscr_stratio": float(np.round(corr, 4)),
            "beta_str_simple": float(np.round(beta_str_simple, 4)),
            "pval_str_simple": float(np.round(pval_str_simple, 4)),
            "beta_str_full": float(np.round(beta_str_full, 4)),
            "pval_str_full": float(np.round(pval_str_full, 4)),
        },
    }

    # Write a human-readable summary for debugging (not required by the task).
    summary_path = base_path / "analysis_summary.json"
    with summary_path.open("w") as f:
        json.dump(conclusion, f, indent=2)


if __name__ == "__main__":
    main()

