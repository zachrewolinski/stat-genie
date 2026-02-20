import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct student-teacher ratio (students per teacher) and overall test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used in the models.
    model_vars = [
        "testscr",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    df_model = df[model_vars].dropna()

    # Simple (bivariate) association: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()

    # Multiple regression controlling for key demographic and resource covariates.
    X_controls = df_model[
        ["stratio", "income", "english", "lunch", "calworks", "expenditure", "computer"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()

    # Extract key statistics for the student-teacher ratio.
    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]

    coef_controls = model_controls.params["stratio"]
    p_controls = model_controls.pvalues["stratio"]

    corr = df["testscr"].corr(df["stratio"])

    # Decide on the answer:
    # A "Yes" requires a clear negative association (lower ratio -> higher scores)
    # that is statistically significant (p < 0.05) both bivariately and with controls.
    negative_and_sig_simple = (coef_simple < 0) and (p_simple < 0.05)
    negative_and_sig_controls = (coef_controls < 0) and (p_controls < 0.05)

    if negative_and_sig_simple and negative_and_sig_controls:
        response = "Yes"
    else:
        response = "No"

    explanation_lines = []
    explanation_lines.append(
        "I examined 420 California school districts, constructing a student-teacher "
        "ratio (students per teacher) and an overall test score as the average of "
        "reading and math scores."
    )
    explanation_lines.append(
        f"The simple correlation between the student-teacher ratio and test scores "
        f"is {corr:.3f}, where a negative value would indicate that lower ratios "
        f"(smaller classes) are associated with higher performance."
    )
    explanation_lines.append(
        f"In a bivariate linear regression of test scores on the student-teacher "
        f"ratio, the estimated coefficient on the ratio is {coef_simple:.3f} with "
        f"a p-value of {p_simple:.3f}."
    )
    explanation_lines.append(
        "I then estimated a multiple regression including controls for district "
        "income, percent English learners, percent qualifying for reduced-price "
        "lunch, percent on CalWorks, per-student expenditure, and number of "
        f"computers. In this model, the coefficient on the student-teacher ratio is "
        f"{coef_controls:.3f} with a p-value of {p_controls:.3f}."
    )

    if response == "Yes":
        explanation_lines.append(
            "In both the simple and controlled models, the student-teacher ratio "
            "has a statistically significant negative coefficient, providing "
            "evidence that lower student-teacher ratios are associated with "
            "higher academic performance in this dataset."
        )
    else:
        explanation_lines.append(
            "Although I tested for this relationship, the coefficient on the "
            "student-teacher ratio is not consistently negative and statistically "
            "significant across models; therefore, the data do not provide strong "
            "evidence that lower student-teacher ratios are associated with "
            "higher academic performance once observable demographics and resources "
            "are taken into account."
        )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

