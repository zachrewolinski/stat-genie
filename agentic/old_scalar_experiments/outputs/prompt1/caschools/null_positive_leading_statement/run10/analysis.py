import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    cwd = Path(__file__).resolve().parent
    info_path = cwd / "info.json"
    data_path = cwd / "caschools.csv"

    with info_path.open() as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Simple correlation between student-teacher ratio and average score
    corr = df["stratio"].corr(df["avgscore"])

    # Bivariate OLS: avgscore ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["avgscore"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    available_controls = [c for c in controls if c in df.columns]
    X_controls = sm.add_constant(df[["stratio"] + available_controls])
    model_full = sm.OLS(df["avgscore"], X_controls).fit()

    simple_beta = model_simple.params["stratio"]
    simple_pvalue = model_simple.pvalues["stratio"]

    full_beta = model_full.params["stratio"]
    full_pvalue = model_full.pvalues["stratio"]

    # In this dataset, lower student-teacher ratio corresponds to smaller stratio.
    # We consider evidence for the research question if the coefficient is negative
    # (higher scores when stratio is lower) and statistically significant.
    significant = (simple_pvalue < 0.05) and (full_pvalue < 0.05)
    negative = (simple_beta < 0) and (full_beta < 0)

    if negative and significant:
        response = "Yes"
    else:
        response = "No"

    explanation = {
        "research_question": info["research_questions"][0],
        "summary": (
            "I examined whether lower student-teacher ratios (fewer students per teacher) "
            "are associated with higher average academic performance (mean of reading and math scores)."
        ),
        "correlation_stratio_avgscore": corr,
        "bivariate_regression": {
            "coef_stratio": simple_beta,
            "pvalue_stratio": simple_pvalue,
            "r_squared": model_simple.rsquared,
        },
        "multiple_regression": {
            "controls_included": available_controls,
            "coef_stratio": full_beta,
            "pvalue_stratio": full_pvalue,
            "r_squared": model_full.rsquared,
        },
        "interpretation": (
            "A negative coefficient on student-teacher ratio means that districts with fewer students per "
            "teacher tend to have higher test scores, holding other observed characteristics constant. "
            "The statistical significance indicates whether this relationship is unlikely to be due to chance."
        ),
    }

    output = {
        "response": response,
        "explanation": json.dumps(explanation, indent=2),
    }

    conclusion_path = cwd / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()

