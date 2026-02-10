from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avgscore"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used for regression
    model_vars = [
        "avgscore",
        "stratio",
        "calworks",
        "lunch",
        "income",
        "english",
        "expenditure",
    ]
    reg_df = df[model_vars].dropna()

    # Linear model: average score on student–teacher ratio and key covariates
    model = smf.ols(
        "avgscore ~ stratio + calworks + lunch + income + english + expenditure",
        data=reg_df,
    )
    result = model.fit()

    # We care about whether LOWER stratio (smaller classes) is associated
    # with HIGHER performance. A negative coefficient on stratio implies this.
    coef = result.params.get("stratio", float("nan"))
    t_value = result.tvalues.get("stratio", float("nan"))

    # Define an evidence score based on the t-statistic, reoriented so that
    # positive values support "lower ratio -> higher performance".
    # effect_t > 0 implies evidence in favor of the research hypothesis.
    if pd.isna(coef) or pd.isna(t_value):
        likert_scalar = 0
    else:
        effect_t = -t_value  # negative t (coef < 0) becomes positive support
        score = effect_t * 15.0
        # Clamp to the required [-100, 100] Likert range and round to int.
        score = max(-100.0, min(100.0, score))
        likert_scalar = int(round(score))

    # Write ONLY the scalar value to conclusion.txt
    Path("conclusion.txt").write_text(str(likert_scalar))


if __name__ == "__main__":
    main()

