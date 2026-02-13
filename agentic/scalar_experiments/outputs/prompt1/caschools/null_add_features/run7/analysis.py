import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables
    df_model = df[["str", "testscr"]].dropna()

    # Simple linear regression of test scores on student–teacher ratio
    X = sm.add_constant(df_model["str"])
    y = df_model["testscr"]
    model = sm.OLS(y, X).fit()

    slope = float(model.params["str"])
    p_value = float(model.pvalues["str"])
    corr = float(df_model["str"].corr(df_model["testscr"]))
    n_obs = int(df_model.shape[0])

    # Decide binary response based on direction and significance of association
    if slope < 0 and p_value < 0.05:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Using data on {n} California school districts, I computed the "
        "student–teacher ratio as students divided by teachers and an "
        "academic performance measure as the average of reading and math "
        "test scores for each district. The Pearson correlation between "
        "student–teacher ratio and average test score was {corr:.3f}, "
        "indicating a {direction} relationship. A simple linear regression "
        "of average test scores on student–teacher ratio yielded an estimated "
        "slope of {slope:.2f} points per additional student per teacher "
        "(p-value = {p_value:.4f}). Because the estimated slope is {sign} "
        "and statistically significant at the 5% level, the data "
        "{support_clause} the conclusion that lower student–teacher ratios "
        "are associated with higher academic performance. This analysis "
        "describes association and does not, by itself, establish a causal "
        "effect of class size on achievement."
    ).format(
        n=n_obs,
        corr=corr,
        direction="negative" if corr < 0 else "non-negative",
        slope=slope,
        p_value=p_value,
        sign="negative" if slope < 0 else "non-negative",
        support_clause="support" if response == "Yes" else "do not support",
    )

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

