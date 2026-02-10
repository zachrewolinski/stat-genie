import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio: total enrollment / number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["avgscore"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop any rows with missing key variables (defensive, though none expected)
    model_df = df[["avgscore", "stratio"]].dropna()

    # Simple linear regression: avgscore ~ stratio
    X = sm.add_constant(model_df["stratio"])
    y = model_df["avgscore"]
    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]

    # Map coefficient to Likert-style scalar in [-100, 100]
    # Negative coefficient (higher ratio -> lower scores) should yield positive "Yes"
    # Scale based on standardized effect size (beta) capped at 1.0 in magnitude.
    stratio_std = model_df["stratio"].std()
    avgscore_std = model_df["avgscore"].std()

    # Guard against division by zero
    if stratio_std == 0 or avgscore_std == 0:
        scalar = 0
    else:
        beta = coef * stratio_std / avgscore_std
        # We expect beta to be negative if a lower ratio improves performance.
        # Convert to a "Yes" strength: positive when evidence supports the hypothesis.
        evidence = -beta  # flip sign so supporting evidence is positive
        evidence_capped = max(min(evidence, 1.0), -1.0)
        scalar = int(round(evidence_capped * 100))

    # Write scalar to conclusion.txt with no extra text
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

