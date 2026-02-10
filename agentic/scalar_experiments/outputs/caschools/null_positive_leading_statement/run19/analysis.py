import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio
    df["stratio"] = df["students"] / df["teachers"]

    # Academic performance: average of reading and math
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Simple bivariate regression: avg_score ~ stratio
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["avg_score"], X).fit()

    coef = model.params["stratio"]
    r2 = model.rsquared

    # Map evidence strength to Likert in [-100, 100]
    # Expect negative coefficient (lower ratio -> higher scores).
    if coef >= 0:
        # Direction opposite to hypothesized relationship.
        score = -60
    else:
        # Strength based on magnitude and fit.
        # Typical stratio range is about 10–30; scale effect size and r^2.
        effect_strength = min(abs(coef) / 2.0, 1.0)  # cap at 1
        fit_strength = min(r2 / 0.2, 1.0)  # treat R^2 = 0.2 as strong for cross-section
        scalar = 40 + 60 * effect_strength * fit_strength
        score = int(round(scalar))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(max(-100, min(100, score))))


if __name__ == "__main__":
    main()

