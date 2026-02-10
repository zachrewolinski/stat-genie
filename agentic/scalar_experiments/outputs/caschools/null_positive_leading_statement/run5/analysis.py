import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio and average test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables we use (should be none, but safe).
    cols = [
        "score",
        "stratio",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
        "computer",
    ]
    data = df[cols].dropna()

    # Simple bivariate regression: score on stratio.
    X_simple = sm.add_constant(data["stratio"])
    model_simple = sm.OLS(data["score"], X_simple).fit()

    # Multiple regression controlling for key demographics and resources.
    X_controls = data[
        [
            "stratio",
            "income",
            "english",
            "lunch",
            "calworks",
            "expenditure",
            "computer",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(data["score"], X_controls).fit()

    print("Bivariate regression: score ~ stratio")
    print(model_simple.summary())
    print("\nControlled regression: score ~ stratio + demographics/resources")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

