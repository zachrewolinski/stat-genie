import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student–teacher ratio: higher value = more students per teacher
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    print("Basic correlations (Pearson):")
    print("stratio vs read:", df["stratio"].corr(df["read"]))
    print("stratio vs math:", df["stratio"].corr(df["math"]))
    print("stratio vs avg_score:", df["stratio"].corr(df["avg_score"]))

    # Simple bivariate regression: avg_score on stratio
    X_simple = sm.add_constant(df["stratio"])
    y = df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\nSimple OLS: avg_score ~ stratio")
    print(model_simple.summary())

    # Multivariate regression controlling for key demographics and resources
    controls = [
        "income",
        "calworks",
        "lunch",
        "english",
        "computer",
        "expenditure",
    ]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(y, X_controls).fit()
    print("\nMultivariate OLS: avg_score ~ stratio + controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

