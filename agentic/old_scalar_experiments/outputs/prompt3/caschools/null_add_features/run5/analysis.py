import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    cols_needed = [
        "stratio",
        "avg_score",
        "income",
        "english",
        "lunch",
        "calworks",
        "computer",
        "expenditure",
    ]
    df_clean = df[cols_needed].dropna()

    print("Number of observations:", len(df_clean))
    print("\nSummary of key variables:")
    print(df_clean[["stratio", "avg_score"]].describe())

    # Simple bivariate association
    r, p = pearsonr(df_clean["stratio"], df_clean["avg_score"])
    print(
        f"\nPearson correlation between student-teacher ratio and average score: "
        f"r = {r:.3f}, p = {p:.3g}"
    )

    # Simple linear regression
    X_simple = sm.add_constant(df_clean["stratio"])
    model_simple = sm.OLS(df_clean["avg_score"], X_simple).fit()
    print("\nSimple OLS: avg_score ~ stratio")
    print(model_simple.summary())

    # Multiple regression with controls for socioeconomic and demographic factors
    X_controls = df_clean[
        ["stratio", "income", "english", "lunch", "calworks", "computer", "expenditure"]
    ]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_clean["avg_score"], X_controls).fit()
    print(
        "\nMultiple OLS: avg_score ~ stratio + income + english + "
        "lunch + calworks + computer + expenditure"
    )
    print(model_controls.summary())


if __name__ == "__main__":
    main()

