import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # According to info.json, "english" is total enrollment and "students" is number of teachers.
    df["stratio"] = df["english"] / df["students"]

    # Academic performance: mean of average reading and math scores.
    df["avg_score"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with any missing values in variables of interest (should be none, but safe).
    model_df = df[["stratio", "avg_score", "income", "school", "computer", "rownames", "grades"]].dropna()

    # Simple bivariate regression: avg_score ~ stratio
    X_simple = sm.add_constant(model_df["stratio"])
    y = model_df["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multivariate regression controlling for key covariates.
    X_controls = model_df[["stratio", "income", "school", "computer", "rownames", "grades"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(y, X_controls).fit()

    print("Bivariate regression: avg_score ~ stratio")
    print(model_simple.summary())
    print()
    print("Multivariate regression with controls")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

