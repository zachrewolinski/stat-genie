import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables of interest
    cols = [
        "avg_score",
        "stratio",
        "calworks",
        "lunch",
        "english",
        "income",
        "expenditure",
        "computer",
        "grades",
    ]
    df_model = df[cols].dropna()

    print("Number of observations in model dataset:", len(df_model))
    print("Student-teacher ratio summary:")
    print(df_model["stratio"].describe())
    print("\nAverage test score summary:")
    print(df_model["avg_score"].describe())

    # Simple correlation
    corr = df_model[["stratio", "avg_score"]].corr().loc["stratio", "avg_score"]
    print("\nCorrelation between student-teacher ratio and average score:", corr)

    # Bivariate regression
    model_simple = smf.ols("avg_score ~ stratio", data=df_model).fit(cov_type="HC3")
    print("\n=== Bivariate regression: avg_score ~ stratio ===")
    print(model_simple.summary())

    # Multivariate regression with controls
    formula = (
        "avg_score ~ stratio + calworks + lunch + english + income "
        "+ expenditure + computer + C(grades)"
    )
    model_controls = smf.ols(formula, data=df_model).fit(cov_type="HC3")
    print("\n=== Regression with controls ===")
    print(model_controls.summary())

    # Extract effect size for stratio
    q25, q75 = df_model["stratio"].quantile([0.25, 0.75])
    delta = q75 - q25
    coef_simple = model_simple.params["stratio"]
    coef_controls = model_controls.params["stratio"]

    print("\nInterquartile range of stratio:", delta)
    print("Predicted score change over IQR (simple model):", coef_simple * delta)
    print("Predicted score change over IQR (controls model):", coef_controls * delta)

    print("\nP-value for stratio (simple model):", model_simple.pvalues["stratio"])
    print("P-value for stratio (controls model):", model_controls.pvalues["stratio"])


if __name__ == "__main__":
    main()

