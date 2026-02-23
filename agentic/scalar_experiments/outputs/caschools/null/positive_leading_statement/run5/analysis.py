import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Define student-teacher ratio and an overall achievement measure
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    print("Correlation between student-teacher ratio and achievement:")
    print(df[["stratio", "read", "math", "avg_score"]].corr(), end="\n\n")

    # Simple bivariate OLS: avg_score ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    y_avg = df["avg_score"]
    model_simple = sm.OLS(y_avg, X_simple).fit()
    print("Simple OLS: avg_score ~ stratio")
    print(model_simple.summary(), end="\n\n")

    # Multiple OLS controlling for key demographics and resources
    controls = ["income", "calworks", "lunch", "english", "expenditure"]
    X_multi = sm.add_constant(df[["stratio"] + controls])
    model_multi = sm.OLS(y_avg, X_multi, missing="drop").fit()
    print("Multiple OLS: avg_score ~ stratio + controls")
    print(model_multi.summary(), end="\n\n")

    # Separate models for reading and math
    for outcome in ["read", "math"]:
        y = df[outcome]
        X = sm.add_constant(df[["stratio"] + controls])
        model = sm.OLS(y, X, missing="drop").fit()
        print(f"Multiple OLS: {outcome} ~ stratio + controls")
        print(model.summary(), end="\n\n")


if __name__ == "__main__":
    main()

