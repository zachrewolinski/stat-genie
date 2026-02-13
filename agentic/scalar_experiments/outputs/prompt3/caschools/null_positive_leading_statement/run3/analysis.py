import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["testscr"] = (df["read"] + df["math"]) / 2.0
    df["stratio"] = df["students"] / df["teachers"]

    cols = ["testscr", "stratio", "income", "english", "lunch", "calworks"]
    df_model = df[cols].dropna()

    print(f"Number of observations used: {len(df_model)}")

    # Simple correlation
    corr = df_model["testscr"].corr(df_model["stratio"])
    print(f"Correlation(testscr, stratio): {corr:.4f}")

    # Simple linear regression: testscr ~ stratio
    y = df_model["testscr"]
    X1 = sm.add_constant(df_model[["stratio"]])
    model1 = sm.OLS(y, X1).fit()
    print("\nModel 1: testscr ~ stratio")
    print(
        f"stratio coef: {model1.params['stratio']:.4f}, "
        f"p-value: {model1.pvalues['stratio']:.4g}, "
        f"R^2: {model1.rsquared:.4f}"
    )

    # Multiple regression with key demographic controls
    X2 = sm.add_constant(df_model[["stratio", "income", "english", "lunch", "calworks"]])
    model2 = sm.OLS(y, X2).fit()
    print("\nModel 2: testscr ~ stratio + controls")
    for var in ["stratio", "income", "english", "lunch", "calworks"]:
        print(
            f"{var:8s} coef: {model2.params[var]:8.4f}, "
            f"p-value: {model2.pvalues[var]:.4g}"
        )
    print(f"Model 2 R^2: {model2.rsquared:.4f}")

    # Average test scores by quartile of class size
    df_model = df_model.copy()
    df_model["str_quartile"] = pd.qcut(df_model["stratio"], 4, labels=False)
    group = df_model.groupby("str_quartile")[["stratio", "testscr"]].mean()
    print("\nAverage stratio and testscr by stratio quartile (0 = smallest classes):")
    print(group)


if __name__ == "__main__":
    main()

