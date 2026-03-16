import pandas as pd
from statsmodels.formula.api import ols


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score.
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Basic description of key variables:")
    print(df[["stratio", "testscr", "read", "math"]].describe(), end="\n\n")

    print("Correlation matrix (stratio vs. scores):")
    print(df[["stratio", "testscr", "read", "math"]].corr(), end="\n\n")

    # Simple (bivariate) regressions of performance on student-teacher ratio.
    for outcome in ["testscr", "read", "math"]:
        model = ols(f"{outcome} ~ stratio", data=df).fit()
        coef = model.params["stratio"]
        se = model.bse["stratio"]
        tval = model.tvalues["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared
        print(f"Bivariate OLS: {outcome} ~ stratio")
        print(
            f"  coef(stratio) = {coef:.3f}, se = {se:.3f}, "
            f"t = {tval:.2f}, p = {pval:.4g}, R^2 = {r2:.3f}"
        )
        print()

    # Multiple regression controlling for key demographic and resource variables.
    formula = (
        "testscr ~ stratio + income + english + lunch + calworks "
        "+ expenditure + computer"
    )
    model_m = ols(formula, data=df).fit()
    coef = model_m.params["stratio"]
    se = model_m.bse["stratio"]
    tval = model_m.tvalues["stratio"]
    pval = model_m.pvalues["stratio"]
    r2 = model_m.rsquared
    print("Multiple OLS: testscr ~ stratio + controls")
    print(
        f"  coef(stratio) = {coef:.3f}, se = {se:.3f}, "
        f"t = {tval:.2f}, p = {pval:.4g}, R^2 = {r2:.3f}"
    )


if __name__ == "__main__":
    main()

