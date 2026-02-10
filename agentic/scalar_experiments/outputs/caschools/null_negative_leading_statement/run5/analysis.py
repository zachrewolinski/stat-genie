import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Student-teacher ratio and combined test score (as in the original CASchools dataset).
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Basic summary:")
    print(df[["str", "testscr"]].describe())
    print()

    corr = df["str"].corr(df["testscr"])
    print(f"Correlation between student-teacher ratio and test score: {corr:.4f}")
    print()

    # Simple bivariate regression.
    model_simple = smf.ols("testscr ~ str", data=df).fit()
    print("Bivariate regression: testscr ~ str")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for observed covariates that plausibly confound the relationship.
    formula = "testscr ~ str + income + english + lunch + calworks + computer + expenditure"
    model_controls = smf.ols(formula, data=df).fit()
    print("Multiple regression with controls:")
    print(model_controls.summary())
    print()

    # Print key coefficients for easier inspection.
    for name, model in [("simple", model_simple), ("controls", model_controls)]:
        coef = model.params.get("str", float("nan"))
        se = model.bse.get("str", float("nan"))
        pval = model.pvalues.get("str", float("nan"))
        print(
            f"Model {name}: coef(str)={coef:.4f}, se={se:.4f}, p-value={pval:.4g}"
        )


if __name__ == "__main__":
    main()

