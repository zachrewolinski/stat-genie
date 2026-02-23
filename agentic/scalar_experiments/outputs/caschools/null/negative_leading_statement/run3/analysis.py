import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Number of observations:", len(df))
    print()

    print("Summary statistics for student-teacher ratio and test scores:")
    print(df[["stratio", "testscr"]].describe())
    print()

    # Simple correlation
    r, p_corr = pearsonr(df["stratio"], df["testscr"])
    print(f"Pearson correlation between stratio and testscr: r = {r:.3f}, p = {p_corr:.4g}")
    print()

    # Simple bivariate regression
    model_simple = smf.ols("testscr ~ stratio", data=df).fit(cov_type="HC3")
    print("Bivariate regression: testscr ~ stratio (HC3 robust SE)")
    print(model_simple.summary())
    print()

    # Multiple regression with key controls
    formula_controls = "testscr ~ stratio + income + english + lunch + calworks + expenditure"
    model_controls = smf.ols(formula_controls, data=df).fit(cov_type="HC3")
    print("Multiple regression with controls (HC3 robust SE):")
    print(formula_controls)
    print(model_controls.summary())
    print()

    # Report effect size for a meaningful change in class size
    coef = model_controls.params["stratio"]
    print(f"Controlled model coefficient on stratio: {coef:.3f} test-score points per +1 student per teacher")
    delta = -5  # change in students per teacher (smaller classes)
    effect = coef * delta
    print(
        f"Predicted change in testscr for a {abs(delta)}-student decrease in students per teacher: {effect:.2f} points"
    )


if __name__ == "__main__":
    main()

