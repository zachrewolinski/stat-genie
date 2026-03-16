import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score.
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    df = df.dropna(subset=["stratio", "testscr"])

    # Simple bivariate correlation.
    r, p_corr = stats.pearsonr(df["stratio"], df["testscr"])

    # Simple linear regression: testscr ~ stratio.
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()

    # Multiple regression with common socioeconomic controls.
    covars = ["income", "english", "lunch", "calworks", "expenditure"]
    covars = [c for c in covars if c in df.columns]
    X_full = sm.add_constant(df[["stratio"] + covars])
    model_full = sm.OLS(df["testscr"], X_full, missing="drop").fit()

    coef_simple = model_simple.params["stratio"]
    p_simple = model_simple.pvalues["stratio"]
    coef_full = model_full.params["stratio"]
    p_full = model_full.pvalues["stratio"]

    print("Correlation between student-teacher ratio and average test score")
    print(f"  r = {r:.3f}, p = {p_corr:.4g}")

    print("\nSimple OLS: testscr ~ stratio")
    print(model_simple.summary().as_text())

    print("\nMultiple OLS: testscr ~ stratio + controls")
    print(model_full.summary().as_text())

    print(
        f"\nSimple model stratio coefficient: {coef_simple:.3f}, "
        f"p = {p_simple:.4g}"
    )
    print(
        f"Full model stratio coefficient:   {coef_full:.3f}, "
        f"p = {p_full:.4g}"
    )

    # Illustrative effect of a 5-student reduction in the ratio.
    delta = -5.0
    eff_simple = coef_simple * delta
    eff_full = coef_full * delta
    print(
        f"\nPredicted change in average test score "
        f"for a 5-student-per-teacher reduction:"
    )
    print(f"  Simple model: {eff_simple:.2f} points")
    print(f"  Full model:   {eff_full:.2f} points")


if __name__ == "__main__":
    main()

