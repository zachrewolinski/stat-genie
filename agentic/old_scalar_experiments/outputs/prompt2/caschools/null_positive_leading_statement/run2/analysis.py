import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any obvious missing values just in case
    df = df.dropna(subset=["str", "testscr"])

    # Simple correlation
    corr = df["str"].corr(df["testscr"])

    # Simple bivariate regression: testscr ~ str
    X_simple = sm.add_constant(df["str"])
    model_simple = sm.OLS(df["testscr"], X_simple).fit()
    coef_str_simple = model_simple.params["str"]
    pval_str_simple = model_simple.pvalues["str"]
    r2_simple = model_simple.rsquared

    # Multiple regression with basic controls
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    X_controls = sm.add_constant(df[["str"] + controls])
    model_controls = sm.OLS(df["testscr"], X_controls).fit()
    coef_str_controls = model_controls.params["str"]
    pval_str_controls = model_controls.pvalues["str"]
    r2_controls = model_controls.rsquared

    # Also check reading and math separately in simple regressions
    out = {
        "corr_str_testscr": corr,
        "simple_coef_str": coef_str_simple,
        "simple_pval_str": pval_str_simple,
        "simple_r2": r2_simple,
        "controls_coef_str": coef_str_controls,
        "controls_pval_str": pval_str_controls,
        "controls_r2": r2_controls,
    }

    # Reading and math separately
    for subject in ["read", "math"]:
        X_subj = sm.add_constant(df["str"])
        model_subj = sm.OLS(df[subject], X_subj).fit()
        out[f"{subject}_coef_str"] = model_subj.params["str"]
        out[f"{subject}_pval_str"] = model_subj.pvalues["str"]
        out[f"{subject}_r2"] = model_subj.rsquared

    # Print a concise summary that we can use to craft the narrative
    print("=== Association between student-teacher ratio (str) and performance ===")
    print(f"Correlation(str, testscr): {out['corr_str_testscr']:.3f}")
    print(
        "Simple OLS testscr ~ str: "
        f"coef(str) = {out['simple_coef_str']:.3f}, "
        f"p-value = {out['simple_pval_str']:.3g}, "
        f"R^2 = {out['simple_r2']:.3f}"
    )
    print(
        "Controls OLS testscr ~ str + controls: "
        f"coef(str) = {out['controls_coef_str']:.3f}, "
        f"p-value = {out['controls_pval_str']:.3g}, "
        f"R^2 = {out['controls_r2']:.3f}"
    )
    print(
        "Simple OLS read ~ str: "
        f"coef(str) = {out['read_coef_str']:.3f}, "
        f"p-value = {out['read_pval_str']:.3g}, "
        f"R^2 = {out['read_r2']:.3f}"
    )
    print(
        "Simple OLS math ~ str: "
        f"coef(str) = {out['math_coef_str']:.3f}, "
        f"p-value = {out['math_pval_str']:.3g}, "
        f"R^2 = {out['math_r2']:.3f}"
    )


if __name__ == "__main__":
    main()

