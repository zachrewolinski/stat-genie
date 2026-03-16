import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and overall test score
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    print("Student-teacher ratio (str) summary - full sample:")
    print(df["str"].describe())
    print()

    print("Correlation between str and testscr - full sample:")
    print(df[["str", "testscr"]].corr())
    print()

    # Simple bivariate regression (full sample)
    model_simple = smf.ols("testscr ~ str", data=df).fit()
    print("Bivariate regression (full sample): testscr ~ str")
    print(model_simple.summary())
    print()

    # Multiple regression controlling for key demographics and resources (full sample)
    model_controls = smf.ols(
        "testscr ~ str + income + english + lunch + calworks + expenditure",
        data=df,
    ).fit()
    print(
        "Multiple regression with controls (full sample): "
        "testscr ~ str + income + english + lunch + calworks + expenditure"
    )
    print(model_controls.summary())
    print()

    # Subject-specific robustness checks (full sample)
    model_read = smf.ols(
        "read ~ str + income + english + lunch + calworks + expenditure",
        data=df,
    ).fit()
    model_math = smf.ols(
        "math ~ str + income + english + lunch + calworks + expenditure",
        data=df,
    ).fit()

    print("Coefficient on str, subject-specific models with controls (full sample):")
    print(
        "read  ~ str + controls: "
        f"coef_str={model_read.params['str']:.3f}, "
        f"p={model_read.pvalues['str']:.4g}"
    )
    print(
        "math  ~ str + controls: "
        f"coef_str={model_math.params['str']:.3f}, "
        f"p={model_math.pvalues['str']:.4g}"
    )
    print()

    # Robustness: trim to a plausible class-size range based on domain knowledge
    # Typical K-6/8 class sizes are roughly 10–30 students per teacher.
    df_trim = df[(df["str"] >= 10) & (df["str"] <= 30)].copy()
    print("Student-teacher ratio (str) summary - trimmed sample (10 <= str <= 30):")
    print(df_trim["str"].describe())
    print(f"Trimmed sample size: {len(df_trim)} (out of {len(df)})")
    print()

    print("Correlation between str and testscr - trimmed sample:")
    print(df_trim[["str", "testscr"]].corr())
    print()

    model_simple_trim = smf.ols("testscr ~ str", data=df_trim).fit()
    print("Bivariate regression (trimmed sample): testscr ~ str")
    print(model_simple_trim.summary())
    print()

    model_controls_trim = smf.ols(
        "testscr ~ str + income + english + lunch + calworks + expenditure",
        data=df_trim,
    ).fit()
    print(
        "Multiple regression with controls (trimmed sample): "
        "testscr ~ str + income + english + lunch + calworks + expenditure"
    )
    print(model_controls_trim.summary())
    print()

    model_read_trim = smf.ols(
        "read ~ str + income + english + lunch + calworks + expenditure",
        data=df_trim,
    ).fit()
    model_math_trim = smf.ols(
        "math ~ str + income + english + lunch + calworks + expenditure",
        data=df_trim,
    ).fit()

    print("Coefficient on str, subject-specific models with controls (trimmed sample):")
    print(
        "read  ~ str + controls: "
        f"coef_str={model_read_trim.params['str']:.3f}, "
        f"p={model_read_trim.pvalues['str']:.4g}"
    )
    print(
        "math  ~ str + controls: "
        f"coef_str={model_math_trim.params['str']:.3f}, "
        f"p={model_math_trim.pvalues['str']:.4g}"
    )


if __name__ == "__main__":
    main()
