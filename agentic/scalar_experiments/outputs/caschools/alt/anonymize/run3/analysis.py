import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct variables based on metadata
    # feature6: total enrollment, feature7: number of teachers
    df["str_ratio"] = df["feature6"] / df["feature7"]

    # feature14: avg reading score, feature15: avg math score
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Drop rows with any missing values in variables used below
    vars_for_simple = ["str_ratio", "feature14", "feature15", "testscr"]
    vars_for_controls = vars_for_simple + [
        "feature6",   # enrollment
        "feature8",   # % CalWorks
        "feature9",   # % reduced-price lunch
        "feature10",  # number of computers
        "feature11",  # expenditure per student
        "feature12",  # district average income
        "feature13",  # % English learners
    ]

    df_simple = df[vars_for_simple].dropna()
    df_controls = df[vars_for_controls].dropna()

    print("Number of observations (simple):", len(df_simple))
    print("Number of observations (with controls):", len(df_controls))

    # Correlations
    corr_read = df_simple["str_ratio"].corr(df_simple["feature14"])
    corr_math = df_simple["str_ratio"].corr(df_simple["feature15"])
    corr_test = df_simple["str_ratio"].corr(df_simple["testscr"])

    print("\nCorrelation between student-teacher ratio and scores:")
    print(f"  Reading (feature14): {corr_read:.4f}")
    print(f"  Math    (feature15): {corr_math:.4f}")
    print(f"  Average (testscr)  : {corr_test:.4f}")

    # Simple OLS: testscr ~ str_ratio
    X_simple = sm.add_constant(df_simple["str_ratio"])
    y_simple = df_simple["testscr"]
    model_simple = sm.OLS(y_simple, X_simple).fit()
    coef_str_simple = model_simple.params["str_ratio"]
    p_str_simple = model_simple.pvalues["str_ratio"]

    print("\nSimple OLS: testscr ~ str_ratio")
    print(f"  Coefficient on str_ratio: {coef_str_simple:.4f}")
    print(f"  p-value for str_ratio  : {p_str_simple:.4g}")

    # OLS with controls
    X_controls = df_controls[
        [
            "str_ratio",
            "feature6",
            "feature8",
            "feature9",
            "feature10",
            "feature11",
            "feature12",
            "feature13",
        ]
    ]
    X_controls = sm.add_constant(X_controls)
    y_controls = df_controls["testscr"]
    model_controls = sm.OLS(y_controls, X_controls).fit()
    coef_str_controls = model_controls.params["str_ratio"]
    p_str_controls = model_controls.pvalues["str_ratio"]

    print("\nOLS with controls: testscr ~ str_ratio + controls")
    print(f"  Coefficient on str_ratio: {coef_str_controls:.4f}")
    print(f"  p-value for str_ratio  : {p_str_controls:.4g}")

    # Effect size: predicted change for a 5-student reduction in str_ratio
    delta_ratio = -5.0
    effect_simple = coef_str_simple * delta_ratio
    effect_controls = coef_str_controls * delta_ratio

    print("\nPredicted change in average test score")
    print("for a 5-student reduction in the student-teacher ratio:")
    print(f"  Simple model   : {effect_simple:.3f} points")
    print(f"  With controls  : {effect_controls:.3f} points")


if __name__ == "__main__":
    main()

