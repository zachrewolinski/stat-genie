import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map scrambled column names to their semantic meaning using info.json descriptions.
    enrollment = df["english"]  # total enrollment
    teachers = df["students"]  # number of teachers (FTE)
    reading = df["district"]  # average reading score
    math = df["expenditure"]  # average math score

    # Construct key derived variables.
    str_ratio = (enrollment / teachers).rename("str")  # students per teacher
    testscr = (reading + math) / 2.0  # overall academic performance

    print("Student-teacher ratio (STR) summary:")
    print(str_ratio.describe())
    print("\nTest score (average of reading & math) summary:")
    print(testscr.describe())

    # Simple correlation between STR and test scores.
    corr = np.corrcoef(str_ratio, testscr)[0, 1]
    print(f"\nCorrelation between STR and test scores: {corr:.4f}")

    # Simple linear regression: testscr ~ STR.
    X_simple = sm.add_constant(str_ratio)
    model_simple = sm.OLS(testscr, X_simple).fit()
    print("\nSimple OLS: testscr ~ STR")
    print(model_simple.summary())
    print(
        "\nSimple model - STR coefficient:",
        f"{model_simple.params['str']:.4f}",
        "p-value:",
        f"{model_simple.pvalues['str']:.4g}",
    )

    # Multiple regression including key controls (income, poverty, language, spending, etc.).
    controls = df[["income", "school", "computer", "rownames", "grades"]]
    X_controls = sm.add_constant(
        pd.concat([str_ratio, controls], axis=1)
    )
    model_controls = sm.OLS(testscr, X_controls).fit()
    print("\nOLS with controls: testscr ~ STR + controls")
    print(model_controls.summary())
    print(
        "\nControls model - STR coefficient:",
        f"{model_controls.params['str']:.4f}",
        "p-value:",
        f"{model_controls.pvalues['str']:.4g}",
    )


if __name__ == "__main__":
    main()

