import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Derive key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop rows with missing values in key variables
    key_cols = ["str", "testscr", "income", "english", "calworks", "lunch"]
    df_clean = df.dropna(subset=key_cols).copy()

    print("Rows, columns (original):", df.shape)
    print("Rows used in analysis:", df_clean.shape[0])

    # Simple bivariate regression: testscr on str
    X_simple = sm.add_constant(df_clean["str"])
    y = df_clean["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()
    print("\n=== Bivariate regression: testscr ~ str ===")
    print(model_simple.summary())

    # Multiple regression with common controls
    controls = ["income", "english", "calworks", "lunch"]
    X_controls = sm.add_constant(df_clean[["str"] + controls])
    model_multi = sm.OLS(y, X_controls).fit()
    print("\n=== Multiple regression: testscr ~ str + controls ===")
    print(model_multi.summary())

    # Print key coefficient details for str
    coef_simple = model_simple.params["str"]
    pval_simple = model_simple.pvalues["str"]
    coef_multi = model_multi.params["str"]
    pval_multi = model_multi.pvalues["str"]

    print("\nKey coefficient estimates for student-teacher ratio (str):")
    print(f"Simple model: coef={coef_simple:.3f}, p-value={pval_simple:.4g}")
    print(f"Multiple model: coef={coef_multi:.3f}, p-value={pval_multi:.4g}")

    # Illustrative effect: change in testscr for a 5-student change in str
    delta_str = -5  # smaller classes: lower str
    effect_simple = coef_simple * delta_str
    effect_multi = coef_multi * delta_str
    print(
        f"\nPredicted change in testscr for a 5-student decrease in str:"
        f"\n  Simple model: {effect_simple:.2f} points"
        f"\n  Multiple model: {effect_multi:.2f} points"
    )


if __name__ == "__main__":
    main()

