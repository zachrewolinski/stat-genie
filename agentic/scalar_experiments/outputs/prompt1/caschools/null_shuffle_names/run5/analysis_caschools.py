import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # According to the metadata in info.json:
    # - "english" holds total enrollment
    # - "students" holds the number of teachers
    # Student–teacher ratio = enrollment / number of teachers.
    df["stratio"] = df["english"] / df["students"]

    # Basic diagnostics on the student–teacher ratio
    desc = df["stratio"].describe()
    print("Student–teacher ratio (stratio) summary:")
    print(desc.to_string())
    print()

    # Academic performance measures:
    # - "district": average reading score
    # - "expenditure": average math score
    outcomes = ["district", "expenditure"]

    # Full-sample associations
    for outcome in outcomes:
        y = df[outcome]
        X = sm.add_constant(df["stratio"])
        model = sm.OLS(y, X).fit()

        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared

        corr = df["stratio"].corr(df[outcome])

        print(f"Outcome: {outcome} (full sample)")
        print(f"  Coefficient on stratio: {coef:.4f}")
        print(f"  p-value: {pval:.4g}")
        print(f"  R-squared: {r2:.4f}")
        print(f"  Pearson correlation(stratio, {outcome}): {corr:.4f}")
        print()

    # Restrict to a plausible band of student–teacher ratios
    band = df[(df["stratio"] >= 10) & (df["stratio"] <= 30)].copy()
    print("Restricted sample where 10 <= stratio <= 30")
    print(f"  Number of districts: {len(band)}")
    print()

    for outcome in outcomes:
        y = band[outcome]
        X = sm.add_constant(band["stratio"])
        model = sm.OLS(y, X).fit()

        coef = model.params["stratio"]
        pval = model.pvalues["stratio"]
        r2 = model.rsquared

        corr = band["stratio"].corr(band[outcome])

        print(f"Outcome: {outcome} (restricted band)")
        print(f"  Coefficient on stratio: {coef:.4f}")
        print(f"  p-value: {pval:.4g}")
        print(f"  R-squared: {r2:.4f}")
        print(f"  Pearson correlation(stratio, {outcome}): {corr:.4f}")
        print()


if __name__ == "__main__":
    main()
