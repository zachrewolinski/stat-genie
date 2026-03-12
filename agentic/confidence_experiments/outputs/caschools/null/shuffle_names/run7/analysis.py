import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to their semantic meanings (see info.json).
    df = df.copy()
    df["enrollment"] = df["english"]  # total enrollment
    df["n_teachers"] = df["students"]  # number of teachers
    df["stratio"] = df["enrollment"] / df["n_teachers"]
    df["stratio_sq"] = df["stratio"] ** 2

    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    df["ell_pct"] = df["rownames"]
    df["calworks_pct"] = df["school"]
    df["lunch_pct"] = df["computer"]
    df["num_computers"] = df["county"]
    df["exp_per_student"] = df["grades"]
    df["income_thousands"] = df["income"]

    # Drop any rows with missing data in variables of interest (there should be none).
    key_cols = [
        "stratio",
        "avg_score",
        "read_score",
        "math_score",
        "ell_pct",
        "calworks_pct",
        "lunch_pct",
        "exp_per_student",
        "income_thousands",
    ]
    df = df.dropna(subset=key_cols)

    print("Basic distributions:")
    print(
        f"  stratio: mean = {df['stratio'].mean():.2f}, "
        f"sd = {df['stratio'].std():.2f}, "
        f"min = {df['stratio'].min():.2f}, "
        f"max = {df['stratio'].max():.2f}"
    )
    print(
        f"  avg_score: mean = {df['avg_score'].mean():.1f}, "
        f"sd = {df['avg_score'].std():.1f}, "
        f"min = {df['avg_score'].min():.1f}, "
        f"max = {df['avg_score'].max():.1f}"
    )

    # Basic Pearson correlations.
    pearson_avg = stats.pearsonr(df["stratio"], df["avg_score"])
    pearson_read = stats.pearsonr(df["stratio"], df["read_score"])
    pearson_math = stats.pearsonr(df["stratio"], df["math_score"])

    print("N observations:", len(df))
    print("\nPearson correlations with student-teacher ratio (stratio):")
    print(f"  avg_score:  r = {pearson_avg.statistic:.3f}, p = {pearson_avg.pvalue:.3g}")
    print(f"  read_score: r = {pearson_read.statistic:.3f}, p = {pearson_read.pvalue:.3g}")
    print(f"  math_score: r = {pearson_math.statistic:.3f}, p = {pearson_math.pvalue:.3g}")

    # Simple linear regression: avg_score on stratio.
    mod_simple = smf.ols("avg_score ~ stratio", data=df).fit()
    b1 = mod_simple.params["stratio"]
    p1 = mod_simple.pvalues["stratio"]
    print("\nSimple OLS: avg_score ~ stratio")
    print(f"  Coef(stratio) = {b1:.3f}, SE = {mod_simple.bse['stratio']:.3f}, p = {p1:.3g}")
    print(f"  R-squared = {mod_simple.rsquared:.3f}")

    # Multiple regression with standard covariates.
    formula_controls = (
        "avg_score ~ stratio + ell_pct + calworks_pct + "
        "lunch_pct + income_thousands + exp_per_student"
    )
    mod_controls = smf.ols(formula_controls, data=df).fit()
    b1c = mod_controls.params["stratio"]
    p1c = mod_controls.pvalues["stratio"]
    print("\nMultiple OLS with controls:")
    print("  Formula:", formula_controls)
    print(f"  Coef(stratio) = {b1c:.3f}, SE = {mod_controls.bse['stratio']:.3f}, p = {p1c:.3g}")
    print(f"  R-squared = {mod_controls.rsquared:.3f}")

    # Quadratic term check (still very close to linear null).
    formula_quad = (
        "avg_score ~ stratio + stratio_sq + ell_pct + "
        "calworks_pct + lunch_pct + income_thousands + exp_per_student"
    )
    mod_quad = smf.ols(formula_quad, data=df).fit()
    print("\nMultiple OLS with quadratic stratio:")
    print("  Formula:", formula_quad)
    print(
        "  Coef(stratio_sq) = "
        f"{mod_quad.params['stratio_sq']:.5f}, "
        f"p = {mod_quad.pvalues['stratio_sq']:.3g}"
    )

    # Also show effect sizes for reading and math separately in simple regressions.
    mod_read = smf.ols("read_score ~ stratio", data=df).fit()
    mod_math = smf.ols("math_score ~ stratio", data=df).fit()
    print("\nSimple OLS by subject:")
    print(
        "  read_score: coef = "
        f"{mod_read.params['stratio']:.3f}, p = {mod_read.pvalues['stratio']:.3g}"
    )
    print(
        "  math_score: coef = "
        f"{mod_math.params['stratio']:.3f}, p = {mod_math.pvalues['stratio']:.3g}"
    )


if __name__ == "__main__":
    main()
