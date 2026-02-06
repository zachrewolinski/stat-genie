import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("teachingratings.csv")

    # Minimal cleaning: drop rows with missing values in fields used
    base_cols = [
        "eval",
        "beauty",
        "age",
        "gender",
        "minority",
        "native",
        "tenure",
        "division",
        "credits",
        "students",
        "allstudents",
    ]
    df_clean = df.dropna(subset=base_cols).copy()

    # Simple bivariate model
    m1 = smf.ols("eval ~ beauty", data=df_clean).fit(cov_type="HC3")

    # Multivariate model with common controls
    formula = (
        "eval ~ beauty + age + students + allstudents "
        "+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)"
    )
    m2 = smf.ols(formula, data=df_clean).fit(cov_type="HC3")

    # Save key results for inspection
    results = {
        "n_rows": len(df_clean),
        "m1_beauty_coef": m1.params.get("beauty", float("nan")),
        "m1_beauty_p": m1.pvalues.get("beauty", float("nan")),
        "m2_beauty_coef": m2.params.get("beauty", float("nan")),
        "m2_beauty_p": m2.pvalues.get("beauty", float("nan")),
        "m1_r2": m1.rsquared,
        "m2_r2": m2.rsquared,
    }

    print("Rows used:", results["n_rows"])
    print("Bivariate model (HC3): beauty coef =", results["m1_beauty_coef"], "p =", results["m1_beauty_p"])
    print("Multivariate model (HC3): beauty coef =", results["m2_beauty_coef"], "p =", results["m2_beauty_p"])
    print("R2: m1 =", results["m1_r2"], ", m2 =", results["m2_r2"])

    # Also show a compact summary table for verification
    print("\nMultivariate model coefficients (beauty + controls):")
    print(m2.summary().tables[1])


if __name__ == "__main__":
    main()
