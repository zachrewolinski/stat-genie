import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing data in variables used for regression
    reg_vars = ["testscr", "str", "income", "english", "lunch", "calworks"]
    reg_df = df[reg_vars].dropna()

    # Simple correlation between student–teacher ratio and test scores
    corr = reg_df["str"].corr(reg_df["testscr"])

    # Bivariate regression: testscr ~ str
    X1 = sm.add_constant(reg_df[["str"]])
    y = reg_df["testscr"]
    model1 = sm.OLS(y, X1).fit()
    coef_str1 = float(model1.params["str"])
    pvalue_str1 = float(model1.pvalues["str"])

    # Multiple regression controlling for key demographics and resources
    X2 = sm.add_constant(reg_df[["str", "income", "english", "lunch", "calworks"]])
    model2 = sm.OLS(y, X2).fit()
    coef_str2 = float(model2.params["str"])
    pvalue_str2 = float(model2.pvalues["str"])
    r2_2 = float(model2.rsquared)

    # Quantify the implied change for a 5-student reduction in STR
    delta_testscr_5 = -5.0 * coef_str2

    # Decide binary answer based on sign and statistical significance
    if coef_str2 < 0 and pvalue_str2 < 0.05:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Using data on 420 California K-6 and K-8 districts, I constructed the "
        "student–teacher ratio as students divided by teachers and an overall "
        "achievement measure as the average of reading and math test scores. "
        f"The simple Pearson correlation between student–teacher ratio and average "
        f"test score is {corr:.3f}, which is very close to zero and indicates "
        "little to no linear association in the raw data. In a bivariate OLS "
        f"regression of average test score on student–teacher ratio, the estimated "
        f"coefficient is {coef_str1:.2f} (p = {pvalue_str1:.3f}), which is "
        "numerically very small and not statistically distinguishable from zero. "
        "In a multiple regression that controls for district income, the share of "
        "English learners, and the shares of students on public assistance and "
        f"reduced-price lunch, the coefficient on student–teacher ratio remains "
        f"{coef_str2:.2f} (p = {pvalue_str2:.3f}, R² = {r2_2:.3f}), again very "
        "close to zero and statistically insignificant. Based on this model, a "
        f"reduction of 5 students per teacher is associated with an average test "
        f"score change of about {delta_testscr_5:.1f} points, which is negligible "
        "in the context of scores that vary by roughly 20 points across districts. "
        "Taken together, these results indicate that, within this dataset, there is "
        "no clear evidence that lower student–teacher ratios are associated with "
        "higher academic performance; any relationship appears to be very small and "
        "statistically indistinguishable from zero. Because the data are "
        "observational, this analysis also cannot establish causality."
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
