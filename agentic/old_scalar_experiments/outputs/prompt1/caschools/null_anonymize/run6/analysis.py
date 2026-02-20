import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base = Path(__file__).parent
    data_path = base / "caschools.csv"

    df = pd.read_csv(data_path)

    # Map metadata columns to meaningful names
    df = df.rename(
        columns={
            "feature6": "enroll",  # total enrollment
            "feature7": "teachers",  # number of teachers
            "feature8": "calworks",  # % on income assistance
            "feature9": "lunch",  # % reduced-price lunch
            "feature11": "expenditure",  # spending per student
            "feature12": "income",  # district average income (1,000 USD)
            "feature13": "english",  # % English learners
            "feature14": "read",  # average reading score
            "feature15": "math",  # average math score
        }
    )

    # Construct key variables
    df["stratio"] = df["enroll"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing key variables, just in case
    key_cols = ["stratio", "testscr", "calworks", "lunch", "english", "income", "expenditure"]
    df_model = df[key_cols].replace([np.inf, -np.inf], np.nan).dropna()

    # Simple bivariate correlation
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Linear regression controlling for major demographics and resources
    X = df_model[["stratio", "calworks", "lunch", "english", "income", "expenditure"]]
    X = sm.add_constant(X)
    y = df_model["testscr"]

    model = sm.OLS(y, X).fit()
    coef_stratio = model.params["stratio"]
    pval_stratio = model.pvalues["stratio"]

    # Effect size for a meaningful change in student-teacher ratio
    stratio_std = df_model["stratio"].std()
    delta_score_per_sd = coef_stratio * stratio_std

    # Also compare quartiles of student-teacher ratio for an intuitive effect
    df_model["stratio_q"] = pd.qcut(df_model["stratio"], 4, labels=False)
    mean_low = df_model.loc[df_model["stratio_q"] == 0, "testscr"].mean()
    mean_high = df_model.loc[df_model["stratio_q"] == 3, "testscr"].mean()
    diff_q = mean_low - mean_high  # low ratio minus high ratio

    # Decide on yes/no based on sign and significance of association
    is_associated = (coef_stratio < 0) and (pval_stratio < 0.05)
    response = "Yes" if is_associated else "No"

    if is_associated:
        explanation = (
            "Using data on 420 California K-6 and K-8 school districts, "
            "I constructed a student–teacher ratio as total enrollment divided by the number of teachers "
            "and an overall academic performance measure as the average of reading and math scores. "
            f"The simple Pearson correlation between the student–teacher ratio and test scores was {corr:.3f}, "
            "showing that districts with more students per teacher tend to have lower scores. "
            f"In a linear regression of test scores on the student–teacher ratio and key demographic covariates "
            f"(% CalWorks, % reduced-price lunch, % English learners, district income, and expenditures per student), "
            f"the coefficient on the student–teacher ratio was {coef_stratio:.2f} with a p-value of {pval_stratio:.3g}. "
            "This negative and statistically significant coefficient implies that, holding demographics and resources roughly constant, "
            "districts with smaller student–teacher ratios have higher test scores. "
            f"A one–standard-deviation increase in the student–teacher ratio is associated with an estimated change of {delta_score_per_sd:.2f} points in test scores, "
            f"and districts in the lowest quartile of the student–teacher ratio score on average {diff_q:.1f} points higher than those in the highest quartile. "
            "Together, these results provide consistent evidence that lower student–teacher ratios are associated with higher academic performance in this dataset."
        )
    else:
        explanation = (
            "Using data on 420 California K-6 and K-8 school districts, "
            "I constructed a student–teacher ratio as total enrollment divided by the number of teachers "
            "and an overall academic performance measure as the average of reading and math scores. "
            f"The simple Pearson correlation between the student–teacher ratio and test scores was {corr:.3f}, "
            "which is very close to zero, indicating little to no linear relationship between class size and average performance. "
            f"In a linear regression of test scores on the student–teacher ratio and key demographic covariates "
            f"(% CalWorks, % reduced-price lunch, % English learners, district income, and expenditures per student), "
            f"the coefficient on the student–teacher ratio was {coef_stratio:.2f} with a p-value of {pval_stratio:.3g}, "
            "so the estimated effect is extremely small and not statistically distinguishable from zero after accounting for demographics and resources. "
            f"A one–standard-deviation increase in the student–teacher ratio corresponds to an estimated change of only {delta_score_per_sd:.2f} points in test scores, "
            f"and districts in the lowest quartile of the student–teacher ratio score on average {diff_q:.1f} points differently than those in the highest quartile—"
            "a gap that is small relative to the overall score variation in the sample. "
            "Taken together, these results do not provide clear evidence in this dataset that lower student–teacher ratios are associated with meaningfully higher academic performance."
        )

    conclusion = {"response": response, "explanation": explanation}

    out_path = base / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
