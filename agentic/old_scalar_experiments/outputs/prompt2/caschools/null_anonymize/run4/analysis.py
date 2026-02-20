import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on metadata in info.json
    # feature6: total enrollment, feature7: number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    # feature14: reading, feature15: math
    df["avgscore"] = (df["feature14"] + df["feature15"]) / 2.0

    # Clean obvious issues
    df = df.replace([np.inf, -np.inf], np.nan)

    # Correlation analysis
    df_corr = df[["stratio", "avgscore"]].dropna()
    corr, corr_p = stats.pearsonr(df_corr["stratio"], df_corr["avgscore"])

    # Simple linear regression
    df_simple = df_corr.copy()
    X_simple = sm.add_constant(df_simple["stratio"])
    model_simple = sm.OLS(df_simple["avgscore"], X_simple).fit()
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    # Multiple regression with key covariates for robustness
    covariates = ["feature8", "feature9", "feature10", "feature11", "feature12", "feature13"]
    df_multi = df[["avgscore", "stratio"] + covariates].dropna()
    X_multi = sm.add_constant(df_multi[["stratio"] + covariates])
    model_multi = sm.OLS(df_multi["avgscore"], X_multi).fit()
    coef_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]

    # Decision logic: consistent negative, statistically significant association
    if (
        coef_simple < 0
        and coef_multi < 0
        and pval_simple < 0.05
        and pval_multi < 0.05
        and corr < 0
        and corr_p < 0.05
    ):
        response = "Yes"
        if abs(corr) > 0.4:
            confidence = 95
        elif abs(corr) > 0.2:
            confidence = 90
        else:
            confidence = 80
    else:
        response = "No"
        confidence = 70
        if (coef_simple * coef_multi) < 0 or (corr * coef_simple) < 0:
            confidence = 60

    # Build explanation text that matches the computed statistics
    if corr_p < 0.05:
        if corr < 0:
            corr_interpretation = (
                "indicating that districts with lower student–teacher ratios tended to have higher test scores."
            )
        else:
            corr_interpretation = (
                "indicating that districts with higher student–teacher ratios tended to have higher test scores."
            )
    else:
        corr_interpretation = (
            "indicating no statistically significant linear relationship between student–teacher ratios and test scores."
        )

    if pval_simple < 0.05:
        if coef_simple < 0:
            simple_interpretation = (
                "so larger classes (more students per teacher) are associated with lower average scores in this bivariate model."
            )
        else:
            simple_interpretation = (
                "so larger classes (more students per teacher) are associated with higher average scores in this bivariate model."
            )
    else:
        simple_interpretation = (
            "so in this bivariate model the estimated effect of student–teacher ratio on scores is not statistically distinguishable from zero."
        )

    if pval_multi < 0.05:
        if coef_multi < 0:
            multi_interpretation = (
                "Even after controlling for poverty, computer availability, expenditures, income, "
                "and the share of English learners, districts with smaller student–teacher ratios tend to have higher scores."
            )
        else:
            multi_interpretation = (
                "Even after controlling for poverty, computer availability, expenditures, income, "
                "and the share of English learners, districts with larger student–teacher ratios tend to have higher scores."
            )
    else:
        multi_interpretation = (
            "After adding controls for poverty, computer availability, expenditures, income, and the share of English learners, "
            "the coefficient on student–teacher ratio remains small and statistically insignificant."
        )

    if response == "Yes":
        overall_conclusion = (
            "Taken together, these results provide consistent evidence that lower student–teacher ratios are associated with higher academic performance in this dataset."
        )
    else:
        overall_conclusion = (
            "Taken together, these results do not provide strong evidence that lower student–teacher ratios are associated with higher academic performance in this dataset."
        )

    explanation = (
        "Using data on 420 California K-6 and K-8 districts, "
        "I computed student–teacher ratio as total enrollment divided by number of teachers "
        "and academic performance as the average of district reading and math scores. "
        f"The Pearson correlation between student–teacher ratio and average test score was {corr:.3f} "
        f"(p = {corr_p:.3g}), {corr_interpretation} "
        f"A simple linear regression of average score on student–teacher ratio yielded a slope of {coef_simple:.3f} points per additional student per teacher "
        f"(p = {pval_simple:.3g}), {simple_interpretation} "
        f"A multiple regression controlling for poverty rates, computer availability, expenditures, income, and the share of English learners produced a slope of "
        f"{coef_multi:.3f} (p = {pval_multi:.3g}). {multi_interpretation} "
        f"{overall_conclusion} This analysis is observational and cannot prove causality."
    )

    result = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

