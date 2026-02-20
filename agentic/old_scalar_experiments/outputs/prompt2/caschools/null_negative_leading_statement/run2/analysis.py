import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data = pd.read_csv("caschools.csv")

    data = data.copy()
    data["str_ratio"] = data["students"] / data["teachers"]
    data["testscr"] = (data["read"] + data["math"]) / 2.0
    data = data.dropna(subset=["str_ratio", "testscr"])

    corr = data[["str_ratio", "testscr"]].corr().loc["str_ratio", "testscr"]

    x1 = sm.add_constant(data["str_ratio"])
    model1 = sm.OLS(data["testscr"], x1).fit()
    coef_str1 = float(model1.params["str_ratio"])
    pval1 = float(model1.pvalues["str_ratio"])
    ci1_low, ci1_high = model1.conf_int().loc["str_ratio"]
    r2_1 = float(model1.rsquared)

    covariates = ["income", "english", "lunch", "calworks", "expenditure"]
    x2 = sm.add_constant(data[["str_ratio"] + covariates])
    model2 = sm.OLS(data["testscr"], x2).fit()
    coef_str2 = float(model2.params["str_ratio"])
    pval2 = float(model2.pvalues["str_ratio"])
    ci2_low, ci2_high = model2.conf_int().loc["str_ratio"]
    r2_2 = float(model2.rsquared)

    assoc_negative = (
        coef_str1 < 0
        and coef_str2 < 0
        and pval1 < 0.05
        and pval2 < 0.05
    )

    if assoc_negative:
        response = "Yes"
        confidence = 90
    else:
        response = "No"
        confidence = 80

    if corr < 0:
        corr_direction = (
            "indicating that districts with larger ratios tend to have lower scores."
        )
    elif corr > 0:
        corr_direction = (
            "indicating that districts with larger ratios tend to have slightly higher scores."
        )
    else:
        corr_direction = (
            "indicating essentially no linear relationship between the ratio and scores."
        )

    explanation_lines = [
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?",
        "I constructed student-teacher ratio as students divided by teachers and an overall test score as the average of reading and math scores.",
        f"The Pearson correlation between student-teacher ratio and the average test score is {corr:.3f}, {corr_direction}",
        f"In an unadjusted linear regression of test scores on student-teacher ratio, the coefficient on the ratio is {coef_str1:.3f} (p-value = {pval1:.3g}, 95% CI [{ci1_low:.3f}, {ci1_high:.3f}]), with R-squared {r2_1:.3f}.",
        f"In a regression controlling for income, English-learner share, reduced-price-lunch share, CalWorks share, and per-pupil expenditures, the coefficient on the student-teacher ratio is {coef_str2:.3f} (p-value = {pval2:.3g}, 95% CI [{ci2_low:.3f}, {ci2_high:.3f}]), with R-squared {r2_2:.3f}.",
    ]

    if assoc_negative:
        explanation_lines.append(
            "Because the estimated association between the student-teacher ratio and test scores is negative and statistically significant both before and after adjusting for key socioeconomic and demographic covariates, this dataset provides evidence that districts with lower student-teacher ratios tend to have higher academic performance, though this is an observational association and not a causal estimate."
        )
    else:
        explanation_lines.append(
            "Because the estimated associations between the student-teacher ratio and test scores are very small and not statistically significant in either the unadjusted or adjusted models, this dataset does not provide evidence that districts with lower student-teacher ratios have higher academic performance; within this sample, test scores are essentially unrelated to class size once measured in this way, and the results remain observational rather than causal."
        )
    explanation = "\n".join(explanation_lines)

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
