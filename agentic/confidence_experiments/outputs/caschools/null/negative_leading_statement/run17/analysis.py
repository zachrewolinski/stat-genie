import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student–teacher ratio and overall test score
    df["str"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic correlation between ratio and performance
    corr = df["str"].corr(df["testscr"])

    # Simple linear regression: testscr ~ str
    X1 = sm.add_constant(df[["str"]])
    model1 = sm.OLS(df["testscr"], X1).fit()
    coef_str1 = float(model1.params["str"])
    pval_str1 = float(model1.pvalues["str"])
    r2_1 = float(model1.rsquared)

    # Multiple regression with common socioeconomic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X2 = sm.add_constant(df[["str"] + controls])
    model2 = sm.OLS(df["testscr"], X2).fit()
    coef_str2 = float(model2.params["str"])
    pval_str2 = float(model2.pvalues["str"])
    r2_2 = float(model2.rsquared)

    # Determine Likert-scale strength and verbal conclusion based on consistency and significance
    strong_evidence = (
        coef_str1 < 0
        and coef_str2 < 0
        and pval_str1 < 0.001
        and pval_str2 < 0.001
    )
    moderate_evidence = (
        not strong_evidence
        and coef_str1 < 0
        and coef_str2 < 0
        and pval_str1 < 0.05
        and pval_str2 < 0.05
    )
    some_evidence = (
        not strong_evidence
        and not moderate_evidence
        and coef_str1 < 0
        and pval_str1 < 0.05
    )

    if strong_evidence:
        response = 85
        conclusion_sentence = (
            "Both the simple and adjusted regressions show a clear, negative and highly "
            "statistically significant association between the student–teacher ratio and "
            "test scores. These results provide strong evidence that lower student–teacher "
            "ratios are associated with higher academic performance, so the appropriate "
            "answer to the research question is 'Yes'."
        )
    elif moderate_evidence:
        response = 70
        conclusion_sentence = (
            "Both the simple and adjusted regressions show a negative and statistically "
            "significant association between the student–teacher ratio and test scores. "
            "These results provide reasonably strong evidence that lower student–teacher "
            "ratios are associated with higher academic performance, so the answer to the "
            "research question is 'Yes'."
        )
    elif some_evidence:
        response = 60
        conclusion_sentence = (
            "The simple regression shows a statistically significant negative association "
            "between the student–teacher ratio and test scores, although the estimated "
            "effect size is modest and less stable once covariates are included. Overall, "
            "there is some evidence that lower student–teacher ratios are associated with "
            "higher academic performance, so the answer to the research question is "
            "better described as 'Yes', but with moderate strength."
        )
    else:
        # Limited or no evidence of association
        response = 30
        conclusion_sentence = (
            "Across both the simple and adjusted regressions, the estimated association "
            "between the student–teacher ratio and test scores is very small in magnitude "
            "and not statistically distinguishable from zero. This dataset therefore does "
            "not provide convincing evidence that lower student–teacher ratios are "
            "associated with higher academic performance, so the appropriate answer to the "
            "research question is 'No'."
        )

    explanation = (
        "Using data on 420 California K-6 and K-8 school districts, "
        "I defined the student–teacher ratio as students divided by teachers and "
        "academic performance as the average of reading and math scores. "
        f"The Pearson correlation between the student–teacher ratio and the average "
        f"test score is {corr:.3f}. "
        f"In a simple linear regression of average test score on the student–teacher ratio, "
        f"the coefficient on the ratio is {coef_str1:.2f} "
        f"(p-value {pval_str1:.4g}, R² = {r2_1:.3f}). "
        f"When controlling for district income, the share of English learners, poverty "
        f"(CalWorks and reduced-price lunch), per-pupil expenditure, and computers, the "
        f"coefficient on the student–teacher ratio is {coef_str2:.2f} "
        f"(p-value {pval_str2:.4g}, R² = {r2_2:.3f}). "
        f"{conclusion_sentence}"
    )

    result = {"response": int(response), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
