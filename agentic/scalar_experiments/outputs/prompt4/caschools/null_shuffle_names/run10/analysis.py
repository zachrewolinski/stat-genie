import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # According to the metadata in info.json:
    # - "english" is total enrollment
    # - "students" is the number of teachers (FTE)
    # - "district" is the average reading score
    # - "expenditure" is the average math score
    # - "income" is district average income (in $1,000)
    # - "school" is percent qualifying for CalWorks (income assistance)
    # - "computer" is percent qualifying for reduced-price lunch
    # - "rownames" is percent of English learners

    enrollment = df["english"]
    teachers = df["students"]
    read_score = df["district"]
    math_score = df["expenditure"]

    income = df["income"]
    pct_calworks = df["school"]
    pct_lunch = df["computer"]
    pct_ell = df["rownames"]

    # Student–teacher ratio: students per teacher
    stratio = enrollment / teachers

    # Overall academic performance: average of reading and math scores
    testscr = (read_score + math_score) / 2.0

    df_analysis = pd.DataFrame(
        {
            "testscr": testscr,
            "stratio": stratio,
            "income": income,
            "pct_calworks": pct_calworks,
            "pct_lunch": pct_lunch,
            "pct_ell": pct_ell,
        }
    )

    # Drop any missing or infinite values just in case
    df_analysis = df_analysis.replace([np.inf, -np.inf], np.nan).dropna()

    # Simple correlation between student–teacher ratio and test scores
    corr = float(df_analysis["stratio"].corr(df_analysis["testscr"]))

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_analysis["stratio"])
    model_simple = sm.OLS(df_analysis["testscr"], X_simple).fit(cov_type="HC1")
    slope_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with key demographic and resource controls
    X_multi = df_analysis[
        ["stratio", "income", "pct_calworks", "pct_lunch", "pct_ell"]
    ]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(df_analysis["testscr"], X_multi).fit(cov_type="HC1")
    slope_multi = float(model_multi.params["stratio"])
    p_multi = float(model_multi.pvalues["stratio"])
    r2_multi = float(model_multi.rsquared)

    # Map evidence strength to a 0–100 Likert-style "Yes" score.
    # We interpret more negative and highly significant coefficients on stratio
    # (students per teacher) as stronger evidence that lower ratios improve performance.
    if slope_multi < 0 and p_multi < 0.001:
        response = 90
        qualitative = (
            "there is a strong, statistically significant negative association between the "
            "student–teacher ratio and test scores"
        )
        answer = (
            "This provides strong evidence that lower student–teacher ratios are associated "
            "with higher academic performance."
        )
    elif slope_multi < 0 and p_multi < 0.01:
        response = 80
        qualitative = (
            "there is a statistically significant negative association between the "
            "student–teacher ratio and test scores"
        )
        answer = (
            "This provides clear evidence that lower student–teacher ratios are associated "
            "with higher academic performance."
        )
    elif slope_multi < 0 and p_multi < 0.05:
        response = 70
        qualitative = (
            "there is a moderately strong negative association between the student–teacher "
            "ratio and test scores that is statistically significant at the 5% level"
        )
        answer = (
            "This supports a 'Yes' answer: lower student–teacher ratios are associated with "
            "higher academic performance."
        )
    elif slope_multi < 0 and p_multi < 0.1:
        response = 60
        qualitative = (
            "there is a negative association between the student–teacher ratio and test "
            "scores, but it is only marginally significant"
        )
        answer = (
            "This suggests, but does not strongly confirm, that lower student–teacher ratios "
            "are associated with higher academic performance."
        )
    elif slope_multi < 0:
        response = 55
        qualitative = (
            "the estimated association between the student–teacher ratio and test scores is "
            "negative but statistically indistinguishable from zero"
        )
        answer = (
            "These results offer weak and inconclusive support for the idea that lower "
            "student–teacher ratios are associated with higher academic performance."
        )
    elif slope_multi > 0 and p_multi < 0.05:
        # Significant association in the opposite direction
        response = 10
        qualitative = (
            "there is a statistically significant positive association between the "
            "student–teacher ratio and test scores"
        )
        answer = (
            "This provides evidence against the hypothesis: in this dataset, higher student–"
            "teacher ratios are associated with higher academic performance."
        )
    else:
        # Little or no clear association
        response = 50
        qualitative = (
            "both the correlation and regression coefficients linking the student–teacher "
            "ratio to test scores are very close to zero and not statistically significant"
        )
        answer = (
            "these data do not provide clear evidence that lower student–teacher ratios are "
            "systematically associated with higher academic performance."
        )

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?\n"
        "I used the California K–6/K–8 district dataset described in info.json. "
        "Following the metadata, I constructed the student–teacher ratio as total enrollment "
        "divided by the number of teachers (students per teacher) and defined academic performance "
        "as the average of district reading and math scores.\n"
        f"First, the Pearson correlation between the student–teacher ratio and average test score "
        f"is {corr:.3f}, which by itself indicates that {qualitative} when this value is near zero.\n"
        f"In a simple linear regression of average test score on the student–teacher ratio, "
        f"each additional student per teacher is associated with a {slope_simple:.2f}-point change in the test score "
        f"(p = {p_simple:.4g}, R² = {r2_simple:.3f}). A negative coefficient here would imply that smaller classes "
        "are associated with higher performance, while a coefficient near zero implies little systematic relationship.\n"
        f"To address potential confounding, I estimated a multiple regression including district income, the "
        f"percentage of students on income assistance, the percentage qualifying for reduced-price lunch, and the "
        f"percentage of English learners. In this model, the coefficient on the student–teacher ratio is "
        f"{slope_multi:.2f} (p = {p_multi:.4g}, R² = {r2_multi:.3f}). This coefficient summarizes how test scores change "
        "with the student–teacher ratio after adjusting for key demographic factors.\n"
        f"In light of these results, {answer} "
        f"The scalar response {response} on a 0–100 scale encodes this conclusion, with higher values corresponding to "
        "stronger support for a 'Yes' answer and lower values to stronger support for 'No'."
    )

    # Write required JSON conclusion
    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
