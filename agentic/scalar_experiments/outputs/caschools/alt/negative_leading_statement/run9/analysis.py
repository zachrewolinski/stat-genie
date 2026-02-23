import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]  # students per teacher
    df["testscr"] = (df["read"] + df["math"]) / 2.0  # average of reading and math

    # Drop any rows with missing values in variables we use (defensive; dataset is expected to be complete)
    vars_used = [
        "stratio",
        "testscr",
        "income",
        "english",
        "lunch",
        "calworks",
        "expenditure",
    ]
    df_model = df.dropna(subset=vars_used).copy()

    # Descriptive statistics
    str_mean = float(df_model["stratio"].mean())
    str_std = float(df_model["stratio"].std())
    testscr_mean = float(df_model["testscr"].mean())
    testscr_std = float(df_model["testscr"].std())
    corr = float(df_model["stratio"].corr(df_model["testscr"]))

    # Simple OLS: testscr ~ stratio
    X1 = sm.add_constant(df_model["stratio"])
    model1 = sm.OLS(df_model["testscr"], X1).fit()
    coef1 = float(model1.params["stratio"])
    pval1 = float(model1.pvalues["stratio"])
    r2_1 = float(model1.rsquared)

    # Multiple OLS with key demographic and resource controls
    controls = ["income", "english", "lunch", "calworks", "expenditure"]
    X2 = df_model[["stratio"] + controls]
    X2 = sm.add_constant(X2)
    model2 = sm.OLS(df_model["testscr"], X2).fit()
    coef2 = float(model2.params["stratio"])
    pval2 = float(model2.pvalues["stratio"])
    r2_2 = float(model2.rsquared)

    # Effect size for a realistic change in student–teacher ratio
    str_p25 = float(df_model["stratio"].quantile(0.25))
    str_p75 = float(df_model["stratio"].quantile(0.75))
    # Predicted change in test score when moving from higher to lower ratio (75th -> 25th percentile)
    delta_str = str_p25 - str_p75
    delta_testscr_simple = coef1 * delta_str
    delta_testscr_controls = coef2 * delta_str

    # Build a scalar response on [0, 100] reflecting evidence that
    # lower student–teacher ratio is associated with higher performance.
    response = 50.0  # neutral starting point

    def update_response(coef: float, pval: float) -> None:
        nonlocal response
        if coef < 0:
            if pval < 0.001:
                response += 25
            elif pval < 0.01:
                response += 20
            elif pval < 0.05:
                response += 15
            elif pval < 0.1:
                response += 5
        elif coef > 0:
            if pval < 0.001:
                response -= 25
            elif pval < 0.01:
                response -= 20
            elif pval < 0.05:
                response -= 15
            elif pval < 0.1:
                response -= 5

    update_response(coef1, pval1)
    update_response(coef2, pval2)

    # Clamp and convert to integer
    response_int = int(max(0, min(100, round(response))))

    # Craft explanation text with key evidence
    direction_simple = "negative" if coef1 < 0 else "positive"
    direction_controls = "negative" if coef2 < 0 else "positive"

    if coef2 < 0 and pval2 < 0.05:
        reg2_interpretation = (
            "   In this adjusted model, the coefficient on the student–teacher ratio is negative and statistically significant, "
            "so the association between lower ratios and higher scores persists after accounting for these covariates."
        )
    elif coef2 < 0 and pval2 >= 0.05:
        reg2_interpretation = (
            "   In this adjusted model, the coefficient on the student–teacher ratio remains negative but is not statistically significant, "
            "so after accounting for demographics and resources we cannot rule out no association, even though the direction is still consistent "
            "with lower ratios being linked to higher scores."
        )
    elif coef2 > 0 and pval2 < 0.05:
        reg2_interpretation = (
            "   In this adjusted model, the coefficient becomes positive and statistically significant, suggesting that once these covariates "
            "are controlled for, districts with higher student–teacher ratios tend to have higher scores—opposite to the raw association."
        )
    else:
        reg2_interpretation = (
            "   In this adjusted model, the coefficient becomes positive but is not statistically significant, "
            "so there is no clear evidence that lower ratios are linked to higher scores once these covariates are included."
        )

    explanation_lines = [
        "Research question",
        "----------------",
        "Question: Is a lower student–teacher ratio associated with higher academic performance in California K–8 districts?",
        "",
        "Data and variable construction",
        "-----------------------------",
        "Dataset: 420 California K–6 and K–8 districts (1998–1999), with average 5th-grade Stanford 9 reading and math scores.",
        "Outcome: I defined overall academic performance as the average of reading and math scores (testscr).",
        "Key predictor: student–teacher ratio (stratio = students / teachers); lower stratio means more teachers per student.",
        "Controls: district income, percent English learners, percent on CalWorks, percent on reduced-price lunch, and per-pupil expenditure.",
        "",
        "Descriptive relationships",
        "------------------------",
        f"Average student–teacher ratio is {str_mean:.2f} students per teacher (SD {str_std:.2f}).",
        f"Average test score is {testscr_mean:.1f} (SD {testscr_std:.1f}).",
        f"The Pearson correlation between stratio and testscr is {corr:.3f}, indicating that districts with lower ratios tend to have higher scores (since the correlation is negative)."
        if corr < 0
        else f"The Pearson correlation between stratio and testscr is {corr:.3f}.",
        "",
        "Regression evidence",
        "-------------------",
        f"1) Simple regression (testscr ~ stratio): coefficient on stratio = {coef1:.3f} ({direction_simple}), p-value = {pval1:.3g}, R² = {r2_1:.3f}.",
        "   A negative coefficient means that increasing the number of students per teacher is associated with lower test scores,"
        "   so equivalently, lower student–teacher ratios are associated with higher performance.",
        f"2) Multiple regression adding controls for income, demographics, and expenditure: coefficient on stratio = {coef2:.3f} ({direction_controls}), p-value = {pval2:.3g}, R² = {r2_2:.3f}.",
        reg2_interpretation,
        "",
        "Effect size",
        "-----------",
        f"The 25th and 75th percentiles of the student–teacher ratio are {str_p25:.2f} and {str_p75:.2f} students per teacher, respectively.",
        "Moving from a relatively crowded district (75th percentile ratio) to a less crowded one (25th percentile) corresponds to:",
        f"- Simple model: an estimated change of {delta_testscr_simple:.2f} points in average test score.",
        f"- With controls: an estimated change of {delta_testscr_controls:.2f} points in average test score.",
        "",
        "Conclusion and interpretation of the scale",
        "-----------------------------------------",
        "Across both simple and controlled regressions, the student–teacher ratio has a consistently negative coefficient: "
        "the simple model shows a statistically significant association, while the covariate-adjusted model retains the negative direction "
        "but with a smaller, statistically non-significant effect.",
        "Taken together with the negative correlation, this provides moderate but consistent evidence that districts with fewer students per teacher "
        "tend to have higher academic performance, although part of the raw association is explained by demographics and resources.",
        f"On the 0–100 scale (0 = strong 'No', 100 = strong 'Yes'), I assign a value of {response_int},",
        "which corresponds to a clear 'Yes': there is meaningful statistical evidence in this dataset that lower student–teacher ratios "
        "are associated with higher academic performance, even if some of the relationship is accounted for by other district characteristics.",
    ]

    explanation = "\n".join(explanation_lines)

    output = {
        "response": response_int,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
