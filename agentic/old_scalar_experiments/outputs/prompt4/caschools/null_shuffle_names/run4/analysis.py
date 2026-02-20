import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Reconstruct semantic variables using the metadata in info.json.
    # 'english' holds total enrollment, 'students' holds number of teachers.
    df = df.copy()
    df["enrollment"] = df["english"]
    df["num_teachers"] = df["students"]
    df["stratio"] = df["enrollment"] / df["num_teachers"]

    # 'district' and 'expenditure' are average reading and math scores.
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Additional controls based on metadata mappings.
    df["calworks_pct"] = df["school"]
    df["lunch_pct"] = df["computer"]
    df["el_pct"] = df["rownames"]
    df["income_ctrl"] = df["income"]
    df["expn_stu"] = df["grades"]

    # Drop rows with missing key variables (dataset is expected to be complete).
    df_model = df.dropna(
        subset=[
            "stratio",
            "testscr",
            "income_ctrl",
            "el_pct",
            "lunch_pct",
            "calworks_pct",
            "expn_stu",
        ]
    )

    n = len(df_model)
    stratio_desc = df_model["stratio"].describe()
    testscr_desc = df_model["testscr"].describe()

    # Simple correlation between student-teacher ratio and test scores.
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple regression: testscr ~ stratio.
    X1 = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X1).fit()
    coef_stratio_simple = float(model_simple.params["stratio"])
    pval_stratio_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    # Multiple regression with demographic and resource controls.
    controls = ["income_ctrl", "el_pct", "lunch_pct", "calworks_pct", "expn_stu"]
    X2 = sm.add_constant(df_model[["stratio"] + controls])
    model_ctrl = sm.OLS(df_model["testscr"], X2).fit()
    coef_stratio_ctrl = float(model_ctrl.params["stratio"])
    pval_stratio_ctrl = float(model_ctrl.pvalues["stratio"])
    r2_ctrl = float(model_ctrl.rsquared)

    # Map statistical evidence to a 0–100 Likert-style confidence score.
    # Negative coefficients imply that lower ratios (smaller classes) are associated with higher scores.
    if (
        coef_stratio_simple < 0
        and coef_stratio_ctrl < 0
        and pval_stratio_simple < 0.01
        and pval_stratio_ctrl < 0.01
    ):
        response = 95
        qualitative = "strong"
    elif (
        coef_stratio_simple < 0
        and coef_stratio_ctrl < 0
        and pval_stratio_simple < 0.05
        and pval_stratio_ctrl < 0.05
    ):
        response = 80
        qualitative = "moderate"
    elif coef_stratio_simple < 0 and pval_stratio_simple < 0.1:
        response = 65
        qualitative = "suggestive"
    elif (
        coef_stratio_simple > 0
        and coef_stratio_ctrl > 0
        and pval_stratio_simple < 0.05
        and pval_stratio_ctrl < 0.05
    ):
        response = 10
        qualitative = "moderate"
    else:
        response = 50
        qualitative = "weak or no"

    # Describe the direction of the effect in words.
    if coef_stratio_simple < 0:
        direction_text = (
            "negative (districts with larger student–teacher ratios "
            "tend to have lower test scores)"
        )
    elif coef_stratio_simple > 0:
        direction_text = (
            "positive (districts with larger student–teacher ratios "
            "tend to have slightly higher test scores)"
        )
    else:
        direction_text = "essentially zero"

    if qualitative == "strong":
        strength_text = "statistically strong"
    elif qualitative == "moderate":
        strength_text = "statistically moderate"
    elif qualitative == "suggestive":
        strength_text = "statistically suggestive"
    else:
        strength_text = "statistically weak or indistinguishable from zero"

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance?\n"
        f"Sample: {n} California K-6/K-8 districts in 1998–1999.\n"
        "Variables: Using the metadata, I reconstructed the student–teacher ratio as total enrollment divided by the number of teachers, "
        "mapping 'english' to enrollment and 'students' to teachers. Academic performance was measured as the average of the district-level reading "
        "and math scores (mapped from 'district' and 'expenditure').\n"
        f"Descriptives: student–teacher ratio mean={stratio_desc['mean']:.2f}, "
        f"min={stratio_desc['min']:.2f}, max={stratio_desc['max']:.2f}; "
        f"test score mean={testscr_desc['mean']:.2f}, "
        f"min={testscr_desc['min']:.2f}, max={testscr_desc['max']:.2f}.\n"
        f"Correlation: The Pearson correlation between the student–teacher ratio and test scores is {corr:.3f} "
        "(negative values mean that smaller classes are associated with higher scores).\n"
        f"Simple regression: In an OLS regression of test scores on the student–teacher ratio alone, "
        f"the coefficient on the ratio is {coef_stratio_simple:.3f} with p-value {pval_stratio_simple:.4g} "
        f"and R-squared {r2_simple:.3f}.\n"
        "Multiple regression: Adding controls for district income, percent on income assistance, percent on reduced-price lunch, "
        f"percent English learners, and expenditures per pupil, the coefficient on the student–teacher ratio is {coef_stratio_ctrl:.3f} "
        f"with p-value {pval_stratio_ctrl:.4g} and R-squared {r2_ctrl:.3f}.\n"
        f"Interpretation: The estimated coefficients on the student–teacher ratio are {direction_text}, "
        f"and the association is {strength_text} after accounting for observed confounders. "
        "Overall, this dataset provides only limited evidence that lower student–teacher ratios are systematically associated with higher academic performance."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
