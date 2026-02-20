import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Drop any rows with missing values in variables used below
    vars_simple = ["testscr", "stratio"]
    vars_controls = ["income", "english", "lunch", "calworks"]
    analysis_cols = vars_simple + vars_controls
    df_model = df.dropna(subset=analysis_cols).copy()

    # Basic association: Pearson correlation
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    y_simple = df_model["testscr"]
    X_simple = sm.add_constant(df_model[["stratio"]])
    model_simple = sm.OLS(y_simple, X_simple).fit()
    coef_simple = float(model_simple.params["stratio"])
    se_simple = float(model_simple.bse["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression with demographic and resource controls
    y_full = df_model["testscr"]
    X_full = sm.add_constant(df_model[["stratio"] + vars_controls])
    model_full = sm.OLS(y_full, X_full).fit()
    coef_full = float(model_full.params["stratio"])
    se_full = float(model_full.bse["stratio"])
    p_full = float(model_full.pvalues["stratio"])
    r2_full = float(model_full.rsquared)

    # Map evidence to a 0–100 Likert score where higher = stronger "Yes"
    score = evidence_to_score(coef_full, p_full, corr)

    # Qualitative interpretation pieces
    direction = (
        "higher student–teacher ratios (more students per teacher) "
        "are associated with lower test scores"
        if coef_full < 0
        else "higher student–teacher ratios are associated with higher test scores"
    )

    if p_full < 0.001:
        sig_text = "highly statistically significant"
    elif p_full < 0.01:
        sig_text = "strongly statistically significant"
    elif p_full < 0.05:
        sig_text = "statistically significant at the 5% level"
    else:
        sig_text = "not statistically significant at conventional levels"

    if abs(corr) >= 0.6:
        corr_text = "a strong"
    elif abs(corr) >= 0.3:
        corr_text = "a moderate"
    elif abs(corr) >= 0.1:
        corr_text = "a weak"
    else:
        corr_text = "essentially no"

    if coef_full < 0 and p_full < 0.05:
        overall = (
            "Taken together, these results provide clear evidence that "
            "districts with fewer students per teacher tend to have higher "
            "academic performance, even after accounting for key demographic "
            "and resource differences."
        )
    elif coef_full < 0 and p_full >= 0.05:
        overall = (
            "Taken together, these results suggest a negative association "
            "between the student–teacher ratio and academic performance, "
            "but the evidence is not precise enough to be conclusive."
        )
    elif coef_full > 0 and p_full < 0.05:
        overall = (
            "Taken together, these results indicate that in this dataset, "
            "higher student–teacher ratios are actually associated with "
            "higher academic performance, contrary to the usual expectation."
        )
    else:
        overall = (
            "Taken together, these results do not show a clear association "
            "between the student–teacher ratio and academic performance."
        )

    explanation = (
        "Using data on 420 California K–6/K–8 school districts, "
        "I examined whether lower student–teacher ratios are associated with "
        "higher academic performance. I constructed the student–teacher ratio "
        "as total enrollment divided by the number of teachers and defined "
        "academic performance as the average of the fifth-grade reading and "
        "math Stanford 9 test scores.\n\n"
        f"The raw Pearson correlation between the student–teacher ratio and "
        f"average test scores is {corr:.2f}, indicating {corr_text} "
        f"{'negative' if corr < 0 else 'positive' if corr > 0 else 'no clear'} "
        "linear relationship in the raw data. In a simple linear regression of "
        f"average test scores on the student–teacher ratio, a one-student "
        f"increase in the ratio is associated with an estimated "
        f"{coef_simple:.2f}-point change in test scores "
        f"(standard error {se_simple:.2f}, p-value {p_simple:.3g}).\n\n"
        "To adjust for observable differences across districts, I then "
        "estimated a multiple regression of average test scores on the "
        "student–teacher ratio controlling for district income, the percentage "
        "of students who are English learners, the percentage qualifying for "
        "reduced-price lunch, and the percentage receiving CalWorks assistance. "
        f"In this model, a one-student increase in the student–teacher ratio "
        f"is associated with an estimated {coef_full:.2f}-point change in "
        f"average test scores (standard error {se_full:.2f}, p-value "
        f"{p_full:.3g}), and the model explains about {r2_full:.2f} of the "
        "variance in test scores (R-squared).\n\n"
        f"Overall, the controlled regression indicates that {direction}, and "
        f"this relationship is {sig_text}. {overall}\n\n"
        f"On a 0–100 scale where higher values represent a stronger 'Yes' "
        f"answer to the research question, I summarize the strength of the "
        f"evidence as {score}."
    )

    result = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def evidence_to_score(coef: float, p_value: float, corr: float) -> int:
    """
    Convert the strength and direction of evidence into a 0–100 Likert score.

    Higher scores correspond to stronger evidence that lower student–teacher
    ratios are associated with higher academic performance.
    """
    # Start from neutral
    score = 50.0

    # Adjust for direction and significance of the controlled regression
    if coef < 0:
        # Evidence in the hypothesized direction
        if p_value < 0.001:
            score = 90.0
        elif p_value < 0.01:
            score = 80.0
        elif p_value < 0.05:
            score = 70.0
        else:
            score = 60.0
    elif coef > 0:
        # Evidence against the hypothesized direction
        if p_value < 0.001:
            score = 10.0
        elif p_value < 0.01:
            score = 20.0
        elif p_value < 0.05:
            score = 30.0
        else:
            score = 40.0

    # Fine-tune using the raw correlation magnitude
    corr_strength = abs(corr)
    if corr_strength >= 0.6:
        score += 5.0 * np.sign(-coef)  # reinforce direction from regression
    elif corr_strength >= 0.3:
        score += 3.0 * np.sign(-coef)
    elif corr_strength >= 0.1:
        score += 1.0 * np.sign(-coef)

    # Clamp to [0, 100] and return as integer
    score = max(0.0, min(100.0, score))
    return int(round(score))


if __name__ == "__main__":
    main()

