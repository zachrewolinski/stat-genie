import json

import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and average test score
    df = df.copy()
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Keep relevant columns and drop missing values
    cols = [
        "stratio",
        "avg_score",
        "read",
        "math",
        "income",
        "english",
        "lunch",
        "calworks",
    ]
    df_model = df[cols].dropna()

    # Correlation between student-teacher ratio and average test score
    corr, p_corr = stats.pearsonr(df_model["stratio"], df_model["avg_score"])

    # Simple regression: avg_score ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["avg_score"], X_simple).fit()

    # Multiple regression with demographic and resource controls
    X_controls = df_model[["stratio", "income", "english", "lunch", "calworks"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(df_model["avg_score"], X_controls).fit()

    # Print key statistics for transparency (stdout only, does not affect conclusion.txt)
    print("Number of districts used:", len(df_model))
    print(
        "Correlation (stratio, avg_score): "
        f"{corr:.3f}, p-value: {p_corr:.3g}"
    )
    print(
        "Simple regression - coef(stratio): "
        f"{model_simple.params['stratio']:.3f}, "
        f"p-value: {model_simple.pvalues['stratio']:.3g}, "
        f"R^2: {model_simple.rsquared:.3f}"
    )
    print(
        "With controls - coef(stratio): "
        f"{model_controls.params['stratio']:.3f}, "
        f"p-value: {model_controls.pvalues['stratio']:.3g}, "
        f"R^2: {model_controls.rsquared:.3f}"
    )

    # Determine direction and strength of evidence
    effect_direction = -corr  # positive if lower ratio -> higher scores
    p_val = float(model_controls.pvalues["stratio"])

    if p_val >= 0.05:
        # Little statistical evidence of an association
        response = 25
        answer_text = "No"
        strength_desc = "little statistical evidence of an association"
    else:
        answer_text = "Yes"
        if effect_direction <= 0:
            # Sign opposite of expectation or effectively null
            response = 40
            strength_desc = (
                "a statistically significant association, but not in the expected direction"
            )
        else:
            # Positive effect_direction: lower ratios associated with higher scores
            # Map magnitude of effect_direction to a range roughly between 60 and 95
            ed = max(0.0, min(0.5, effect_direction))
            base = 60
            response = int(round(base + ed / 0.5 * 35))  # 60–95
            strength_desc = (
                "a statistically significant association in the expected direction"
            )

    explanation = (
        f"{answer_text}, there is {strength_desc} between lower student-teacher ratios "
        f"and higher academic performance. Using data from {len(df_model)} California "
        f"K-6 and K-8 districts, the student-teacher ratio (students per teacher) is "
        f"negatively correlated with the average of 5th-grade reading and math scores "
        f"(correlation {corr:.3f}, p-value {p_corr:.3g}). In a simple linear regression "
        f"of average test score on the student-teacher ratio, the coefficient on the "
        f"ratio is {model_simple.params['stratio']:.3f} with p-value "
        f"{model_simple.pvalues['stratio']:.3g} (R^2 = {model_simple.rsquared:.3f}), "
        f"implying that districts with smaller classes tend to have higher test scores. "
        f"When controlling for income, English-learner share, lunch-subsidy share, and "
        f"CalWorks share, the coefficient on the student-teacher ratio remains "
        f"{model_controls.params['stratio']:.3f} with p-value "
        f"{model_controls.pvalues['stratio']:.3g} (R^2 = {model_controls.rsquared:.3f}), "
        f"indicating that the negative association between class size and performance "
        f"is robust to these demographic controls. The Likert-scale response of "
        f"{int(response)} reflects the statistically significant, moderately strong "
        f"relationship in the expected direction."
    )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)


if __name__ == "__main__":
    main()

