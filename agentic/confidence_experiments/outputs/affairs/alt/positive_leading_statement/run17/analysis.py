import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Binary indicator: any extramarital affair in the last year.
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status.
    group_children = df.groupby("children", observed=True)
    mean_affairs = group_children["affairs"].mean()
    prop_has_affair = group_children["has_affair"].mean()

    # Fit logistic regression for probability of any affair.
    # Model: has_affair ~ children + age + yearsmarried + religiousness
    #                      + education + occupation + rating + gender
    logit_formula = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )

    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

    # Also fit a linear model on the affair frequency as a secondary check.
    ols_formula = (
        "affairs ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    ols_model = smf.ols(ols_formula, data=df).fit()

    # Extract effect of having children (children == 'yes')
    # Reference category is children == 'no'.
    coeff_name = "C(children)[T.yes]"
    if coeff_name not in logit_model.params.index:
        raise KeyError(f"{coeff_name} not found in logistic model parameters.")

    logit_coef = float(logit_model.params[coeff_name])
    logit_pval = float(logit_model.pvalues[coeff_name])

    ols_coef = float(ols_model.params.get(coeff_name, np.nan))
    ols_pval = float(ols_model.pvalues.get(coeff_name, np.nan))

    # Descriptive differences: children 'yes' minus 'no'
    # (negative values indicate fewer affairs among those with children).
    mean_affairs_yes = float(mean_affairs.get("yes", np.nan))
    mean_affairs_no = float(mean_affairs.get("no", np.nan))
    diff_mean_affairs = mean_affairs_yes - mean_affairs_no

    prop_affair_yes = float(prop_has_affair.get("yes", np.nan))
    prop_affair_no = float(prop_has_affair.get("no", np.nan))
    diff_prop_affair = prop_affair_yes - prop_affair_no

    # Map statistical evidence to a 0–100 response scale where
    # 0 = strong "No", 100 = strong "Yes" to:
    # "Does having children decrease engagement in extramarital affairs?"
    #
    # We treat "Yes" as: children associated with lower odds/frequency
    # of affairs (negative coefficients, lower means),
    # and "No" otherwise. The scale reflects strength and consistency.
    if logit_coef < 0:
        # Directionally consistent with "children decrease affairs".
        if logit_pval < 0.01:
            base_response = 85
        elif logit_pval < 0.05:
            base_response = 75
        elif logit_pval < 0.10:
            base_response = 65
        else:
            base_response = 55
    else:
        # Coefficient zero/positive: no evidence or opposite direction.
        if logit_pval < 0.01:
            base_response = 10
        elif logit_pval < 0.05:
            base_response = 20
        elif logit_pval < 0.10:
            base_response = 30
        else:
            base_response = 45

    # Adjust slightly based on linear model and descriptive means.
    # If all three indicators agree that children reduce affairs,
    # nudge the response upward; if they agree in the opposite direction,
    # nudge downward.
    indicators_decrease = 0
    indicators_increase = 0

    if np.isfinite(ols_coef):
        if ols_coef < 0:
            indicators_decrease += 1
        elif ols_coef > 0:
            indicators_increase += 1

    if np.isfinite(diff_mean_affairs):
        if diff_mean_affairs < 0:
            indicators_decrease += 1
        elif diff_mean_affairs > 0:
            indicators_increase += 1

    if np.isfinite(diff_prop_affair):
        if diff_prop_affair < 0:
            indicators_decrease += 1
        elif diff_prop_affair > 0:
            indicators_increase += 1

    response = base_response
    if indicators_decrease >= 2:
        response += 5
    elif indicators_increase >= 2:
        response -= 5

    # Clamp to [0, 100] and convert to int.
    response = int(min(100, max(0, round(response))))

    # Interpret the numerical response as an explicit Yes/No statement.
    yes_no_answer = "Yes" if response >= 50 else "No"

    explanation_lines = [
        "Research question: Does having children decrease engagement in extramarital affairs?",
        "",
        f"Descriptive statistics (means by children status):",
        f"- Mean affair frequency (children = yes): {mean_affairs_yes:.3f}",
        f"- Mean affair frequency (children = no): {mean_affairs_no:.3f}",
        f"- Difference in mean frequency (yes - no): {diff_mean_affairs:.3f}",
        f"- Proportion with any affair (children = yes): {prop_affair_yes:.3f}",
        f"- Proportion with any affair (children = no): {prop_affair_no:.3f}",
        f"- Difference in proportions (yes - no): {diff_prop_affair:.3f}",
        "",
        "Logistic regression on any affair (has_affair):",
        f"- Model: has_affair ~ children + age + yearsmarried + religiousness + "
        f"education + occupation + rating + gender",
        f"- Coefficient for children = yes (vs no): {logit_coef:.3f}",
        f"- p-value for this coefficient: {logit_pval:.4f}",
        "",
        "Linear regression on affair frequency (affairs):",
        f"- Coefficient for children = yes (vs no): {ols_coef:.3f}",
        f"- p-value for this coefficient: {ols_pval:.4f}",
        "",
    ]

    if yes_no_answer == "Yes":
        interpretation = (
            "Interpretation: The negative association between having children and "
            "extramarital affairs suggests that, after adjusting for age, years "
            "married, religiousness, education, occupation, marital satisfaction, "
            "and gender, individuals with children tend to show lower engagement "
            "in extramarital affairs. The strength of this evidence is reflected "
            f"in the response score of {response} on a 0–100 scale, where higher "
            "values represent stronger support for a 'Yes' answer."
        )
    else:
        interpretation = (
            "Interpretation: The statistical analysis does not provide convincing "
            "evidence that having children decreases engagement in extramarital "
            "affairs. The estimated effects are small, statistically weak, or even "
            "point toward equal or greater engagement among those with children "
            "once we adjust for age, years married, religiousness, education, "
            "occupation, marital satisfaction, and gender. The response score of "
            f"{response} on a 0–100 scale therefore reflects a 'No' answer or at "
            "most very weak evidence in favor of a decrease."
        )

    explanation_lines.append(interpretation)

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    output_path = Path("conclusion.txt")
    output_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

