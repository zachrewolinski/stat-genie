import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Define outcome measures
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Basic group summaries by presence of children
    group_freq = df.groupby("feature6")["feature2"].agg(["mean", "std", "count"])
    group_affair = df.groupby("feature6")["has_affair"].mean()

    # Split groups
    freq_yes = df.loc[df["feature6"] == "yes", "feature2"]
    freq_no = df.loc[df["feature6"] == "no", "feature2"]

    # Two-sample t-test on affair frequency (Welch)
    t_stat, p_ttest = stats.ttest_ind(freq_yes, freq_no, equal_var=False)

    # Chi-square test on any affair vs children (2x2 table)
    contingency = pd.crosstab(df["feature6"], df["has_affair"])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)

    # Multivariable logistic regression on any affair
    # Children and gender treated as categorical; others numeric covariates
    formula = (
        "has_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)

    # Extract children effect
    child_param_name = None
    for name in logit_model.params.index:
        if "C(feature6)" in name:
            child_param_name = name
            break

    if child_param_name is None:
        raise RuntimeError("Could not find children parameter in logistic model.")

    child_coef = float(logit_model.params[child_param_name])
    child_pval = float(logit_model.pvalues[child_param_name])
    child_or = float(np.exp(child_coef))
    ci_low, ci_high = logit_model.conf_int().loc[child_param_name]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Directional indicators
    mean_yes = float(group_freq.loc["yes", "mean"])
    mean_no = float(group_freq.loc["no", "mean"])
    prop_yes = float(group_affair.loc["yes"])
    prop_no = float(group_affair.loc["no"])

    has_lower_mean = mean_yes < mean_no
    has_lower_prop = prop_yes < prop_no
    coef_negative = child_coef < 0

    # Significance flags
    logistic_sig = child_pval < 0.05
    ttest_sig = p_ttest < 0.05
    chi2_sig = p_chi2 < 0.05

    # Likert score mapping (0 = strong "No", 100 = strong "Yes")
    # Start slightly on the "No" side; increase when there is
    # consistent statistically significant evidence that children reduce engagement.
    score = 40

    # Logistic regression (primary evidence)
    if logistic_sig and coef_negative:
        score += 25
        # Stronger effect size -> higher score
        if child_or < 0.5:
            score += 10
        elif child_or < 0.8:
            score += 5
    elif logistic_sig and not coef_negative:
        score -= 25
        if child_or > 1.5:
            score -= 10
        elif child_or > 1.2:
            score -= 5

    # T-test on mean frequency
    if ttest_sig and has_lower_mean:
        score += 10
    elif ttest_sig and not has_lower_mean:
        score -= 10

    # Chi-square on any affair
    if chi2_sig and has_lower_prop:
        score += 10
    elif chi2_sig and not has_lower_prop:
        score -= 10

    # If nothing is statistically significant either way, reflect lack of evidence
    if not (logistic_sig or ttest_sig or chi2_sig):
        score = 30

    # Clamp score to [0, 100] and integer
    score = int(max(0, min(100, round(score))))

    # Build explanation
    explanation_lines = []

    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs "
        "(measured in the past year among 601 first-marriage respondents)?"
    )

    explanation_lines.append(
        f"Affair frequency (feature2) by children status: "
        f"mean={mean_yes:.3f} (children=yes, n={int(group_freq.loc['yes', 'count'])}) vs "
        f"mean={mean_no:.3f} (children=no, n={int(group_freq.loc['no', 'count'])})."
    )

    explanation_lines.append(
        f"Proportion with any affair (feature2>0): "
        f"{prop_yes:.3f} with children vs {prop_no:.3f} without children."
    )

    explanation_lines.append(
        f"Two-sample Welch t-test on affair frequency between groups: "
        f"t={t_stat:.3f}, p={p_ttest:.3g}."
    )

    explanation_lines.append(
        f"Chi-square test of independence between having children and any affair: "
        f"chi2={chi2:.3f}, p={p_chi2:.3g}."
    )

    direction_word = "lower" if child_or < 1 else "higher"
    significance_word = "statistically significant" if logistic_sig else "not statistically significant"

    explanation_lines.append(
        "Multivariable logistic regression of any affair (yes/no) on children, gender, age, years married, "
        "religiousness, education, occupation, and self-rated marital happiness shows that having children "
        f"is associated with an odds ratio of {child_or:.2f} for any affair "
        f"(95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, p={child_pval:.3g}), "
        f"indicating {direction_word} odds of an affair among those with children, but this effect is "
        f"{significance_word} at the 5% level."
    )

    if score >= 50 and child_or < 1:
        overall_sentence = (
            "Overall, I interpret the combined evidence as supporting the claim that having children "
            "does modestly reduce engagement in extramarital affairs, although the effect size and "
            "statistical significance should be interpreted with caution."
        )
    elif score >= 50 and child_or >= 1:
        overall_sentence = (
            "Overall, the statistical evidence weakly supports a relationship between children and affair "
            "engagement, but the direction appears to be higher rather than lower odds of affairs among "
            "those with children; this is not strong support for the hypothesis that children are protective."
        )
    elif child_or < 1:
        overall_sentence = (
            "Overall, while point estimates tend to show lower affair involvement among respondents with "
            "children, the differences are small and not consistently statistically significant; I therefore "
            "do not find strong evidence that having children meaningfully decreases engagement in "
            "extramarital affairs in this sample."
        )
    else:
        overall_sentence = (
            "Overall, I do not find evidence that having children decreases engagement in extramarital "
            "affairs; if anything, the data suggest similar or slightly higher affair involvement among "
            "respondents with children, though estimates are imprecise."
        )

    explanation_lines.append(overall_sentence)

    explanation_lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes' answer to the question "
        f'\"Does having children decrease engagement in extramarital affairs?\", I assign a score of {score}, '
        "reflecting the limited and at best modest evidence for such a protective effect in these data."
    )

    explanation = "\n".join(explanation_lines)

    # Write required JSON output
    output = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    # Also print key results for transparency when run interactively
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

