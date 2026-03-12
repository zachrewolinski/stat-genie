import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # Based on info.json descriptions, columns are mislabelled:
    # - 'age' column actually stores affair frequency in the past year.
    # - 'religiousness' column is a yes/no indicator: are there children in the marriage?
    # We interpret:
    #   affair_freq: numeric frequency of extramarital intercourse (0 = none, higher = more)
    #   has_children: 1 if there are children, 0 otherwise.
    affair_freq = df["age"]
    has_children = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop rows with missing values in key variables, if any.
    mask = has_children.notna() & affair_freq.notna()
    affair_freq = affair_freq[mask]
    has_children = has_children[mask]

    # Create a binary indicator for any affair vs none, which is a natural mapping
    # for the research question.
    any_affair = (affair_freq > 0).astype(int)

    # 1) Descriptive statistics by children status
    desc = (
        pd.DataFrame(
            {
                "n": affair_freq.groupby(has_children).size(),
                "mean_freq": affair_freq.groupby(has_children).mean(),
                "median_freq": affair_freq.groupby(has_children).median(),
                "prop_any_affair": any_affair.groupby(has_children).mean(),
            }
        )
        .rename(index={0: "no_children", 1: "children"})
        .to_dict(orient="index")
    )

    # 2) Mean difference in frequency: two-sample t-test (Welch)
    freq_children = affair_freq[has_children == 1]
    freq_no_children = affair_freq[has_children == 0]
    t_stat, p_ttest = stats.ttest_ind(
        freq_children, freq_no_children, equal_var=False, nan_policy="omit"
    )

    # 3) Difference in probability of any affair: chi-squared test of independence
    contingency = pd.crosstab(has_children, any_affair)
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # 4) Logistic regression of any affair on children indicator
    X = sm.add_constant(has_children.values.astype(float))
    y = any_affair.values
    logit_model = sm.Logit(y, X).fit(disp=False)
    coef_children = logit_model.params[1]
    p_children = logit_model.pvalues[1]
    odds_ratio = float(np.exp(coef_children))

    # Summarize statistical evidence
    evidence = {
        "descriptives": desc,
        "ttest": {"t_stat": float(t_stat), "p_value": float(p_ttest)},
        "chi2": {"chi2": float(chi2), "p_value": float(p_chi2)},
        "logit": {
            "coef_children": float(coef_children),
            "p_value_children": float(p_children),
            "odds_ratio_children": odds_ratio,
        },
    }

    # Determine Likert-scale response and narrative based on significance and direction.
    alpha = 0.05

    # Direction: odds_ratio < 1 implies children associated with *lower* odds of any affair.
    decreases_affairs = odds_ratio < 1

    # Strength of evidence: combine p-values from chi-squared and logistic regression.
    p_vals = [p_chi2, p_children]
    p_min = min(p_vals)

    if decreases_affairs and p_min < alpha:
        # Statistically significant decrease.
        # Scale strength inversely with p-value and effect size.
        # For very strong evidence (p < 0.001), push toward 90+.
        if p_min < 0.001:
            base = 90
        elif p_min < 0.01:
            base = 80
        else:
            base = 70

        # Adjust slightly by distance of odds_ratio from 1.
        effect_strength = min(0.2, abs(1 - odds_ratio))  # cap effect contribution
        response_score = int(round(base + effect_strength * 50))  # up to ~100
        response_score = max(0, min(100, response_score))
        yes_no_text = "Yes"
    elif not decreases_affairs and p_min < alpha:
        # Significant increase or no decrease.
        # For this research question, that corresponds to a strong \"No\".
        if p_min < 0.001:
            base = 10
        elif p_min < 0.01:
            base = 20
        else:
            base = 30
        effect_strength = min(0.2, abs(1 - odds_ratio))
        response_score = int(round(base - effect_strength * 50))
        response_score = max(0, min(100, response_score))
        yes_no_text = "No"
    else:
        # No consistent statistically significant effect.
        # Answer leans \"No\" with moderate confidence.
        response_score = 30
        yes_no_text = "No"

    # Build human-readable explanation using the evidence
    expl_parts = []
    expl_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    expl_parts.append(
        "I used the survey data where the 'age' column encodes the frequency of extramarital intercourse in the past year and the 'religiousness' column indicates whether there are children in the marriage (yes/no). I constructed a binary outcome for having any affair (frequency > 0)."  # noqa: E501
    )
    expl_parts.append(
        (
            f"Descriptively, among couples without children (n={int(desc['no_children']['n'])}), "
            f"the mean affair frequency was {desc['no_children']['mean_freq']:.2f} with "
            f"{desc['no_children']['prop_any_affair']*100:.1f}% engaging in at least one affair. "
            f"Among couples with children (n={int(desc['children']['n'])}), the mean affair frequency "
            f"was {desc['children']['mean_freq']:.2f} with "
            f"{desc['children']['prop_any_affair']*100:.1f}% engaging in at least one affair."
        )
    )
    expl_parts.append(
        (
            f"A Welch two-sample t-test comparing affair frequency by children status yielded "
            f"t = {t_stat:.2f} with p = {p_ttest:.4f}. "
            f"A chi-squared test of independence between children status and any affair produced "
            f"chi2 = {chi2:.2f} with p = {p_chi2:.4f}."
        )
    )
    if decreases_affairs:
        direction_text = "having children is associated with lower odds of any affair"
    else:
        direction_text = (
            "having children is not associated with lower odds of any affair "
            "(odds are similar or higher)"
        )

    expl_parts.append(
        (
            "In a logistic regression of any affair on the children indicator, the coefficient "
            f"for having children was {coef_children:.3f} (odds ratio = {odds_ratio:.2f}, "
            f"p = {p_children:.4f}), indicating that {direction_text}."
        )
    )
    expl_parts.append(
        (
            "Combining these results, the statistical evidence "
            f"{'does' if decreases_affairs and p_min < alpha else 'does not'} support the claim "
            "that having children decreases engagement in extramarital affairs. "
            f"I therefore answer '{yes_no_text}' to the research question and map this to a "
            f"Likert-scale response of {response_score} on a 0–100 scale, where higher values "
            "represent stronger support for 'Yes'."
        )
    )

    explanation = " ".join(expl_parts)

    print("RESPONSE_SCORE", response_score)
    print("EXPLANATION_START")  # marker to capture the full explanation text
    print(explanation)

    # Also write the required JSON output file.
    conclusion = {"response": int(response_score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
