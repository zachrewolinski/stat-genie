import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children
    desc = (
        df.groupby("children")
        .agg(
            n=("affairs", "size"),
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("has_affair", "mean"),
        )
        .reset_index()
    )

    # Two-sample tests for differences by children
    children_yes = df[df["children"] == "yes"]
    children_no = df[df["children"] == "no"]

    # Affairs as a count-like variable: Welch t-test on means
    t_stat, t_pvalue = stats.ttest_ind(
        children_yes["affairs"],
        children_no["affairs"],
        equal_var=False,
    )

    # Proportion test for any affair
    count_yes = int(children_yes["has_affair"].sum())
    count_no = int(children_no["has_affair"].sum())
    n_yes = int(children_yes.shape[0])
    n_no = int(children_no.shape[0])
    prop_yes = count_yes / n_yes if n_yes > 0 else np.nan
    prop_no = count_no / n_no if n_no > 0 else np.nan

    # Approximate z-test for two proportions
    pooled = (count_yes + count_no) / (n_yes + n_no)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_yes + 1 / n_no))
    z_stat = (prop_yes - prop_no) / se
    # Two-sided p-value
    z_pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Logistic regression for having any affair, controlling for covariates
    # Use children=='no' as baseline by letting statsmodels handle coding.
    logit_formula = (
        "has_affair ~ C(children) + age + yearsmarried + rating "
        "+ religiousness + education + occupation + C(gender)"
    )
    logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)

    # Extract coefficient for having children (yes vs no baseline)
    coef_children = None
    pvalue_children = None
    odds_ratio = None
    for term, coef in logit_model.params.items():
        if term.startswith("C(children)[T.yes]"):
            coef_children = coef
            pvalue_children = logit_model.pvalues[term]
            odds_ratio = float(np.exp(coef_children))
            break

    # Also compute marginal probabilities of any affair by children group
    mean_prob_by_children = (
        df.groupby("children")["has_affair"].mean().to_dict()
    )

    # Decide Likert-style response (0-100) and construct explanation
    explanation_lines = []

    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        "I analyzed 601 married individuals from the Fair (1978) affairs dataset."
    )
    explanation_lines.append(
        "Descriptively, I compared frequency of affairs and the proportion reporting any affair between marriages with and without children."
    )

    # Descriptive summary text
    for _, row in desc.iterrows():
        explanation_lines.append(
            f"For marriages with children = {row['children']}, "
            f"n = {int(row['n'])}, mean coded affair frequency = {row['mean_affairs']:.2f}, "
            f"median = {row['median_affairs']:.1f}, "
            f"and proportion with any affair = {row['prop_any_affair']:.3f}."
        )

    explanation_lines.append(
        "A Welch two-sample t-test comparing mean coded affair frequency between groups "
        f"gave t = {t_stat:.2f} with p = {t_pvalue:.4f}."
    )
    explanation_lines.append(
        "A two-sample test of proportions for having any affair gave "
        f"z = {z_stat:.2f} with p = {z_pvalue:.4f}."
    )

    # Regression results explanation
    if coef_children is not None:
        explanation_lines.append(
            "I then fit a logistic regression for having any affair "
            "with predictors: children, age, years married, marital rating, "
            "religiousness, education, occupation, and gender."
        )
        explanation_lines.append(
            f"The coefficient for having children (yes vs no) was {coef_children:.3f}, "
            f"with p = {pvalue_children:.4f}, corresponding to an odds ratio of {odds_ratio:.2f}."
        )

    explanation_lines.append(
        "Observed proportions of any affair by group were: "
        + ", ".join(
            f"{k} = {v:.3f}" for k, v in mean_prob_by_children.items()
        )
        + "."
    )

    # Map findings to a 0-100 scale where higher = stronger 'Yes'
    # (children decrease engagement in affairs).
    #
    # Use a simple rule based on direction and significance of the
    # children coefficient in the logistic regression, complemented
    # by descriptive differences.
    if coef_children is not None and pvalue_children < 0.05:
        if coef_children < 0:
            # Statistically significant protective association
            # Calibrate strength by effect size and descriptive gap.
            diff_prop = prop_no - prop_yes
            magnitude = abs(coef_children)
            # Start from a strong-yes baseline.
            score = 80
            if magnitude > 0.5:
                score += 5
            if magnitude > 1.0:
                score += 5
            if diff_prop > 0.10:
                score += 5
            score = max(60, min(100, int(round(score))))
            explanation_lines.append(
                "In the adjusted logistic model, having children is associated "
                "with significantly lower odds of reporting any affair. "
                "Together with the descriptive differences, this supports a "
                "Yes answer to the research question."
            )
        else:
            # Significant effect but in the opposite direction
            diff_prop = prop_yes - prop_no
            magnitude = abs(coef_children)
            score = 20
            if magnitude > 0.5:
                score -= 5
            if magnitude > 1.0:
                score -= 5
            if diff_prop > 0.10:
                score -= 5
            score = max(0, min(40, int(round(score))))
            explanation_lines.append(
                "In the adjusted logistic model, having children is associated "
                "with significantly higher odds of reporting any affair. "
                "This contradicts the claim that children decrease affairs."
            )
    else:
        # No statistically clear association in the regression
        if np.isnan(prop_yes) or np.isnan(prop_no):
            score = 50
            explanation_lines.append(
                "Group-specific proportions could not be reliably estimated, "
                "so the evidence is inconclusive."
            )
        else:
            diff_prop = abs(prop_yes - prop_no)
            if diff_prop < 0.03:
                score = 40
                explanation_lines.append(
                    "Neither the regression nor the descriptive differences "
                    "show a meaningful difference in affairs by children status. "
                    "This suggests little evidence that having children changes "
                    "engagement in extramarital affairs."
                )
            else:
                score = 50
                explanation_lines.append(
                    "The descriptive differences in affairs by children status "
                    "are modest and not clearly statistically significant after "
                    "adjustment, so the evidence is mixed."
                )

    result = {
        "response": int(score),
        "explanation": " ".join(explanation_lines),
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

