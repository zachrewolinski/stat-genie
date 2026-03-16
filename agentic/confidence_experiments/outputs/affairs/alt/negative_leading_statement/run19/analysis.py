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

    # Binary indicator for any extramarital affair.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status.
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    # Two-sample t-test on affair counts (Welch) between children groups.
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    t_stat, t_pvalue = stats.ttest_ind(
        affairs_yes, affairs_no, equal_var=False, nan_policy="omit"
    )

    # Chi-square test of independence for any_affair vs. children.
    contingency = pd.crosstab(df["children"], df["any_affair"])
    chi2_stat, chi2_pvalue, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression: any_affair on children alone.
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    coef_children_simple = logit_simple.params.get("C(children)[T.yes]", np.nan)
    p_children_simple = logit_simple.pvalues.get("C(children)[T.yes]", np.nan)
    or_children_simple = float(np.exp(coef_children_simple))

    # Logistic regression with controls.
    logit_formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(gender) + occupation + rating"
    )
    logit_full = smf.logit(logit_formula, data=df).fit(disp=False)
    coef_children_full = logit_full.params.get("C(children)[T.yes]", np.nan)
    p_children_full = logit_full.pvalues.get("C(children)[T.yes]", np.nan)
    or_children_full = float(np.exp(coef_children_full))

    # Poisson regression for affair counts with same controls.
    poisson_formula = (
        "affairs ~ C(children) + age + yearsmarried + religiousness + "
        "education + C(gender) + occupation + rating"
    )
    poisson_model = smf.glm(
        poisson_formula, data=df, family=sm.families.Poisson()
    ).fit(cov_type="HC3")
    coef_children_pois = poisson_model.params.get("C(children)[T.yes]", np.nan)
    p_children_pois = poisson_model.pvalues.get("C(children)[T.yes]", np.nan)
    rr_children_pois = float(np.exp(coef_children_pois))

    # Summarize key numeric findings for the explanation.
    stats_summary = {
        "group_stats": group_stats.to_dict(orient="records"),
        "t_test": {"t_stat": float(t_stat), "p_value": float(t_pvalue)},
        "chi2_test": {"chi2_stat": float(chi2_stat), "p_value": float(chi2_pvalue)},
        "logit_simple": {
            "coef_children": float(coef_children_simple),
            "p_value_children": float(p_children_simple),
            "odds_ratio_children": or_children_simple,
        },
        "logit_full": {
            "coef_children": float(coef_children_full),
            "p_value_children": float(p_children_full),
            "odds_ratio_children": or_children_full,
        },
        "poisson_full": {
            "coef_children": float(coef_children_pois),
            "p_value_children": float(p_children_pois),
            "rate_ratio_children": rr_children_pois,
        },
    }

    # Determine qualitative conclusion and Likert-scale response.
    # Research question: "Does having children decrease engagement in extramarital affairs?"
    # 0 = strong "No", 100 = strong "Yes".

    # We weigh:
    # - direction of the children effect in models,
    # - statistical significance across tests,
    # - simple group differences.
    mean_affairs_yes = float(
        group_stats.loc[group_stats["children"] == "yes", "mean_affairs"].iloc[0]
    )
    mean_affairs_no = float(
        group_stats.loc[group_stats["children"] == "no", "mean_affairs"].iloc[0]
    )
    prop_any_yes = float(
        group_stats.loc[group_stats["children"] == "yes", "prop_any_affair"].iloc[0]
    )
    prop_any_no = float(
        group_stats.loc[group_stats["children"] == "no", "prop_any_affair"].iloc[0]
    )

    # By design, if children truly decreased affairs we would expect:
    # means and proportions LOWER for "yes" than "no", and negative/significant
    # coefficients in the regression models.
    evidence_for_decrease = []
    evidence_against_decrease = []

    # Descriptive differences.
    if mean_affairs_yes < mean_affairs_no and prop_any_yes < prop_any_no:
        evidence_for_decrease.append("lower_means_and_proportions")
    else:
        evidence_against_decrease.append("descriptives_not_lower_for_children")

    # Logistic models.
    if coef_children_simple < 0 and p_children_simple < 0.05:
        evidence_for_decrease.append("logit_simple_negative_significant")
    elif coef_children_simple >= 0 and p_children_simple < 0.05:
        evidence_against_decrease.append("logit_simple_positive_significant")

    if coef_children_full < 0 and p_children_full < 0.05:
        evidence_for_decrease.append("logit_full_negative_significant")
    elif coef_children_full >= 0 and p_children_full < 0.05:
        evidence_against_decrease.append("logit_full_positive_significant")

    # Poisson model.
    if coef_children_pois < 0 and p_children_pois < 0.05:
        evidence_for_decrease.append("poisson_negative_significant")
    elif coef_children_pois >= 0 and p_children_pois < 0.05:
        evidence_against_decrease.append("poisson_positive_significant")

    # Infer response score.
    if evidence_for_decrease and not evidence_against_decrease:
        # Consistent evidence that children are associated with fewer affairs.
        response = 75
        qualitative = "Yes"
    elif evidence_for_decrease and evidence_against_decrease:
        # Mixed or model-dependent evidence.
        response = 50
        qualitative = "Unclear"
    else:
        # No evidence that children decrease affairs; descriptives and/or models
        # fail to show a negative effect, and may even suggest more affairs among
        # parents.
        response = 20
        qualitative = "No"

    # Build human-readable explanation.
    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        f"Descriptively, individuals with children report on average {mean_affairs_yes:.2f} affair-units "
        f"versus {mean_affairs_no:.2f} for those without children."
    )
    explanation_lines.append(
        f"The proportion with any affair in the past year is {prop_any_yes:.2%} among those with children "
        f"and {prop_any_no:.2%} among those without."
    )
    explanation_lines.append(
        f"A Welch two-sample t-test comparing affair counts between parents and non-parents yields "
        f"t = {t_stat:.2f}, p = {t_pvalue:.3f}, providing {'little' if t_pvalue >= 0.05 else 'statistically significant'} "
        "evidence of a difference in average affair counts."
    )
    explanation_lines.append(
        f"A chi-square test of independence between having children and any affair gives "
        f"chi² = {chi2_stat:.2f}, p = {chi2_pvalue:.3f}, "
        f"{'not indicating a strong association' if chi2_pvalue >= 0.05 else 'indicating a statistically significant association'}."
    )
    explanation_lines.append(
        "In a logistic regression predicting the probability of any affair using only the children indicator, "
        f"the coefficient for having children corresponds to an odds ratio of {or_children_simple:.2f} "
        f"(p = {p_children_simple:.3f})."
    )
    explanation_lines.append(
        "In a richer logistic model controlling for age, years married, religiousness, education, gender, "
        f"occupation, and marital rating, the children coefficient implies an odds ratio of {or_children_full:.2f} "
        f"(p = {p_children_full:.3f})."
    )
    explanation_lines.append(
        "A Poisson regression for the number of affairs with the same controls yields a rate ratio of "
        f"{rr_children_pois:.2f} for parents versus non-parents (p = {p_children_pois:.3f})."
    )
    if qualitative == "Yes":
        explanation_lines.append(
            "Across these analyses, having children is consistently associated with fewer affairs and the "
            "children effect is statistically significant in multiple models, providing evidence that parents "
            "engage less in extramarital affairs in this sample."
        )
    elif qualitative == "Unclear":
        explanation_lines.append(
            "The direction and significance of the children effect are not consistent across models: some "
            "analyses suggest fewer affairs among parents, while others do not or even point in the opposite "
            "direction. Overall, the evidence does not allow a confident conclusion that children clearly "
            "decrease engagement in extramarital affairs."
        )
    else:
        explanation_lines.append(
            "Overall, the descriptive comparisons and regression models do not provide convincing evidence "
            "that having children decreases engagement in extramarital affairs; in this dataset, parents are "
            "not clearly less likely to have affairs and may in some specifications even appear slightly more "
            "at risk. Thus, the data support a 'No' answer to the research question."
        )

    explanation_lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"this analysis corresponds to a {qualitative} answer with score {response}, "
        "reflecting limited evidence that children reduce extramarital affairs in this sample."
    )

    full_explanation = "\n".join(explanation_lines) + "\n\nDetails of key statistics:\n"
    full_explanation += json.dumps(stats_summary, indent=2)

    conclusion = {"response": int(response), "explanation": full_explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

