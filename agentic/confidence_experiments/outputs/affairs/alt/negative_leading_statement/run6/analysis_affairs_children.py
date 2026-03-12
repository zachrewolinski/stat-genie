import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    # Binary indicator for presence of children in the marriage
    df["children_binary"] = (df["children"] == "yes").astype(int)

    # Basic descriptive statistics by children status
    group = df.groupby("children", observed=True)
    prevalence_by_children = group["any_affair"].mean()
    mean_affairs_by_children = group["affairs"].mean()
    count_by_children = group["affairs"].size()

    # Two-sample tests on the affairs count between groups
    affairs_with_children = df.loc[df["children"] == "yes", "affairs"]
    affairs_without_children = df.loc[df["children"] == "no", "affairs"]

    # Because the outcome is highly skewed and discrete, use both t-test and
    # Mann–Whitney U as robustness checks.
    ttest_res = stats.ttest_ind(
        affairs_with_children,
        affairs_without_children,
        equal_var=False,
    )
    mwu_res = stats.mannwhitneyu(
        affairs_with_children,
        affairs_without_children,
        alternative="two-sided",
    )

    # Logistic regression for having any affair, controlling for key covariates.
    # children_binary is the variable of interest.
    formula = (
        "any_affair ~ children_binary + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df)
    logit_result = logit_model.fit(disp=False, maxiter=200)

    coef_children = float(logit_result.params["children_binary"])
    pvalue_children = float(logit_result.pvalues["children_binary"])
    ci_children = logit_result.conf_int().loc["children_binary"].tolist()
    ci_low, ci_high = float(ci_children[0]), float(ci_children[1])
    odds_ratio = float(math.exp(coef_children))
    or_ci_low, or_ci_high = float(math.exp(ci_low)), float(math.exp(ci_high))

    # Map statistical evidence to a 0–100 Likert-style score where
    # 0 = strong "No" (children clearly do NOT decrease affairs, or increase them)
    # 100 = strong "Yes" (children clearly decrease affairs).
    response_score = score_evidence(coef_children, pvalue_children)

    explanation = build_explanation(
        prevalence_by_children=prevalence_by_children.to_dict(),
        mean_affairs_by_children=mean_affairs_by_children.to_dict(),
        count_by_children=count_by_children.to_dict(),
        ttest_stat=float(ttest_res.statistic),
        ttest_pvalue=float(ttest_res.pvalue),
        mwu_stat=float(mwu_res.statistic),
        mwu_pvalue=float(mwu_res.pvalue),
        coef_children=coef_children,
        pvalue_children=pvalue_children,
        ci_low=ci_low,
        ci_high=ci_high,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        response_score=response_score,
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


def score_evidence(coef_children: float, pvalue_children: float) -> int:
    """
    Convert the evidence about children -> affairs into a 0–100 Likert score.

    Interpretation for the research question:
    "Does having children decrease engagement in extramarital affairs?"

    - Scores > 50 indicate "Yes", with higher values = stronger evidence
      that having children is associated with *lower* affair engagement.
    - Scores < 50 indicate "No", with lower values = stronger evidence that
      having children do not decrease affairs or may even increase them.
    """
    # Non-significant results: treat as essentially "no clear evidence"
    # and give a mild "No" answer, leaning slightly according to the sign.
    if pvalue_children >= 0.05:
        if coef_children < 0:
            # Point estimate is protective but not statistically reliable.
            return 40
        elif coef_children > 0:
            # Point estimate suggests higher affair engagement with children,
            # but evidence is weak.
            return 30
        else:
            return 50

    # Statistically significant: strength depends on both p-value and effect sign.
    # Negative coefficient => odds ratio < 1: children associated with fewer affairs.
    if coef_children < 0:
        # Stronger significance pushes score upward.
        if pvalue_children < 0.001:
            base = 85
        elif pvalue_children < 0.01:
            base = 75
        else:
            base = 65

        # Adjust for how far the odds ratio is from 1.
        or_value = math.exp(coef_children)
        effect_strength = max(0.0, min(1.0, (1.0 - or_value) / 0.7))
        return int(round(base + effect_strength * (100 - base)))

    # Positive coefficient and significant: evidence that children are associated
    # with *more* affair engagement, i.e., a strong "No" to the question.
    if pvalue_children < 0.001:
        base = 10
    elif pvalue_children < 0.01:
        base = 15
    else:
        base = 20

    or_value = math.exp(coef_children)
    effect_strength = max(0.0, min(1.0, (or_value - 1.0) / 1.5))
    return int(round(max(0, base - effect_strength * base)))


def build_explanation(
    prevalence_by_children,
    mean_affairs_by_children,
    count_by_children,
    ttest_stat: float,
    ttest_pvalue: float,
    mwu_stat: float,
    mwu_pvalue: float,
    coef_children: float,
    pvalue_children: float,
    ci_low: float,
    ci_high: float,
    odds_ratio: float,
    or_ci_low: float,
    or_ci_high: float,
    response_score: int,
) -> str:
    # Helper to format percentages and means.
    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    yes_n = int(count_by_children.get("yes", 0))
    no_n = int(count_by_children.get("no", 0))

    prev_yes = float(prevalence_by_children.get("yes", np.nan))
    prev_no = float(prevalence_by_children.get("no", np.nan))

    mean_yes = float(mean_affairs_by_children.get("yes", np.nan))
    mean_no = float(mean_affairs_by_children.get("no", np.nan))

    direction_text = (
        "lower" if coef_children < 0 else "higher"
    )

    if pvalue_children < 0.001:
        sig_text = "highly statistically significant (p < 0.001)"
    elif pvalue_children < 0.01:
        sig_text = "statistically significant (p < 0.01)"
    elif pvalue_children < 0.05:
        sig_text = "statistically significant (p < 0.05)"
    else:
        sig_text = f"not statistically significant (p = {pvalue_children:.3f})"

    if response_score > 50:
        overall_answer = (
            "Yes – the data provide evidence that having children is "
            "associated with reduced engagement in extramarital affairs."
        )
    elif response_score < 50:
        overall_answer = (
            "No – the data do not support the claim that having children "
            "reduces engagement in extramarital affairs."
        )
    else:
        overall_answer = (
            "The data are inconclusive about whether having children reduces "
            "engagement in extramarital affairs."
        )

    explanation_parts = [
        "Research question: Does having children decrease engagement in extramarital affairs?",
        "",
        "Data and sample:",
        f"- The dataset contains 601 first-marriage respondents from the Psychology Today survey.",
        f"- Of these, {yes_n} report having children in the marriage and {no_n} report no children.",
        "",
        "Descriptive patterns:",
        f"- Prevalence of any affair: {pct(prev_yes)} among respondents with children vs {pct(prev_no)} among those without children.",
        f"- Mean affair score (higher values = more frequent affairs): "
        f"{mean_yes:.2f} with children vs {mean_no:.2f} without children.",
        "",
        "Group comparison tests on the affair score:",
        f"- Welch t-test comparing the mean affair score by children status: "
        f"t = {ttest_stat:.2f}, p = {ttest_pvalue:.3f}.",
        f"- Mann–Whitney U test (non-parametric): U = {mwu_stat:.2f}, p = {mwu_pvalue:.3f}.",
        "  These tests assess whether the overall distribution of affair frequency differs between "
        "marriages with and without children.",
        "",
        "Multivariable logistic regression:",
        "- Outcome: indicator for having at least one extramarital affair in the past year.",
        "- Predictors: having children (yes/no), gender, age, years married, religiousness, education, "
        "occupation, and self-rated marital happiness.",
        f"- Coefficient for having children on the log-odds scale: {coef_children:.3f} "
        f"(95% CI [{ci_low:.3f}, {ci_high:.3f}]), {sig_text}.",
        f"- Interpreted as an odds ratio, having children is associated with an odds ratio of "
        f"{odds_ratio:.2f} for having an affair (95% CI [{or_ci_low:.2f}, {or_ci_high:.2f}]), "
        f"i.e., {direction_text} odds of an affair for respondents with children relative to those without, "
        "after adjusting for the listed covariates.",
        "",
        "Synthesis:",
        "- The descriptive statistics show how affair prevalence and frequency differ between marriages "
        "with and without children.",
        "- The group comparison tests evaluate whether these differences are larger than would be expected "
        "by chance alone.",
        "- The logistic regression isolates the association of having children with affair involvement while "
        "controlling for demographic, socio-economic, and marital satisfaction factors.",
        "",
        f"Overall conclusion (scaled response = {response_score} on a 0–100 scale where 0 is a strong "
        '"No" and 100 is a strong "Yes"):',
        overall_answer,
    ]

    return "\n".join(explanation_parts)


if __name__ == "__main__":
    main()

