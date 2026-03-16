import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator of any extramarital affair in the past year.
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by presence of children.
    group_means = df.groupby("children")["affairs"].mean()
    group_props = df.groupby("children")["has_affair"].mean()

    # Logistic regression for probability of any affair.
    logit_formula = (
        "has_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

    # Poisson regression for count of affairs.
    poisson_formula = (
        "affairs ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    poisson_model = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
    ).fit()

    # Extract key statistics for the children effect (yes vs no).
    coef_child_logit = float(logit_model.params.get("C(children)[T.yes]", np.nan))
    p_child_logit = float(logit_model.pvalues.get("C(children)[T.yes]", np.nan))
    or_child_logit = float(np.exp(coef_child_logit)) if np.isfinite(coef_child_logit) else np.nan
    ci_logit = logit_model.conf_int().loc["C(children)[T.yes]"]
    or_ci_low, or_ci_high = np.exp(ci_logit[0]), np.exp(ci_logit[1])

    coef_child_pois = float(poisson_model.params.get("C(children)[T.yes]", np.nan))
    p_child_pois = float(poisson_model.pvalues.get("C(children)[T.yes]", np.nan))
    rr_child_pois = float(np.exp(coef_child_pois)) if np.isfinite(coef_child_pois) else np.nan

    # Map the statistical evidence to a 0-100 Likert-style score.
    response_score = map_evidence_to_scale(
        or_child_logit=or_child_logit,
        p_child_logit=p_child_logit,
        rr_child_pois=rr_child_pois,
        p_child_pois=p_child_pois,
    )

    # Build human-readable explanation with key numerical evidence.
    mean_affairs_children_no = float(group_means.get("no", np.nan))
    mean_affairs_children_yes = float(group_means.get("yes", np.nan))
    prop_affair_children_no = float(group_props.get("no", np.nan))
    prop_affair_children_yes = float(group_props.get("yes", np.nan))

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"- Sample size: {len(df)} currently married individuals.\n"
        "- Outcome variable: number of extramarital sexual encounters in the past year "
        "(`affairs`), with a derived binary indicator of any affair.\n"
        "- Key predictor: `children` (yes/no for whether there are children in the marriage).\n"
        "\n"
        "Descriptive patterns:\n"
        f"- Mean number of affairs (children = no): {mean_affairs_children_no:.3f}.\n"
        f"- Mean number of affairs (children = yes): {mean_affairs_children_yes:.3f}.\n"
        f"- Proportion with any affair (children = no): {prop_affair_children_no:.3f}.\n"
        f"- Proportion with any affair (children = yes): {prop_affair_children_yes:.3f}.\n"
        "These summaries indicate how average involvement in affairs differs between couples "
        "with and without children.\n"
        "\n"
        "Inferential analysis (adjusting for age, years married, religiousness, education, "
        "occupation, and self-rated marriage quality):\n"
        "- Logistic regression for having any affair uses `children` as a predictor alongside "
        "the covariates above.\n"
        f"- Odds ratio for having an affair if there are children vs no children: "
        f"{or_child_logit:.3f} (95% CI {or_ci_low:.3f} to {or_ci_high:.3f}, "
        f"p-value = {p_child_logit:.4f}).\n"
        "- Poisson regression for the count of affairs gives a rate ratio for `children` "
        f"of {rr_child_pois:.3f} (p-value = {p_child_pois:.4f}).\n"
        "\n"
        "Interpretation:\n"
        "The logistic and Poisson models jointly quantify how the presence of children is "
        "associated with both the likelihood and frequency of affairs after controlling for "
        "other important characteristics of the marriage and spouses. The odds ratio and "
        "rate ratio summarize both the direction (greater or reduced involvement in affairs) "
        "and the magnitude of that association, while the p-values indicate how unlikely the "
        "observed effects would be if children had no true impact.\n"
        "\n"
        "Overall conclusion underlying the Likert-scale response above:\n"
        "Based on the direction, magnitude, and statistical significance of the adjusted "
        "effects for `children`, together with the descriptive differences between groups, "
        "the evidence is used to judge whether having children meaningfully decreases "
        "engagement in extramarital affairs and how strong that evidence is on a 0–100 scale "
        "where 0 is a strong 'No' and 100 a strong 'Yes'."
    )

    result = {
        "response": int(response_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def map_evidence_to_scale(
    *,
    or_child_logit: float,
    p_child_logit: float,
    rr_child_pois: float,
    p_child_pois: float,
) -> int:
    """
    Map effect direction, size, and significance for `children` to a 0–100 score.

    0 means strong evidence that having children does not decrease affairs (or increases them),
    100 means strong evidence that having children does decrease affairs.
    """
    # Default to indeterminate if we fail to estimate anything.
    if not np.isfinite(or_child_logit) or not np.isfinite(p_child_logit):
        return 50

    # Base score from logistic regression (primary model on any affair).
    if p_child_logit <= 0.01:
        if or_child_logit <= 0.6:
            base = 90
        elif or_child_logit <= 0.8:
            base = 80
        elif or_child_logit < 1.0:
            base = 70
        elif or_child_logit <= 1.2:
            base = 40
        else:
            base = 20
    elif p_child_logit <= 0.05:
        if or_child_logit <= 0.6:
            base = 85
        elif or_child_logit <= 0.8:
            base = 75
        elif or_child_logit < 1.0:
            base = 65
        elif or_child_logit <= 1.2:
            base = 45
        else:
            base = 30
    elif p_child_logit <= 0.10:
        if or_child_logit < 1.0:
            base = 60
        elif or_child_logit <= 1.2:
            base = 40
        else:
            base = 30
    else:
        # Not statistically significant at conventional levels.
        if or_child_logit < 0.9:
            # Suggestive but not conclusive downward association.
            base = 55
        elif or_child_logit <= 1.1:
            # Essentially no clear effect.
            base = 50
        else:
            # Suggestive of an increase in affairs when children are present.
            base = 40

    # Adjust based on Poisson (count) model if available and consistent.
    if np.isfinite(rr_child_pois) and np.isfinite(p_child_pois):
        if rr_child_pois < 1.0 and p_child_pois <= 0.05:
            # Both models indicate fewer affairs with children.
            base += 5
        elif rr_child_pois > 1.0 and p_child_pois <= 0.05:
            # Count model contradicts the direction of the logistic result.
            base -= 5

    # Clamp to [0, 100] and return as int.
    base = max(0, min(100, base))
    return int(round(base))


if __name__ == "__main__":
    main()

