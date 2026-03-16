import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def run_analysis():
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Split by children
    has_children = df[df["children"] == "yes"]
    no_children = df[df["children"] == "no"]

    # Basic descriptives
    mean_affairs_children = has_children["affairs"].mean()
    mean_affairs_no_children = no_children["affairs"].mean()

    prop_affair_children = has_children["any_affair"].mean()
    prop_affair_no_children = no_children["any_affair"].mean()

    # Difference in mean number of affairs (Welch t-test)
    t_affairs_stat, t_affairs_p = stats.ttest_ind(
        has_children["affairs"],
        no_children["affairs"],
        equal_var=False,
    )

    # Difference in probability of any affair (two-proportion z-test)
    counts = np.array(
        [has_children["any_affair"].sum(), no_children["any_affair"].sum()]
    )
    nobs = np.array([has_children.shape[0], no_children.shape[0]])
    z_prop_stat, z_prop_p = proportions_ztest(counts, nobs)

    # Logistic regression: any_affair ~ children (unadjusted)
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    coef_child_simple = logit_simple.params.get("C(children)[T.yes]", np.nan)
    p_child_simple = logit_simple.pvalues.get("C(children)[T.yes]", np.nan)
    or_child_simple = float(np.exp(coef_child_simple)) if np.isfinite(coef_child_simple) else np.nan

    # Logistic regression with controls
    formula_full = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)
    coef_child_full = logit_full.params.get("C(children)[T.yes]", np.nan)
    p_child_full = logit_full.pvalues.get("C(children)[T.yes]", np.nan)
    or_child_full = float(np.exp(coef_child_full)) if np.isfinite(coef_child_full) else np.nan

    results = {
        "mean_affairs_children": float(mean_affairs_children),
        "mean_affairs_no_children": float(mean_affairs_no_children),
        "prop_affair_children": float(prop_affair_children),
        "prop_affair_no_children": float(prop_affair_no_children),
        "t_affairs_stat": float(t_affairs_stat),
        "t_affairs_p": float(t_affairs_p),
        "z_prop_stat": float(z_prop_stat),
        "z_prop_p": float(z_prop_p),
        "coef_child_simple": float(coef_child_simple),
        "p_child_simple": float(p_child_simple),
        "or_child_simple": or_child_simple,
        "coef_child_full": float(coef_child_full),
        "p_child_full": float(p_child_full),
        "or_child_full": or_child_full,
        "n_children": int(has_children.shape[0]),
        "n_no_children": int(no_children.shape[0]),
        "n_total": int(df.shape[0]),
    }

    response, explanation = interpret_results(results)
    return int(response), explanation


def interpret_results(res: dict) -> tuple[int, str]:
    """
    Map statistical evidence into a 0-100 Likert score and explanation.

    Positive answer ("Yes") means: having children decreases engagement
    in extramarital affairs (lower probability and/or frequency).
    """
    # Primary evidence: logistic regression with controls
    coef_full = res["coef_child_full"]
    p_full = res["p_child_full"]
    or_full = res["or_child_full"]

    coef_simple = res["coef_child_simple"]
    p_simple = res["p_child_simple"]
    or_simple = res["or_child_simple"]

    # Direction: negative coefficient / OR<1 means children associated with FEWER affairs
    direction_full = -1 if coef_full < 0 else (1 if coef_full > 0 else 0)

    # Significance thresholds
    strong_sig = p_full < 0.01
    weak_sig = 0.01 <= p_full < 0.05
    not_sig = p_full >= 0.05

    # Effect size categories based on odds ratio
    if np.isnan(or_full):
        effect_size = "unknown"
    elif or_full <= 0.6:
        effect_size = "large_decrease"
    elif or_full <= 0.8:
        effect_size = "moderate_decrease"
    elif or_full < 1.0:
        effect_size = "small_decrease"
    elif or_full <= 1.25:
        effect_size = "small_increase"
    elif or_full <= 1.6:
        effect_size = "moderate_increase"
    else:
        effect_size = "large_increase"

    # Base on full model
    if direction_full < 0 and strong_sig:
        # Strong evidence that children are associated with fewer affairs
        if effect_size == "large_decrease":
            response = 90
        elif effect_size == "moderate_decrease":
            response = 80
        else:
            response = 75
        answer_text = (
            "Yes – there is strong statistical evidence that, in this sample, "
            "having children is associated with lower engagement in extramarital affairs."
        )
    elif direction_full < 0 and weak_sig:
        # Statistically significant but not overwhelmingly strong
        if effect_size in ("large_decrease", "moderate_decrease"):
            response = 75
        else:
            response = 65
        answer_text = (
            "Yes – there is statistically significant evidence that having children "
            "is associated with somewhat lower engagement in extramarital affairs, "
            "though the effect is modest."
        )
    elif direction_full < 0 and not_sig:
        # Direction suggests decrease but evidence is not conventionally significant
        response = 45
        answer_text = (
            "No – although the estimated effect of having children points toward "
            "slightly fewer extramarital affairs, this association is not "
            "statistically significant at conventional levels."
        )
    elif direction_full > 0 and strong_sig:
        # Strong evidence that children are associated with MORE affairs
        if effect_size == "large_increase":
            response = 5
        elif effect_size == "moderate_increase":
            response = 15
        else:
            response = 25
        answer_text = (
            "No – the data provide strong evidence that having children is associated "
            "with higher, not lower, engagement in extramarital affairs in this sample."
        )
    elif direction_full > 0 and weak_sig:
        response = 25
        answer_text = (
            "No – there is statistically significant evidence that having children "
            "is associated with a small increase in extramarital affairs, "
            "rather than a decrease."
        )
    elif direction_full > 0 and not_sig:
        response = 40
        answer_text = (
            "No – the estimated effect of having children is slightly in the direction "
            "of more extramarital affairs, and the association is not statistically significant."
        )
    else:
        # No clear direction
        response = 50
        answer_text = (
            "The data do not provide clear evidence that having children either "
            "increases or decreases engagement in extramarital affairs."
        )

    explanation_lines = [
        answer_text,
        "",
        "Key evidence from the analysis:",
        f"- Sample size: {res['n_total']} individuals "
        f"({res['n_children']} with children, {res['n_no_children']} without).",
        "- Mean number of extramarital affairs (last year): "
        f"{res['mean_affairs_children']:.3f} with children vs. "
        f"{res['mean_affairs_no_children']:.3f} without; "
        f"Welch t-test p-value = {res['t_affairs_p']:.4f}.",
        "- Probability of having at least one affair: "
        f"{res['prop_affair_children']:.3f} with children vs. "
        f"{res['prop_affair_no_children']:.3f} without; "
        f"two-proportion z-test p-value = {res['z_prop_p']:.4f}.",
        "- Logistic regression (any affair ~ children, unadjusted): "
        f"odds ratio for having children = {res['or_child_simple']:.3f}, "
        f"p-value = {res['p_child_simple']:.4f}.",
        "- Logistic regression with controls for age, years married, religiousness, "
        "education, occupation, marital rating, and gender: "
        f"odds ratio for having children = {res['or_child_full']:.3f}, "
        f"p-value = {res['p_child_full']:.4f}.",
        "",
        "The conclusion and the 0–100 response scale value are based primarily on "
        "the direction, magnitude, and statistical significance of the adjusted "
        "logistic regression coefficient for having children, cross-checked with "
        "group differences in means and proportions.",
    ]

    explanation = "\n".join(explanation_lines)
    return response, explanation


def main():
    response, explanation = run_analysis()
    output = {"response": int(response), "explanation": explanation}
    # Write a single JSON object with no extra content
    Path("conclusion.txt").write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

