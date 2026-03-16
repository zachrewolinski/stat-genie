import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    desc_affairs = (
        df.groupby("children")["affairs"].agg(["mean", "std", "count"]).to_dict()
    )
    prop_any_affair = df.groupby("children")["any_affair"].mean().to_dict()

    # Unadjusted logistic regression: any_affair ~ children
    model_unadj = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    coef_unadj = float(model_unadj.params.get("C(children)[T.yes]", np.nan))
    p_unadj = float(model_unadj.pvalues.get("C(children)[T.yes]", np.nan))
    or_unadj = float(np.exp(coef_unadj)) if np.isfinite(coef_unadj) else np.nan

    # Adjusted logistic regression controlling for basic demographics and marriage quality.
    model_adj = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + C(gender)"
        " + religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)
    coef_adj = float(model_adj.params.get("C(children)[T.yes]", np.nan))
    p_adj = float(model_adj.pvalues.get("C(children)[T.yes]", np.nan))
    or_adj = float(np.exp(coef_adj)) if np.isfinite(coef_adj) else np.nan

    # In this coding, the coefficient is for children = "yes" vs "no".
    # A negative coefficient would mean children are associated with fewer affairs.
    def direction(coef: float) -> str:
        if not np.isfinite(coef):
            return "unknown"
        return "decrease" if coef < 0 else "increase"

    dir_unadj = direction(coef_unadj)
    dir_adj = direction(coef_adj)
    sig_unadj = np.isfinite(p_unadj) and p_unadj < 0.05
    sig_adj = np.isfinite(p_adj) and p_adj < 0.05

    mean_affairs_no = float(desc_affairs["mean"].get("no"))
    mean_affairs_yes = float(desc_affairs["mean"].get("yes"))
    prop_any_no = float(prop_any_affair.get("no"))
    prop_any_yes = float(prop_any_affair.get("yes"))

    # Determine whether the data support the claim that having children DECREASES affairs.
    # We require the adjusted model to show a negative, statistically significant effect
    # to count as evidence for a decrease.
    supports_decrease = sig_adj and dir_adj == "decrease"
    supports_increase = sig_adj and dir_adj == "increase"

    # Map evidence to a 0–100 Likert score, where 0 = strong "No"
    # (children clearly do not decrease affairs, and may increase them),
    # and 100 = strong "Yes" (children clearly decrease affairs).
    if supports_decrease:
        # Children significantly associated with fewer affairs after adjustment.
        # Use the change in probability as a rough effect size.
        prob_diff = prop_any_no - prop_any_yes  # positive if children reduce affairs
        # Typical differences here are modest; scale them into [60, 90].
        base = 70
        adjustment = np.clip(prob_diff * 100, 0, 20)
        score = base + adjustment
    elif supports_increase:
        # Children significantly associated with MORE affairs.
        # Strong evidence against the claim that children decrease affairs.
        prob_diff = prop_any_yes - prop_any_no
        base = 10
        adjustment = np.clip(prob_diff * 100, 0, 15)
        score = max(0, base - adjustment)
    else:
        # No clear evidence either way after adjustment.
        # If both models trend toward an increase (positive coefficients) but
        # are not significant, we lean toward "No" but not extremely.
        if dir_adj == "increase" and dir_unadj == "increase":
            score = 25
        elif dir_adj == "decrease" and dir_unadj == "decrease":
            # Consistent but weak trend toward a decrease.
            score = 60
        else:
            # Mixed or essentially null results.
            score = 40

    score_int = int(round(np.clip(score, 0, 100)))

    # Build explanation text summarizing evidence and conclusion.
    lines = []
    lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    lines.append(f"Sample size: {len(df)} currently married individuals.")
    lines.append(
        "Outcome was analyzed both as a binary indicator of any affair in the past year "
        "and via logistic regression models."
    )
    lines.append(
        f"Descriptively, the mean affair score was {mean_affairs_no:.2f} for couples "
        f"without children and {mean_affairs_yes:.2f} for couples with children."
    )
    lines.append(
        f"The proportion reporting any affair was {prop_any_no:.3f} without children "
        f"and {prop_any_yes:.3f} with children."
    )
    lines.append(
        "An unadjusted logistic regression of any affair on children status yielded an "
        f"odds ratio of {or_unadj:.2f} for having children (vs. no children) with "
        f"p-value {p_unadj:.4f}, indicating a direction of {dir_unadj} in affair risk."
    )
    lines.append(
        "An adjusted logistic regression controlling for age, years married, gender, "
        f"religiousness, education, occupation, and marital rating yielded an odds ratio "
        f"of {or_adj:.2f} for having children with p-value {p_adj:.4f}, indicating a "
        f"direction of {dir_adj} after adjustment."
    )

    if supports_decrease:
        lines.append(
            "Because the adjusted model shows a statistically significant negative "
            "association between having children and the likelihood of any extramarital "
            "affair, the data provide evidence that having children is associated with "
            "slightly fewer affairs."
        )
        lines.append(
            "However, the absolute differences in probabilities are modest, so this is a "
            "qualified rather than a dramatic decrease."
        )
    elif supports_increase:
        lines.append(
            "Because the adjusted model shows a statistically significant positive "
            "association between having children and the likelihood of any extramarital "
            "affair, the data contradict the claim that children decrease affairs and "
            "instead suggest a small increase."
        )
    else:
        lines.append(
            "The adjusted model does not show a statistically significant effect of "
            "having children on the likelihood of any extramarital affair at the 5% "
            "level, so there is no strong evidence that children either decrease or "
            "increase affairs."
        )
        if dir_adj == "increase" and dir_unadj == "increase":
            lines.append(
                "Both models trend toward a positive (increasing) association, but the "
                "effect is too imprecise to be conclusive."
            )
        elif dir_adj == "decrease" and dir_unadj == "decrease":
            lines.append(
                "Both models trend toward a negative (decreasing) association, but the "
                "effect is too imprecise to be conclusive."
            )

    lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the "
        f"question of whether having children decreases engagement in extramarital "
        f"affairs, this analysis corresponds to a score of {score_int}."
    )

    explanation = " ".join(lines)

    result = {"response": score_int, "explanation": explanation}

    # Write the required JSON object to conclusion.txt with no extra lines.
    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

