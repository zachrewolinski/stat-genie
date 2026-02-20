import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Ensure expected columns exist
    required_cols = [
        "affairs",
        "children",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Basic preprocessing
    df = df.copy()
    # Treat negative or missing counts as zero if any appear
    df["affairs"] = df["affairs"].fillna(0).clip(lower=0)
    df = df[df["children"].isin(["yes", "no"])].copy()

    # Binary outcome for any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Group-level summaries
    group = df.groupby("children")
    mean_affairs = group["affairs"].mean()
    median_affairs = group["affairs"].median()
    prop_any = group["any_affair"].mean()
    n_per_group = group.size()

    # Difference in means (Welch t-test)
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]
    t_stat, p_ttest = stats.ttest_ind(
        affairs_yes, affairs_no, equal_var=False, alternative="less"
    )

    # Logistic regression for any affair, controlling for key covariates
    # children="no" is the reference category
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract effect of having children (yes vs no)
    # statsmodels encodes this as C(children)[T.yes]
    coef_children = logit_model.params.get("C(children)[T.yes]", np.nan)
    p_children = logit_model.pvalues.get("C(children)[T.yes]", np.nan)
    or_children = float(np.exp(coef_children)) if np.isfinite(coef_children) else np.nan

    # Map statistical evidence to Likert-style 0–100 response
    # Direction: negative coefficient => children associated with fewer affairs
    response: int
    explanation_direction: str

    if not np.isfinite(coef_children) or not np.isfinite(p_children):
        # Fallback: base on unadjusted mean difference
        diff = float(mean_affairs.get("yes", np.nan) - mean_affairs.get("no", np.nan))
        if np.isnan(diff) or abs(diff) < 0.01:
            response = 50
            explanation_direction = (
                "the average number of affairs is essentially the same "
                "for parents and non-parents"
            )
        elif diff < 0:
            response = 65
            explanation_direction = (
                "parents report slightly fewer extramarital affairs on average"
            )
        else:
            response = 35
            explanation_direction = (
                "parents report slightly more extramarital affairs on average"
            )
    else:
        # Start from neutral 50 and adjust based on sign, magnitude, and significance
        if p_children < 0.01:
            strength = 30
        elif p_children < 0.05:
            strength = 20
        elif p_children < 0.10:
            strength = 10
        else:
            strength = 5

        if coef_children < 0:
            response = 50 + strength
            explanation_direction = (
                "having children is associated with a lower likelihood of "
                "having an extramarital affair, after controlling for age, "
                "years married, religiosity, education, occupation, "
                "marital satisfaction, and gender"
            )
        elif coef_children > 0:
            response = 50 - strength
            explanation_direction = (
                "having children is associated with a higher likelihood of "
                "having an extramarital affair, after controlling for age, "
                "years married, religiosity, education, occupation, "
                "marital satisfaction, and gender"
            )
        else:
            response = 50
            explanation_direction = (
                "the regression suggests essentially no difference in affairs "
                "between parents and non-parents"
            )

    # Ensure response is within [0, 100] and an integer
    response = int(min(100, max(0, response)))

    # Build textual explanation including key statistics
    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_lines.append(
        f"The dataset contains {len(df)} married individuals, of whom "
        f"{int(n_per_group.get('yes', 0))} report having children and "
        f"{int(n_per_group.get('no', 0))} report not having children."
    )
    explanation_lines.append(
        "On average, parents report "
        f"{mean_affairs.get('yes', np.nan):.2f} affairs in the past year "
        f"(median {median_affairs.get('yes', np.nan):.0f}), compared with "
        f"{mean_affairs.get('no', np.nan):.2f} affairs "
        f"(median {median_affairs.get('no', np.nan):.0f}) among non-parents."
    )
    explanation_lines.append(
        "The proportion with any extramarital affair is "
        f"{prop_any.get('yes', np.nan):.3f} for parents vs "
        f"{prop_any.get('no', np.nan):.3f} for non-parents."
    )
    explanation_lines.append(
        "A Welch t-test comparing the mean number of affairs between the two "
        f"groups yields t = {t_stat:.2f} (one-sided p-value {p_ttest:.3f}) "
        "when testing whether parents have fewer affairs than non-parents."
    )
    if np.isfinite(coef_children) and np.isfinite(p_children):
        explanation_lines.append(
            "A logistic regression for having any affair, adjusted for age, "
            "years married, religiousness, education, occupation, marital "
            "satisfaction rating, and gender, estimates the coefficient for "
            f"having children as {coef_children:.3f} on the log-odds scale "
            f"(odds ratio {or_children:.3f}, p-value {p_children:.3f})."
        )
    explanation_lines.append(
        f"Overall, {explanation_direction}, and this evidence is summarized "
        f"by a Likert-style response score of {response} on a 0–100 scale, "
        "where higher values indicate stronger support for the claim that "
        "children decrease engagement in extramarital affairs."
    )

    explanation = " ".join(explanation_lines)

    output = {"response": response, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

