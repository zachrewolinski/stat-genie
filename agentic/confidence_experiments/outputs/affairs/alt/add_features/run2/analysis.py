import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Focus on key variables for the research question
    if "children" not in df.columns or "affairs" not in df.columns:
        explanation = (
            "The required columns 'children' and 'affairs' are not present in the dataset, "
            "so it is impossible to assess whether having children decreases engagement in extramarital affairs."
        )
        result = {"response": 50, "explanation": explanation}
        with open("conclusion.txt", "w") as f:
            json.dump(result, f)
        return

    # Restrict to rows with clearly coded children status
    df = df[df["children"].isin(["yes", "no"])].copy()

    # Binary outcome: any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Encode children as indicator (1 = has children)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    n_obs = int(df.shape[0])

    # Descriptive statistics by children status
    group = df.groupby("children")["any_affair"].agg(["mean", "count"])
    mean_no = float(group.loc["no", "mean"]) if "no" in group.index else np.nan
    mean_yes = float(group.loc["yes", "mean"]) if "yes" in group.index else np.nan

    # Contingency table and chi-squared test (children x any_affair)
    ct = pd.crosstab(df["children"], df["any_affair"])
    chi2, chi_p, _, _ = chi2_contingency(ct)

    # Logistic regression: any_affair ~ children + controls (where available)
    covariates = ["children_yes"]
    for col in ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]:
        if col in df.columns:
            covariates.append(col)

    X = df[covariates].copy()
    X = sm.add_constant(X, has_constant="add")
    y = df["any_affair"]

    beta_children = np.nan
    p_children = np.nan
    or_children = np.nan

    try:
        model = sm.Logit(y, X)
        res = model.fit(disp=False)
        if "children_yes" in res.params.index:
            beta_children = float(res.params["children_yes"])
            p_children = float(res.pvalues["children_yes"])
            or_children = float(np.exp(beta_children))
    except Exception:
        # If the model fails to converge, fall back to chi-squared only
        beta_children = np.nan
        p_children = chi_p
        or_children = np.nan

    def map_score(beta: float, p_val: float, odds_ratio: float) -> int:
        """Map effect direction and significance to a 0-100 Likert-scale score.

        0  = strong 'No' (children clearly do NOT decrease affairs)
        50 = indeterminate / no clear evidence
        100 = strong 'Yes' (children clearly DO decrease affairs)
        """
        if np.isnan(p_val):
            return 50

        # If effect is not estimated, rely purely on significance of association
        if np.isnan(beta):
            if p_val < 0.001:
                return 75
            if p_val < 0.01:
                return 65
            if p_val < 0.05:
                return 60
            if p_val < 0.1:
                return 55
            return 50

        # Children coefficient is positive (if anything, more affairs with children)
        if beta >= 0:
            if p_val < 0.001:
                return 5
            if p_val < 0.01:
                return 10
            if p_val < 0.05:
                return 20
            if p_val < 0.1:
                return 30
            return 40

        # Negative association: children associated with fewer affairs
        if p_val < 0.001:
            base = 90
        elif p_val < 0.01:
            base = 80
        elif p_val < 0.05:
            base = 70
        elif p_val < 0.1:
            base = 60
        else:
            base = 50

        # Adjust based on the size of the odds ratio departure from 1
        if not np.isnan(odds_ratio):
            if odds_ratio < 0.5:
                base += 5
            elif odds_ratio < 0.8:
                base += 0
            elif odds_ratio < 0.95:
                base -= 5
            else:
                base -= 10

        base = max(0, min(100, base))
        return int(round(base))

    score = map_score(beta_children, p_children, or_children)

    # Build human-readable explanation
    mean_no_pct = mean_no * 100 if not np.isnan(mean_no) else np.nan
    mean_yes_pct = mean_yes * 100 if not np.isnan(mean_yes) else np.nan

    # Safely format percentages if available
    if np.isnan(mean_no_pct) or np.isnan(mean_yes_pct):
        rate_sentence = (
            "The dataset does not provide stable estimates of affair rates by child status, "
            "so descriptive differences cannot be reliably summarized."
        )
    else:
        rate_sentence = (
            f"Among the {n_obs} married individuals in this dataset, "
            f"approximately {mean_no_pct:.1f}% of those without children and "
            f"{mean_yes_pct:.1f}% of those with children reported at least one extramarital affair in the past year."
        )

    if np.isnan(beta_children):
        effect_sentence = (
            f"A chi-squared test of the association between children and any extramarital affair "
            f"yields p={chi_p:.3f}, which provides "
        )
        if chi_p < 0.05:
            effect_sentence += "evidence of some relationship between having children and affairs, "
        else:
            effect_sentence += "little evidence of a systematic relationship between having children and affairs, "
        effect_sentence += (
            "but without a stable regression estimate we cannot precisely quantify how strongly children change the odds."
        )
    else:
        direction = "lower" if beta_children < 0 else "higher"
        magnitude_sentence = (
            f"In a logistic regression controlling for available covariates "
            f"(age, years married, religiousness, education, occupation, and marital rating where present), "
            f"the coefficient on having children corresponds to an odds ratio of {or_children:.2f}, "
            f"indicating {direction} odds of any affair for parents compared with non-parents."
        )

        if p_children < 0.001:
            sig_phrase = "strong statistical evidence (p<0.001)"
        elif p_children < 0.01:
            sig_phrase = "strong statistical evidence (p<0.01)"
        elif p_children < 0.05:
            sig_phrase = "moderate statistical evidence (p<0.05)"
        elif p_children < 0.1:
            sig_phrase = "weak statistical evidence (p<0.10)"
        else:
            sig_phrase = "little statistical evidence (p>=0.10)"

        effect_sentence = (
            f"{magnitude_sentence} There is {sig_phrase} that this effect differs from zero after adjustment."
        )

    scale_sentence = (
        f"On a 0-100 Likert scale where 0 represents a strong 'No' and 100 a strong 'Yes' "
        f"to the question of whether having children decreases engagement in extramarital affairs, "
        f"this analysis corresponds to a score of {score}."
    )

    explanation = " ".join([rate_sentence, effect_sentence, scale_sentence])

    result = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

