import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def p_to_weight(p: float) -> float:
    """Convert a p-value into an evidence weight in [0, 1]."""
    if p < 1e-4:
        return 1.0
    if p < 1e-3:
        return 0.9
    if p < 1e-2:
        return 0.8
    if p < 5e-2:
        return 0.7
    if p < 1e-1:
        return 0.5
    if p < 0.2:
        return 0.3
    return 0.1


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")
    df = df.copy()

    # Ensure the key variables are present and clean
    df = df.dropna(subset=["feature2", "feature6"])
    df["children"] = df["feature6"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Group-level descriptive statistics
    group_stats = (
        df.groupby("feature6")["feature2"]
        .agg(["mean", "median", "std", "count"])
        .to_dict("index")
    )
    prop_nonzero = (
        df.groupby("feature6")["any_affair"].mean().to_dict()
    )

    yes = df.loc[df["children"] == 1, "feature2"]
    no = df.loc[df["children"] == 0, "feature2"]

    # Two-sample tests for difference in affair frequency
    ttest = stats.ttest_ind(no, yes, equal_var=False)
    mwu = stats.mannwhitneyu(no, yes, alternative="two-sided")

    # Regression models to adjust for covariates
    coef_children_logit = None
    p_children_logit = None
    or_children = None
    or_ci = None

    coef_children_ols = None
    p_children_ols = None

    try:
        logit_model = smf.logit(
            "any_affair ~ children + feature4 + feature5 + C(feature3) + "
            "feature7 + feature8 + feature9 + feature10",
            data=df,
        ).fit(disp=False)

        coef_children_logit = float(logit_model.params["children"])
        p_children_logit = float(logit_model.pvalues["children"])

        or_children = float(np.exp(coef_children_logit))
        ci_low, ci_high = logit_model.conf_int().loc["children"].tolist()
        or_ci = [float(np.exp(ci_low)), float(np.exp(ci_high))]
    except Exception:
        # If the logistic regression fails (e.g., separation), ignore it but continue.
        pass

    try:
        ols_model = smf.ols(
            "feature2 ~ children + feature4 + feature5 + C(feature3) + "
            "feature7 + feature8 + feature9 + feature10",
            data=df,
        ).fit()
        coef_children_ols = float(ols_model.params["children"])
        p_children_ols = float(ols_model.pvalues["children"])
    except Exception:
        # If OLS fails for some reason, continue with the rest.
        pass

    # Effect direction and size from unadjusted means
    mean_yes = float(group_stats["yes"]["mean"])
    mean_no = float(group_stats["no"]["mean"])
    diff = mean_no - mean_yes  # positive if children associated with fewer affairs

    # Cohen's d effect size
    if len(yes) > 1 and len(no) > 1:
        pooled_var = (
            yes.var(ddof=1) * (len(yes) - 1)
            + no.var(ddof=1) * (len(no) - 1)
        ) / (len(yes) + len(no) - 2)
        pooled_sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
    else:
        pooled_sd = 0.0

    if pooled_sd > 0:
        d = diff / pooled_sd
    else:
        d = 0.0

    # Combine evidence from multiple p-values
    p_vals = [float(ttest.pvalue), float(mwu.pvalue)]
    if p_children_logit is not None:
        p_vals.append(p_children_logit)
    if p_children_ols is not None:
        p_vals.append(p_children_ols)

    weight = max(p_to_weight(p) for p in p_vals)
    direction = np.sign(diff)

    # Map effect size and evidence into 0–100 scale
    # Zero corresponds to "strong No", 100 to "strong Yes",
    # and 50 represents "no clear evidence either way".
    magnitude = float(np.tanh(abs(d)))  # in [0, 1)
    combined = magnitude * weight
    delta = 40 * combined  # max 40-point shift from the neutral 50

    if direction > 0:
        # Children associated with fewer affairs -> answer "Yes"
        score = 50 + delta
    elif direction < 0:
        # Children associated with more affairs -> answer "No"
        score = 50 - delta
    else:
        score = 50.0

    score_int = int(round(min(100, max(0, score))))

    # Build explanation string with key numerical evidence
    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in "
        "extramarital affairs (measured by feature2)?"
    )
    explanation_parts.append(
        f"In the sample (n={len(df)}), the mean affair frequency is "
        f"{mean_yes:.3f} for couples with children (feature6='yes', "
        f"n={len(yes)}) and {mean_no:.3f} for couples without children "
        f"(feature6='no', n={len(no)}), giving a difference of "
        f"{diff:.3f} (positive means fewer affairs among those with "
        "children). "
        f"Cohen's d for this difference is approximately {d:.3f}."
    )
    explanation_parts.append(
        "A Welch t-test comparing affair frequency between couples with "
        "and without children yields "
        f"t={ttest.statistic:.3f} with p={float(ttest.pvalue):.3g}. "
        "A Mann–Whitney U test, which is robust to non-normality, gives "
        f"U={mwu.statistic:.3f} with p={float(mwu.pvalue):.3g}."
    )

    if coef_children_logit is not None and or_children is not None and or_ci:
        explanation_parts.append(
            "Using a logistic regression for any extramarital affair "
            "(feature2 > 0) on the presence of children and controls "
            "(age, years married, gender, religiousness, education, "
            "occupation, and self-rated marriage), the coefficient for "
            f"children is {coef_children_logit:.3f} "
            f"(odds ratio={or_children:.3f}, "
            f"95% CI [{or_ci[0]:.3f}, {or_ci[1]:.3f}], "
            f"p={p_children_logit:.3g}). "
            "An odds ratio below 1 indicates that having children is "
            "associated with a lower likelihood of any affair."
        )

    if coef_children_ols is not None and p_children_ols is not None:
        if coef_children_ols < 0:
            direction_text = "negative"
        elif coef_children_ols > 0:
            direction_text = "positive"
        else:
            direction_text = "near-zero"
        explanation_parts.append(
            "An OLS regression of affair frequency on children and the "
            "same controls estimates the children coefficient as "
            f"{coef_children_ols:.3f} with p={p_children_ols:.3g}, "
            f"consistent with a {direction_text} association after "
            "adjustment."
        )

    if direction > 0 and weight >= 0.5:
        qualitative = (
            "The results consistently indicate that having children is "
            "associated with fewer extramarital affairs."
        )
    elif weight < 0.4 or abs(d) < 0.2:
        qualitative = (
            "The results do not provide strong evidence that having "
            "children reduces extramarital affairs."
        )
    else:
        qualitative = (
            "The association between having children and extramarital "
            "affairs is weak and uncertain in direction."
        )

    explanation_parts.append(qualitative)
    explanation_parts.append(
        "On a 0–100 Likert scale where 0 means a strong 'No' and 100 "
        "means a strong 'Yes' to the statement "
        "'Having children decreases engagement in extramarital affairs', "
        f"I map the combined statistical evidence to a response of "
        f"{score_int}."
    )

    explanation = "\n\n".join(explanation_parts)

    result = {"response": score_int, "explanation": explanation}
    Path("conclusion.txt").write_text(
        json.dumps(result, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()

