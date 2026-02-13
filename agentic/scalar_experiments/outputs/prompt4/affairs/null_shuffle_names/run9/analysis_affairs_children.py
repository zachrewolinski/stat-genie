import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def map_p_to_strength(p_value: float) -> str:
    if p_value < 0.001:
        return "very strong"
    if p_value < 0.01:
        return "strong"
    if p_value < 0.05:
        return "moderate"
    if p_value < 0.1:
        return "weak"
    return "little"


def map_to_likert(diff: float, p_value: float) -> int:
    """
    Map effect direction and strength to a 0-100 Likert score.

    diff: difference in affair rate (no-children minus children).
          Positive diff means children are associated with *fewer* affairs.
    """
    # No directional signal at all.
    if np.isclose(diff, 0.0):
        return 50

    # Normalize absolute effect size to [0, 1], saturating around 0.20.
    effect = min(abs(diff) / 0.20, 1.0)

    # Map p-value to a significance factor in [0.2, 1].
    if p_value < 0.001:
        sig = 1.0
    elif p_value < 0.01:
        sig = 0.85
    elif p_value < 0.05:
        sig = 0.7
    elif p_value < 0.1:
        sig = 0.5
    else:
        sig = 0.25

    strength = effect * sig  # in [0,1]

    if diff > 0:
        # Evidence that having children reduces affair engagement.
        score = 50 + strength * 50
    else:
        # Evidence that having children does NOT reduce (or may increase) affair engagement.
        score = 50 - strength * 50

    score_int = int(round(score))
    return max(0, min(100, score_int))


def main() -> None:
    df = pd.read_csv("affairs.csv")

    n = len(df)

    # According to the provided metadata, the columns are semantically shuffled:
    # - "age" actually encodes frequency of extramarital intercourse in the past year:
    #       0 = none, 1 = once, 2 = twice, 3 = three times,
    #       7 = 4–10 times, 12 = monthly/weekly/daily.
    # - "religiousness" is described as: "Are there children in the marriage?" (yes/no).
    #
    # We therefore define:
    #   has_children: indicator based on "religiousness"
    #   affair_freq_code: coded affair frequency from "age"
    #   had_affair: binary engagement indicator (any non-zero affair frequency).
    has_children = df["religiousness"].astype(str).str.lower().eq("yes")
    affair_freq_code = df["age"].astype(float)
    had_affair = (affair_freq_code > 0).astype(int)

    n_children = int(has_children.sum())
    n_nochildren = int((~has_children).sum())

    rate_children = float(had_affair[has_children].mean())
    rate_nochildren = float(had_affair[~has_children].mean())
    diff = rate_nochildren - rate_children  # positive => fewer affairs among those with children

    # Two-sample test for proportion of respondents with any affair.
    counts = np.array(
        [
            int(had_affair[~has_children].sum()),
            int(had_affair[has_children].sum()),
        ]
    )
    nobs = np.array([n_nochildren, n_children])

    # H0: p_nochildren = p_children
    # H1 (one-sided): p_nochildren > p_children  (children reduce affair engagement).
    z_stat, p_value = proportions_ztest(counts, nobs, alternative="larger")

    # Logistic regression for additional effect size description.
    X = pd.DataFrame({"const": 1.0, "has_children": has_children.astype(int)})
    y = had_affair
    logit_model = sm.Logit(y, X).fit(disp=False)
    coef = float(logit_model.params["has_children"])
    or_est = float(np.exp(coef))

    ci = logit_model.conf_int().loc["has_children"]
    or_low = float(np.exp(ci[0]))
    or_high = float(np.exp(ci[1]))
    p_logit = float(logit_model.pvalues["has_children"])

    # Narrative descriptors.
    sig_strength = map_p_to_strength(p_value)
    direction_desc = (
        "lower"
        if diff > 0
        else "similar or higher"
    )

    if diff > 0 and p_value < 0.05:
        qualitative_conclusion = "does modestly reduce"
        overall_strength = f"{sig_strength} statistical"
    elif diff > 0 and p_value >= 0.05:
        qualitative_conclusion = "may reduce slightly but the evidence is statistically weak for this sample and model"
        overall_strength = f"{sig_strength} statistical"
    elif diff <= 0 and p_value < 0.05:
        qualitative_conclusion = "does not reduce and may instead be associated with greater"
        overall_strength = f"{sig_strength} statistical"
    else:
        qualitative_conclusion = "does not show a clear, statistically reliable change in"
        overall_strength = f"{sig_strength} statistical"

    evidence_strength = (
        f"{sig_strength} (children group affair rate {rate_children:.1%}, "
        f"no-children group {rate_nochildren:.1%})"
    )

    diff_pct = diff * 100.0

    explanation = (
        f"Using the 1969 Psychology Today survey sample of {n} married respondents, "
        f"I treated the 'age' column as the coded frequency of extramarital sexual intercourse "
        f"during the past year (0 = none; higher values indicate more frequent affairs) based on the metadata, "
        f"and defined 'engagement in extramarital affairs' as any non-zero value of this variable. "
        f"The 'religiousness' column was interpreted per the metadata as indicating whether there are children "
        f"in the marriage ('yes' vs 'no'), and I used it to construct a binary 'has children' indicator. "
        f"Among couples with children (n = {n_children}), {rate_children:.1%} reported at least one extramarital encounter "
        f"in the past year, compared with {rate_nochildren:.1%} among couples without children (n = {n_nochildren}), "
        f"a difference of {diff_pct:.1f} percentage points (no-children minus children). "
        f"A one-sided two-sample test for proportions assessing whether the affair rate is higher among couples "
        f"without children than among those with children yielded z = {z_stat:.2f} and p = {p_value:.3g}, "
        f"providing {sig_strength} evidence in that direction. "
        f"In a logistic regression of affair engagement on the children indicator, respondents with children had an estimated "
        f"odds ratio of {or_est:.2f} for reporting an affair relative to those without children "
        f"(95% confidence interval {or_low:.2f}–{or_high:.2f}, p = {p_logit:.3g}). "
        f"Overall, the estimated effect of having children on affair engagement is {direction_desc} in this dataset, and the "
        f"statistical evidence is {evidence_strength}, so I judge that the data provide {overall_strength} evidence that "
        f"having children {qualitative_conclusion} engagement in extramarital affairs."
    )

    response_score = map_to_likert(diff, p_value)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

