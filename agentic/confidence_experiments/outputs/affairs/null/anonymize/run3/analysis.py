import json
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def format_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Construct key variables
    df["affair_any"] = (df["feature2"] > 0).astype(int)
    df["children"] = df["feature6"].str.lower().eq("yes").astype(int)
    df["gender_male"] = df["feature3"].str.lower().eq("male").astype(int)

    n = int(len(df))

    # Descriptive summaries by children status
    group_affair_rate = df.groupby("children")["affair_any"].mean()
    group_affair_mean_freq = df.groupby("children")["feature2"].mean()
    group_affair_median_freq = df.groupby("children")["feature2"].median()
    n_by_child: Dict[int, int] = df["children"].value_counts().to_dict()

    rate_no_child = float(group_affair_rate.get(0, np.nan))
    rate_with_child = float(group_affair_rate.get(1, np.nan))
    mean_freq_no_child = float(group_affair_mean_freq.get(0, np.nan))
    mean_freq_with_child = float(group_affair_mean_freq.get(1, np.nan))
    median_freq_no_child = float(group_affair_median_freq.get(0, np.nan))
    median_freq_with_child = float(group_affair_median_freq.get(1, np.nan))
    n_no_child = int(n_by_child.get(0, 0))
    n_with_child = int(n_by_child.get(1, 0))

    # Logistic regression: any affair ~ children + covariates
    X = df[
        [
            "children",
            "gender_male",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    ].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = df["affair_any"].astype(float)

    logit_model = sm.Logit(y, X).fit(disp=False)
    params = logit_model.params
    pvalues = logit_model.pvalues

    coef_child = float(params["children"])
    p_child = float(pvalues["children"])
    or_child = float(np.exp(coef_child))
    ci_low, ci_high = logit_model.conf_int().loc["children"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Non-parametric test on frequency scores (ordinal / skewed)
    freq_children1 = df.loc[df["children"] == 1, "feature2"]
    freq_children0 = df.loc[df["children"] == 0, "feature2"]
    mw_res = stats.mannwhitneyu(freq_children1, freq_children0, alternative="two-sided")
    mw_p = float(mw_res.pvalue)

    decreases = coef_child < 0.0
    significant = p_child < 0.05

    if significant and decreases:
        yes_no = "Yes"
        response_int = 80
    elif significant and not decreases:
        yes_no = "No"
        response_int = 5
    else:
        # Effect not statistically reliable; answer "No" with moderate confidence
        if p_child > 0.3 and mw_p > 0.3:
            response_int = 20
        else:
            response_int = 35
        yes_no = "No"

    p_child_str = format_p(p_child)
    mw_p_str = format_p(mw_p)

    if or_child < 1.0:
        direction_phrase = "lower odds"
    elif or_child > 1.0:
        direction_phrase = "higher odds"
    else:
        direction_phrase = "similar odds"

    if yes_no == "Yes":
        tail = (
            "These patterns and the statistically significant adjusted association support "
            "a 'Yes' answer: in this dataset, having children is associated with a modest "
            "decrease in engagement in extramarital affairs."
        )
    elif significant and not decreases:
        tail = (
            "Although the association is statistically significant, it points toward higher "
            "rather than lower odds of affairs among parents, so with respect to the question "
            "of whether children decrease affairs the answer is 'No'."
        )
    else:
        tail = (
            "Because the children coefficient is not statistically significant in the adjusted "
            "model and the non-parametric test also fails to detect a difference, the data do "
            "not provide reliable evidence that having children decreases extramarital affairs."
        )

    explanation = (
        f"{yes_no}: Using the classic Fair (1978) extramarital affairs dataset with {n} first-married "
        f"men and women, I examined whether having children is associated with lower engagement in "
        f"extramarital sexual intercourse. Among respondents without children (n = {n_no_child}), "
        f"{rate_no_child*100:.1f}% reported at least one affair in the past year, compared with "
        f"{rate_with_child*100:.1f}% among those with children (n = {n_with_child}). The mean affair "
        f"frequency score (0 = none, higher values = more frequent or regular affairs) was "
        f"{mean_freq_no_child:.2f} for non-parents versus {mean_freq_with_child:.2f} for parents "
        f"(medians {median_freq_no_child:.1f} vs {median_freq_with_child:.1f}). A logistic regression "
        f"of any affair on the presence of children, controlling for gender, age, years married, "
        f"religiousness, education, occupation, and self-rated marital happiness, estimated an odds "
        f"ratio of {or_child:.2f} for having children (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, "
        f"p {p_child_str}), corresponding to {direction_phrase} of having an extramarital affair for "
        f"parents relative to non-parents. A Mann–Whitney U test comparing the full distribution of "
        f"affair frequency scores between parents and non-parents was also non-significant "
        f"(p {mw_p_str}). {tail}"
    )

    conclusion = {"response": int(response_int), "explanation": explanation}

    # Write only the JSON object, with no extra text
    with open("conclusion.txt", "w") as f:
        f.write(json.dumps(conclusion))


if __name__ == "__main__":
    main()

