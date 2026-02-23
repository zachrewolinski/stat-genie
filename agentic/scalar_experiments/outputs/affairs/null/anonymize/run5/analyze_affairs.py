import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def load_data():
    df = pd.read_csv("affairs.csv")

    # Core variables
    df["children"] = df["feature6"].map({"yes": 1, "no": 0}).astype(int)
    df["affairs_freq"] = df["feature2"].astype(float)
    df["any_affair"] = (df["affairs_freq"] > 0).astype(int)

    # Covariates for adjusted analysis
    df["male"] = (df["feature3"] == "male").astype(int)
    df["age"] = df["feature4"].astype(float)
    df["yearsmarried"] = df["feature5"].astype(float)
    df["religiousness"] = df["feature7"].astype(float)
    df["education"] = df["feature8"].astype(float)
    df["occupation"] = df["feature9"].astype(float)
    df["rating"] = df["feature10"].astype(float)

    return df


def analyze_relationship(df: pd.DataFrame):
    # Descriptive statistics
    grouped = df.groupby("children")
    mean_freq_by_children = grouped["affairs_freq"].mean().to_dict()
    prop_any_affair_by_children = grouped["any_affair"].mean().to_dict()

    # Keys will be 0 and 1
    mean_freq_children = mean_freq_by_children.get(1, np.nan)
    mean_freq_no_children = mean_freq_by_children.get(0, np.nan)
    prop_affair_children = prop_any_affair_by_children.get(1, np.nan)
    prop_affair_no_children = prop_any_affair_by_children.get(0, np.nan)

    delta_mean_freq = mean_freq_no_children - mean_freq_children
    delta_prop_affair = prop_affair_no_children - prop_affair_children

    # Mann-Whitney U test on the ordinal frequency score
    freq_with_children = df.loc[df["children"] == 1, "affairs_freq"]
    freq_no_children = df.loc[df["children"] == 0, "affairs_freq"]
    mannwhitney_u, mannwhitney_p = stats.mannwhitneyu(
        freq_with_children, freq_no_children, alternative="two-sided"
    )

    # Logistic regression for "any affair" ~ children + covariates
    X = df[
        [
            "children",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
            "male",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = df["any_affair"]

    logit_model = sm.Logit(y, X).fit(disp=False)

    child_coef = float(logit_model.params["children"])
    child_pval = float(logit_model.pvalues["children"])
    child_odds_ratio = float(np.exp(child_coef))

    results = {
        "mean_freq_children": float(mean_freq_children),
        "mean_freq_no_children": float(mean_freq_no_children),
        "prop_affair_children": float(prop_affair_children),
        "prop_affair_no_children": float(prop_affair_no_children),
        "delta_mean_freq_no_minus_yes_children": float(delta_mean_freq),
        "delta_prop_affair_no_minus_yes_children": float(delta_prop_affair),
        "mannwhitney_u": float(mannwhitney_u),
        "mannwhitney_p": float(mannwhitney_p),
        "logit_child_coef": child_coef,
        "logit_child_odds_ratio": child_odds_ratio,
        "logit_child_pval": child_pval,
        "logit_nobs": int(logit_model.nobs),
    }

    return results


def score_evidence(results: dict) -> int:
    """
    Map statistical evidence onto a 0-100 Likert scale where
    higher values mean stronger evidence that having children
    DECREASES engagement in extramarital affairs.
    """
    delta_prop = results["delta_prop_affair_no_minus_yes_children"]
    delta_mean_freq = results["delta_mean_freq_no_minus_yes_children"]
    odds_ratio = results["logit_child_odds_ratio"]
    p_logit = results["logit_child_pval"]
    p_mw = results["mannwhitney_p"]

    has_sig = (p_logit < 0.05) or (p_mw < 0.05)
    strong_sig = (p_logit < 0.01) or (p_mw < 0.01)

    direction_supports_decrease = (delta_prop > 0) and (delta_mean_freq > 0) and (
        odds_ratio < 1
    )
    direction_opposite = (delta_prop < 0) or (delta_mean_freq < 0) or (odds_ratio > 1)

    effect_prop = abs(delta_prop)
    effect_freq = abs(delta_mean_freq)

    # Strong "No" if evidence points in the opposite direction and is significant.
    if direction_opposite and strong_sig:
        return 10
    if direction_opposite and has_sig:
        return 20

    # Strong "Yes" if consistent direction and strong significance with non-trivial effect size.
    if direction_supports_decrease and strong_sig and (
        effect_prop >= 0.10 or odds_ratio <= 0.70
    ):
        return 85

    # Moderate "Yes"
    if direction_supports_decrease and has_sig and (
        effect_prop >= 0.05 or odds_ratio <= 0.85
    ):
        return 70

    # Weak evidence in the expected direction but not significant or very small effects.
    if direction_supports_decrease and not has_sig:
        return 55

    # Little to no evidence either way.
    if not has_sig and effect_prop < 0.05 and effect_freq < 0.5:
        return 30

    # Catch-all moderate "No" / inconclusive.
    return 40


def build_explanation(results: dict, score: int) -> str:
    mean_freq_children = results["mean_freq_children"]
    mean_freq_no_children = results["mean_freq_no_children"]
    prop_affair_children = results["prop_affair_children"]
    prop_affair_no_children = results["prop_affair_no_children"]
    delta_prop = results["delta_prop_affair_no_minus_yes_children"]
    delta_mean_freq = results["delta_mean_freq_no_minus_yes_children"]
    mannwhitney_p = results["mannwhitney_p"]
    odds_ratio = results["logit_child_odds_ratio"]
    child_pval = results["logit_child_pval"]

    if score >= 60:
        yes_no = "Yes"
    else:
        yes_no = "No"

    explanation = (
        f"Research question: Does having children decrease engagement in extramarital affairs?\n"
        f"Answer on a Yes/No scale: {yes_no} (Likert score {score} on 0–100).\n\n"
        f"Using the 601 married respondents, I compared affair behaviour between those with and without children.\n"
        f"- Descriptively, the mean extramarital-affair frequency score was "
        f"{mean_freq_children:.2f} for respondents with children and {mean_freq_no_children:.2f} for those without, "
        f"so the difference in means (no children minus children) was {delta_mean_freq:.2f}.\n"
        f"- The proportion reporting any affair in the past year was "
        f"{prop_affair_children:.3f} with children versus {prop_affair_no_children:.3f} without children, "
        f"a difference (no children minus children) of {delta_prop:.3f}.\n"
        f"- A Mann–Whitney U test on the ordinal affair-frequency score yielded p = {mannwhitney_p:.4f}, "
        f"and a logistic regression of having any affair on the presence of children (adjusting for age, years married, "
        f"gender, religiousness, education, occupation, and marital satisfaction) produced an odds ratio for children "
        f"of {odds_ratio:.3f} with p = {child_pval:.4f}.\n\n"
        f"Taken together, these results "
    )

    if score >= 80:
        explanation += (
            "provide strong and consistent statistical evidence that having children is associated with a lower "
            "likelihood and frequency of extramarital affairs in this sample."
        )
    elif score >= 60:
        explanation += (
            "indicate a statistically reliable association where having children is linked to somewhat lower engagement "
            "in extramarital affairs, although the effect size is only moderate."
        )
    elif score <= 20:
        explanation += (
            "provide statistically significant evidence in the opposite direction or no evidence that children reduce "
            "extramarital affairs; if anything, having children is not associated with fewer affairs in this dataset."
        )
    else:
        explanation += (
            "do not provide strong statistical evidence that having children reduces extramarital affairs: observed "
            "differences are small and/or not consistently statistically significant."
        )

    return explanation


def main():
    # Load metadata mainly for context (research question is fixed here).
    if Path("info.json").exists():
        with open("info.json", "r") as f:
            _ = json.load(f)

    df = load_data()
    results = analyze_relationship(df)
    score = score_evidence(results)
    explanation = build_explanation(results, score)

    conclusion = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

