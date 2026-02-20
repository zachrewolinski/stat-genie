import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital affair in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Key predictor: children in the marriage (yes/no)
    df["has_children"] = df["feature6"].map({"yes": 1, "no": 0})

    # Drop rows with missing values in variables used
    model_vars = [
        "any_affair",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
    ]
    model_df = df[model_vars].dropna().copy()

    # Group-wise descriptive statistics
    mean_freq = df.groupby("feature6")["feature2"].mean()
    prop_any = df.groupby("feature6")["any_affair"].mean()
    count_by_children = df["feature6"].value_counts()

    # Defensive defaults in case categories are missing
    mean_with_children = float(mean_freq.get("yes", np.nan))
    mean_without_children = float(mean_freq.get("no", np.nan))
    prop_with_children = float(prop_any.get("yes", np.nan))
    prop_without_children = float(prop_any.get("no", np.nan))

    # Logistic regression: probability of any affair as a function of children,
    # controlling for basic demographics and relationship characteristics.
    logit_result = smf.logit(
        "any_affair ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",
        data=model_df,
    ).fit(disp=False)

    coef_children = float(logit_result.params.get("C(feature6)[T.yes]", np.nan))
    pval_children = float(logit_result.pvalues.get("C(feature6)[T.yes]", np.nan))

    conf_int = logit_result.conf_int().loc["C(feature6)[T.yes]"]
    ci_low, ci_high = float(conf_int[0]), float(conf_int[1])

    odds_ratio = float(np.exp(coef_children))
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Differences (children minus no-children)
    delta_mean = mean_with_children - mean_without_children
    delta_prop = prop_with_children - prop_without_children

    # Map statistical evidence to a 0-100 Likert-style score
    response_score = score_from_evidence(
        coef_children=coef_children,
        pval_children=pval_children,
        odds_ratio=odds_ratio,
        delta_mean=delta_mean,
        delta_prop=delta_prop,
    )

    explanation = build_explanation(
        n_total=len(df),
        count_by_children=count_by_children.to_dict(),
        mean_with_children=mean_with_children,
        mean_without_children=mean_without_children,
        prop_with_children=prop_with_children,
        prop_without_children=prop_without_children,
        coef_children=coef_children,
        pval_children=pval_children,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        delta_mean=delta_mean,
        delta_prop=delta_prop,
        response_score=response_score,
    )

    conclusion = {"response": int(response_score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


def score_from_evidence(
    coef_children: float,
    pval_children: float,
    odds_ratio: float,
    delta_mean: float,
    delta_prop: float,
) -> int:
    """
    Convert statistical evidence into a 0-100 score, where higher values mean
    stronger evidence that having children DECREASES engagement in extramarital affairs.
    """

    # Start from a neutral position.
    score = 50.0

    if np.isnan(coef_children) or np.isnan(pval_children):
        # If the model failed, fall back to descriptive differences.
        if delta_prop < 0:
            score = 60.0
        elif delta_prop > 0:
            score = 40.0
        else:
            score = 50.0
    else:
        # Direction and significance from the logistic regression coefficient.
        if coef_children < 0 and pval_children < 0.05:
            score = 80.0
        elif coef_children < 0 and pval_children < 0.10:
            score = 70.0
        elif coef_children < 0:
            score = 60.0
        elif coef_children > 0 and pval_children < 0.05:
            score = 20.0
        elif coef_children > 0 and pval_children < 0.10:
            score = 30.0
        elif coef_children > 0:
            score = 40.0
        else:
            score = 50.0

        # Adjust modestly for effect size (odds ratio) and descriptive differences.
        if odds_ratio < 0.7:
            score += 5.0
        if odds_ratio < 0.5:
            score += 5.0
        if odds_ratio > 1.3:
            score -= 5.0
        if odds_ratio > 1.5:
            score -= 5.0

        # Descriptive check: if all differences point in the same direction, nudge a bit more.
        if delta_prop < 0 and delta_mean < 0:
            score += 3.0
        if delta_prop > 0 and delta_mean > 0:
            score -= 3.0

    # Clamp to [0, 100] and return as integer.
    return int(max(0.0, min(100.0, round(score))))


def build_explanation(
    n_total: int,
    count_by_children: dict,
    mean_with_children: float,
    mean_without_children: float,
    prop_with_children: float,
    prop_without_children: float,
    coef_children: float,
    pval_children: float,
    odds_ratio: float,
    or_ci_low: float,
    or_ci_high: float,
    delta_mean: float,
    delta_prop: float,
    response_score: int,
) -> str:
    """
    Assemble a human-readable explanation summarizing the evidence and conclusion.
    """
    n_with_children = int(count_by_children.get("yes", 0))
    n_without_children = int(count_by_children.get("no", 0))

    explanation = (
        "I analyzed the affairs dataset (n={n_total}) to assess whether having children "
        "is associated with lower engagement in extramarital affairs. The key outcome "
        "was whether a respondent reported any extramarital sexual intercourse in the "
        "past year, derived from the frequency variable, and the main predictor was an "
        "indicator for children in the marriage.\n\n"
        "Descriptively, there were {n_with_children} respondents with children and "
        "{n_without_children} without children. The average affair frequency score was "
        "{mean_with_children:.3f} for respondents with children and "
        "{mean_without_children:.3f} for those without children, a difference of "
        "{delta_mean:.3f} (children minus no children). The proportion reporting any "
        "affair was {prop_with_children:.3f} among respondents with children versus "
        "{prop_without_children:.3f} among those without, a difference of "
        "{delta_prop:.3f}.\n\n"
        "To adjust for demographic and relationship factors (gender, age, years married, "
        "religiousness, education, occupation, and self-rated marital happiness), I fit "
        "a logistic regression model for any affair with a categorical indicator for "
        "having children. In this model, the coefficient for having children (yes versus "
        "no) was {coef_children:.3f}, corresponding to an odds ratio of "
        "{odds_ratio:.3f} with a 95% confidence interval of "
        "[{or_ci_low:.3f}, {or_ci_high:.3f}], and a p-value of {pval_children:.3f}.\n\n"
        "Taken together, the descriptive differences and the adjusted odds ratio "
        "indicate that having children {direction} associated with engagement in "
        "extramarital affairs in this sample. Based on the size and statistical "
        "strength of this association, I summarize my answer on a 0–100 scale (where "
        "100 is a strong 'yes' that children decrease affairs and 0 is a strong 'no' or "
        "evidence of the opposite) as {response_score:d}."
    )

    if odds_ratio < 1:
        direction = "is modestly"
    elif odds_ratio > 1:
        direction = "is not"
    else:
        direction = "is only weakly"

    return explanation.format(
        n_total=n_total,
        n_with_children=n_with_children,
        n_without_children=n_without_children,
        mean_with_children=mean_with_children,
        mean_without_children=mean_without_children,
        delta_mean=delta_mean,
        prop_with_children=prop_with_children,
        prop_without_children=prop_without_children,
        delta_prop=delta_prop,
        coef_children=coef_children,
        odds_ratio=odds_ratio,
        or_ci_low=or_ci_low,
        or_ci_high=or_ci_high,
        pval_children=pval_children,
        response_score=response_score,
        direction=direction,
    )


if __name__ == "__main__":
    main()

