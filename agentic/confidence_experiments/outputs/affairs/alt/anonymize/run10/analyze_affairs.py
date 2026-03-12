import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Key variables
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    df["children"] = (df["feature6"] == "yes").astype(int)  # 1 = children present
    df["is_male"] = (df["feature3"] == "male").astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_freq=("feature2", "mean"),
            median_freq=("feature2", "median"),
            prop_any=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
        .reset_index()
    )

    # Two-sample z-test for difference in proportions of any affair
    counts = df.groupby("children")["any_affair"].sum().values
    nobs = df.groupby("children")["any_affair"].count().values
    z_stat, p_prop = proportions_ztest(count=counts, nobs=nobs)

    # Logistic regression for any affair, controlling for covariates
    y = df["any_affair"]
    X = df[
        [
            "children",
            "is_male",
            "feature4",  # age
            "feature5",  # years married
            "feature7",  # religiousness
            "feature8",  # education
            "feature9",  # occupation
            "feature10",  # marriage rating
        ]
    ]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X).fit(disp=False)
    children_coef = float(logit_model.params["children"])
    children_pvalue = float(logit_model.pvalues["children"])

    # Predicted probabilities at mean covariates with and without children
    mean_cov = X.mean()
    effects = {}
    for child_flag in (0, 1):
        vec = mean_cov.copy()
        vec["children"] = child_flag
        lin_pred = float((logit_model.params * vec).sum())
        prob = 1.0 / (1.0 + np.exp(-lin_pred))
        effects[child_flag] = prob

    diff_prob = effects[1] - effects[0]

    # Heuristic mapping to Likert 0-100
    # Start from neutral 50 and adjust by direction, effect size and significance.
    response_score = 50

    # Direction: negative diff_prob means children reduce probability of any affair.
    if diff_prob < 0 and children_pvalue < 0.05:
        # Strong evidence that children reduce affairs
        magnitude = min(abs(diff_prob), 0.25)  # cap extreme probabilities
        response_score = int(75 + 25 * (magnitude / 0.25))
    elif diff_prob < 0 and children_pvalue < 0.1:
        # Weak-to-moderate evidence for a reduction
        magnitude = min(abs(diff_prob), 0.25)
        response_score = int(60 + 15 * (magnitude / 0.25))
    elif diff_prob > 0 and children_pvalue < 0.05:
        # Strong evidence in the opposite direction
        magnitude = min(abs(diff_prob), 0.25)
        response_score = int(25 - 25 * (magnitude / 0.25))
    elif diff_prob > 0 and children_pvalue < 0.1:
        # Weak-to-moderate opposite effect
        magnitude = min(abs(diff_prob), 0.25)
        response_score = int(40 - 15 * (magnitude / 0.25))
    else:
        # No clear evidence either way
        response_score = 40 if diff_prob < 0 else 60

    response_score = max(0, min(100, int(response_score)))

    # Build textual explanation with main evidence
    # Extract descriptive stats in a simple form
    stats_children0 = desc.loc[desc["children"] == 0].iloc[0]
    stats_children1 = desc.loc[desc["children"] == 1].iloc[0]

    explanation = {
        "research_question": "Does having children decrease (if at all) the engagement in extramarital affairs?",
        "summary": (
            "Analyzed whether the presence of children in a marriage is associated "
            "with lower engagement in extramarital affairs using descriptive "
            "statistics, a two-sample test for proportions, and a multivariable "
            "logistic regression model."
        ),
        "descriptive_stats": {
            "no_children": {
                "n": int(stats_children0["n"]),
                "mean_affair_frequency": float(stats_children0["mean_freq"]),
                "median_affair_frequency": float(stats_children0["median_freq"]),
                "proportion_any_affair": float(stats_children0["prop_any"]),
            },
            "with_children": {
                "n": int(stats_children1["n"]),
                "mean_affair_frequency": float(stats_children1["mean_freq"]),
                "median_affair_frequency": float(stats_children1["median_freq"]),
                "proportion_any_affair": float(stats_children1["prop_any"]),
            },
            "proportions_z_test_p_value": float(p_prop),
        },
        "logistic_regression": {
            "children_coef_log_odds": children_coef,
            "children_p_value": children_pvalue,
            "predicted_prob_any_affair_no_children_at_means": float(effects[0]),
            "predicted_prob_any_affair_with_children_at_means": float(effects[1]),
            "difference_in_predicted_probabilities_children_minus_no_children": float(
                diff_prob
            ),
        },
        "interpretation": (
            "The sign of the children coefficient and the difference in predicted "
            "probabilities indicate whether having children is associated with a "
            "higher or lower likelihood of engaging in any extramarital affair, "
            "after controlling for gender, age, years married, religiousness, "
            "education, occupation, and self-rated marital happiness. The p-values "
            "from the logistic regression and the proportions test quantify the "
            "strength of statistical evidence for this association."
        ),
        "response_scale_guidance": (
            "A response near 0 would represent strong evidence that having children "
            "does not decrease (and may increase) engagement in affairs, while a "
            "response near 100 would represent strong evidence that having children "
            "substantially decreases engagement. Values around 50 reflect little or "
            "no clear evidence either way."
        ),
    }

    conclusion = {
        "response": response_score,
        "explanation": json.dumps(explanation, ensure_ascii=False),
    }

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

