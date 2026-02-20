import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Based on info.json, the column named "age" actually encodes
    # frequency of extramarital intercourse in the past year,
    # and "religiousness" is a yes/no column: "Are there children in the marriage?"
    # We work with clearer aliases for analysis.
    df = df.copy()
    df["affair_freq"] = df["age"]
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)
    return df


def summarize_affairs_by_children(df: pd.DataFrame) -> dict:
    summary = {}

    # Basic means of frequency
    freq_means = df.groupby("has_children")["affair_freq"].mean()
    summary["mean_affair_freq_children"] = float(freq_means.get(1, np.nan))
    summary["mean_affair_freq_no_children"] = float(freq_means.get(0, np.nan))

    # Proportion with any affair
    prop_any = df.groupby("has_children")["any_affair"].mean()
    summary["prop_any_affair_children"] = float(prop_any.get(1, np.nan))
    summary["prop_any_affair_no_children"] = float(prop_any.get(0, np.nan))

    # Difference metrics
    summary["diff_mean_freq_children_minus_no_children"] = (
        summary["mean_affair_freq_children"] - summary["mean_affair_freq_no_children"]
    )
    summary["diff_prop_any_children_minus_no_children"] = (
        summary["prop_any_affair_children"] - summary["prop_any_affair_no_children"]
    )

    return summary


def logistic_regression_any_affair(df: pd.DataFrame) -> dict:
    # Logistic regression of having any affair on children, controlling for key covariates.
    # Map columns according to info.json descriptions:
    # - "children" column actually encodes years married.
    # - "rating" encodes religiousness level (1–5).
    # - "yearsmarried" encodes education level.
    # - "rownames" encodes occupation class.

    model_df = df.copy()
    model_df["years_married"] = model_df["children"]
    model_df["religiousness_level"] = model_df["rating"]
    model_df["education_level"] = model_df["yearsmarried"]
    model_df["occupation_code"] = model_df["rownames"]

    X = model_df[
        [
            "has_children",
            "years_married",
            "religiousness_level",
            "education_level",
            "occupation_code",
        ]
    ]
    X = sm.add_constant(X)
    y = model_df["any_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    coef = result.params["has_children"]
    p_value = result.pvalues["has_children"]
    odds_ratio = float(np.exp(coef))

    return {
        "coef_has_children": float(coef),
        "p_value_has_children": float(p_value),
        "odds_ratio_has_children": odds_ratio,
    }


def main() -> None:
    df = load_data(Path("affairs.csv"))

    summary = summarize_affairs_by_children(df)
    logit_stats = logistic_regression_any_affair(df)

    # Determine answer:
    # - If having children clearly reduces affairs (lower means and proportions,
    #   and a negative, at least moderately significant logit coefficient),
    #   answer "Yes"; otherwise "No".
    freq_diff = summary["diff_mean_freq_children_minus_no_children"]
    prop_diff = summary["diff_prop_any_children_minus_no_children"]
    coef = logit_stats["coef_has_children"]
    p_value = logit_stats["p_value_has_children"]

    # Heuristic decision rules
    children_have_lower_affairs = (freq_diff < 0) and (prop_diff < 0) and (coef < 0)
    statistically_supported = p_value < 0.05

    if children_have_lower_affairs and statistically_supported:
        response = "Yes"
        base_conf = 85
    elif children_have_lower_affairs and p_value < 0.1:
        response = "Yes"
        base_conf = 70
    else:
        # Either effects are very small/inconsistent or not statistically convincing.
        response = "No"
        base_conf = 80

    # Build explanation text
    explanation_parts = [
        "I compared extramarital affair involvement between respondents with and without children, using the provided survey data.",
        f"On average, the affair frequency score was "
        f"{summary['mean_affair_freq_children']:.3f} for respondents with children "
        f"and {summary['mean_affair_freq_no_children']:.3f} for those without children.",
        f"The proportion who reported at least one affair was "
        f"{summary['prop_any_affair_children']:.3f} among those with children and "
        f"{summary['prop_any_affair_no_children']:.3f} among those without children.",
        "I then fit a logistic regression model for having any affair on an indicator for having children,",
        "controlling for years married, religiousness level, education, and occupation code.",
        f"In this model, the coefficient for having children was {coef:.3f} with a p-value of {p_value:.3f}, ",
        f"corresponding to an odds ratio of {logit_stats['odds_ratio_has_children']:.3f}.",
    ]

    if response == "Yes":
        explanation_parts.append(
            "These results together suggest that having children is associated with a meaningful decrease "
            "in engagement in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "Taken together, these results do not provide strong evidence that having children decreases "
            "engagement in extramarital affairs; any differences appear small or statistically uncertain."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": base_conf,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

