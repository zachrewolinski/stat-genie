import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Binary indicator for any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    # Binary numeric indicator for having children
    df["children_binary"] = (df["children"] == "yes").astype(int)
    # Drop rows with missing values in variables used in the model (if any)
    model_vars = [
        "has_affair",
        "children_binary",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender",
    ]
    df = df[model_vars].dropna()
    # One-hot encode gender with male as reference if present
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)
    return df


def descriptive_stats(df: pd.DataFrame) -> dict:
    # Descriptive comparison by children status
    desc = {}
    grouped = df.groupby("children")
    for children_value, sub in grouped:
        desc[children_value] = {
            "n": int(len(sub)),
            "mean_affairs": float(sub["affairs"].mean()),
            "prop_any_affair": float((sub["affairs"] > 0).mean()),
        }
    return desc


def fit_logistic(df_raw: pd.DataFrame) -> dict:
    df = df_raw.copy()
    # Align with prepare_data encoding
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_binary"] = (df["children"] == "yes").astype(int)
    model_df = df[
        [
            "has_affair",
            "children_binary",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
            "gender",
        ]
    ].dropna()
    model_df = pd.get_dummies(model_df, columns=["gender"], drop_first=True)

    y = model_df["has_affair"]
    X = model_df.drop(columns=["has_affair"])
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues

    coef_children = params["children_binary"]
    pvalue_children = pvalues["children_binary"]
    ci_lower, ci_upper = conf.loc["children_binary"]

    odds_ratio = float(np.exp(coef_children))
    ci_lower_or = float(np.exp(ci_lower))
    ci_upper_or = float(np.exp(ci_upper))

    return {
        "coef_children": float(coef_children),
        "pvalue_children": float(pvalue_children),
        "odds_ratio_children": odds_ratio,
        "ci_lower_or_children": ci_lower_or,
        "ci_upper_or_children": ci_upper_or,
        "n_obs": int(result.nobs),
    }


def assess_hypothesis(
    descriptives: dict,
    logistic_results: dict,
) -> tuple[str, str]:
    """
    Decide whether data support the claim that having children decreases engagement in affairs.
    """
    # Descriptive comparison
    mean_affairs_children = None
    mean_affairs_no_children = None
    prop_affair_children = None
    prop_affair_no_children = None

    if "yes" in descriptives and "no" in descriptives:
        mean_affairs_children = descriptives["yes"]["mean_affairs"]
        mean_affairs_no_children = descriptives["no"]["mean_affairs"]
        prop_affair_children = descriptives["yes"]["prop_any_affair"]
        prop_affair_no_children = descriptives["no"]["prop_any_affair"]

    coef_children = logistic_results["coef_children"]
    pvalue_children = logistic_results["pvalue_children"]
    odds_ratio = logistic_results["odds_ratio_children"]
    ci_lower_or = logistic_results["ci_lower_or_children"]
    ci_upper_or = logistic_results["ci_upper_or_children"]
    n_obs = logistic_results["n_obs"]

    alpha = 0.05

    # Determine support for the hypothesis based primarily on the multivariable model,
    # using descriptives as supporting context.
    if coef_children < 0 and pvalue_children < alpha and ci_upper_or < 1.0:
        response = "Yes"
    else:
        response = "No"

    explanation_parts = []
    if mean_affairs_children is not None and mean_affairs_no_children is not None:
        explanation_parts.append(
            f"Mean affair score was {mean_affairs_no_children:.3f} for couples without children "
            f"and {mean_affairs_children:.3f} for couples with children."
        )
    if prop_affair_children is not None and prop_affair_no_children is not None:
        explanation_parts.append(
            f"The proportion with any affair was {prop_affair_no_children:.3f} without children "
            f"versus {prop_affair_children:.3f} with children."
        )

    direction = "lower" if coef_children < 0 else "higher"
    explanation_parts.append(
        "In a logistic regression of having any affair on an indicator for having children, "
        "adjusting for age, years married, religiousness, education, occupation, and self-rated marital happiness "
        f"(N = {n_obs}), the coefficient for having children was {coef_children:.3f} "
        f"(odds ratio = {odds_ratio:.3f}, 95% CI {ci_lower_or:.3f}–{ci_upper_or:.3f}, p = {pvalue_children:.3f}), "
        f"indicating {direction} odds of an affair for couples with children compared with those without."
    )

    if response == "Yes":
        explanation_parts.append(
            "Because the effect of having children is negative and statistically significant with the entire "
            "confidence interval for the odds ratio below 1, the data support the claim that having children "
            "is associated with decreased engagement in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "However, this effect is not both clearly negative and statistically significant at the 5% level "
            "with the confidence interval entirely below 1, so the data do not provide strong evidence that "
            "having children decreases engagement in extramarital affairs. At best, any association appears "
            "weak and should be interpreted cautiously in this observational study."
        )

    explanation = " ".join(explanation_parts)
    return response, explanation


def main() -> None:
    csv_path = Path("affairs.csv")
    df = pd.read_csv(csv_path)

    descriptives = descriptive_stats(df)
    logistic_results = fit_logistic(df)

    response, explanation = assess_hypothesis(descriptives, logistic_results)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

