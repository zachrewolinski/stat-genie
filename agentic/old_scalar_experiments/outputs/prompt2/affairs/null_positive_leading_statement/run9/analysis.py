import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DATA_FILE = Path("affairs.csv")
CONCLUSION_FILE = Path("conclusion.txt")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Ensure expected columns are present
    expected_cols = {
        "rownames",
        "affairs",
        "gender",
        "age",
        "yearsmarried",
        "children",
        "religiousness",
        "education",
        "occupation",
        "rating",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def summarize_affairs_by_children(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    group = df.groupby("children", observed=True)

    summary = group["affairs"].agg(["mean", "median", "std", "count"]).to_dict(orient="index")
    prop_any = group["has_affair"].mean().to_dict()

    # Convert numpy types to plain Python
    clean_summary = {}
    for key, stats in summary.items():
        clean_summary[key] = {
            "mean_affairs": float(stats["mean"]),
            "median_affairs": float(stats["median"]),
            "std_affairs": float(stats["std"]) if not np.isnan(stats["std"]) else None,
            "n": int(stats["count"]),
            "prop_with_any_affair": float(prop_any[key]),
        }
    return clean_summary


def run_logit_models(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    results = {}

    # Simple model: children only
    simple_formula = "has_affair ~ C(children)"
    simple_model = smf.logit(simple_formula, data=df).fit(disp=False)
    simple_params = simple_model.params
    simple_pvalues = simple_model.pvalues

    results["simple"] = {
        "params": simple_params.to_dict(),
        "pvalues": simple_pvalues.to_dict(),
    }

    # Adjusted model: include common covariates
    adjusted_formula = (
        "has_affair ~ C(children) + gender + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    # Encode gender as numeric for stability
    df["gender"] = df["gender"].astype("category")
    df["gender_num"] = df["gender"].cat.codes
    adjusted_formula = adjusted_formula.replace("gender", "gender_num")

    adjusted_model = smf.logit(adjusted_formula, data=df).fit(disp=False)
    adjusted_params = adjusted_model.params
    adjusted_pvalues = adjusted_model.pvalues

    results["adjusted"] = {
        "params": adjusted_params.to_dict(),
        "pvalues": adjusted_pvalues.to_dict(),
    }

    return results


def interpret_results(
    summary_by_children: dict,
    logit_results: dict,
) -> tuple[str, int, str]:
    """
    Return (response, confidence, explanation).
    """
    # Descriptive comparison
    desc_lines = []
    children_levels = sorted(summary_by_children.keys())
    for level in children_levels:
        stats = summary_by_children[level]
        desc_lines.append(
            f"children={level}: n={stats['n']}, "
            f"mean_affairs={stats['mean_affairs']:.3f}, "
            f"prop_with_any_affair={stats['prop_with_any_affair']:.3f}"
        )

    simple = logit_results["simple"]
    adjusted = logit_results["adjusted"]

    # Determine direction of effect from adjusted model if available
    params = adjusted["params"]
    pvalues = adjusted["pvalues"]

    # In the formula, statsmodels will typically use the alphabetically first
    # level of children as the reference. We check the parameter that compares
    # the non-reference level to the reference.
    children_param_keys = [k for k in params.keys() if "children" in k]

    effect_direction = None
    effect_significant = False
    effect_param_name = None

    if children_param_keys:
        effect_param_name = children_param_keys[0]
        beta = params[effect_param_name]
        pval = pvalues[effect_param_name]
        if beta < 0:
            effect_direction = "decrease"
        elif beta > 0:
            effect_direction = "increase"
        else:
            effect_direction = "none"
        effect_significant = pval < 0.05

    # Also inspect descriptive means
    mean_with_children = summary_by_children.get("yes", {}).get("mean_affairs")
    mean_without_children = summary_by_children.get("no", {}).get("mean_affairs")
    prop_with_children = summary_by_children.get("yes", {}).get("prop_with_any_affair")
    prop_without_children = summary_by_children.get("no", {}).get("prop_with_any_affair")

    desc_supports_decrease = False
    desc_supports_increase = False
    if mean_with_children is not None and mean_without_children is not None:
        if mean_with_children < mean_without_children and prop_with_children < prop_without_children:
            desc_supports_decrease = True
        elif mean_with_children > mean_without_children and prop_with_children > prop_without_children:
            desc_supports_increase = True

    # Decide on the answer
    if effect_direction == "decrease" and effect_significant and desc_supports_decrease:
        response = "Yes"
        confidence = 85
        reasoning = (
            "Both descriptive statistics and an adjusted logistic regression suggest that "
            "having children is associated with a lower likelihood of engaging in extramarital affairs. "
            f"In the adjusted model, the children-related coefficient ({effect_param_name}) is negative and "
            "statistically significant (p < 0.05), and individuals with children show lower mean affair scores "
            "and a lower proportion reporting any affair compared to those without children."
        )
    elif effect_direction == "decrease" and (desc_supports_decrease or effect_significant):
        response = "Yes"
        confidence = 70
        reasoning = (
            "The evidence leans toward having children being associated with fewer extramarital affairs. "
            "The children-related coefficient in the adjusted logistic regression is negative, and at least one "
            "of statistical significance or descriptive comparisons supports a decrease, although the signal is "
            "not uniformly strong across all checks."
        )
    elif effect_direction == "increase" and effect_significant and desc_supports_increase:
        response = "No"
        confidence = 85
        reasoning = (
            "Both descriptive statistics and an adjusted logistic regression suggest that "
            "having children is associated with a higher likelihood of engaging in extramarital affairs. "
            f"In the adjusted model, the children-related coefficient ({effect_param_name}) is positive and "
            "statistically significant (p < 0.05), and individuals with children show higher mean affair scores "
            "and a higher proportion reporting any affair compared to those without children."
        )
    elif effect_direction == "increase" and (desc_supports_increase or effect_significant):
        response = "No"
        confidence = 70
        reasoning = (
            "The evidence leans toward having children being associated with more extramarital affairs rather "
            "than fewer. The children-related coefficient in the adjusted logistic regression is positive, and "
            "at least one of statistical significance or descriptive comparisons supports an increase."
        )
    else:
        response = "No"
        confidence = 65
        reasoning = (
            "The data do not provide clear, consistent evidence that having children decreases engagement in "
            "extramarital affairs. Descriptive statistics and regression-based estimates either suggest very "
            "small differences between those with and without children or yield effects that are not statistically "
            "reliable at conventional levels."
        )

    detailed_explanation_lines = [
        "Descriptive comparison of affair involvement by children status:",
        *desc_lines,
        "",
        "Key findings from logistic regression models (outcome: any affair vs none):",
        f"Simple model (children only) parameters: {simple['params']}",
        f"Simple model p-values: {simple['pvalues']}",
        f"Adjusted model parameters: {adjusted['params']}",
        f"Adjusted model p-values: {adjusted['pvalues']}",
        "",
        f"Interpreted children effect direction in adjusted model: {effect_direction}, "
        f"statistically significant: {effect_significant}.",
        f"Descriptive means support decrease: {desc_supports_decrease}, "
        f"support increase: {desc_supports_increase}.",
    ]

    explanation = reasoning + "\n\n" + "\n".join(detailed_explanation_lines)
    return response, confidence, explanation


def save_conclusion(response: str, confidence: int, explanation: str) -> None:
    obj = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }
    CONCLUSION_FILE.write_text(json.dumps(obj, ensure_ascii=False))


def main() -> None:
    df = load_data()
    summary = summarize_affairs_by_children(df)
    logit_results = run_logit_models(df)
    response, confidence, explanation = interpret_results(summary, logit_results)
    save_conclusion(response, confidence, explanation)


if __name__ == "__main__":
    main()

