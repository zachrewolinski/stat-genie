import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    expected = {
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
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def describe_difference(df: pd.DataFrame) -> str:
    # Basic comparison of affair counts by children status
    desc = []
    grouped = df.groupby("children")["affairs"]
    means = grouped.mean()
    medians = grouped.median()
    props_any = df.assign(any_affair=df["affairs"] > 0).groupby("children")["any_affair"].mean()

    for children_status in sorted(df["children"].unique()):
        mask = df["children"] == children_status
        n = mask.sum()
        mean_affairs = means.loc[children_status]
        median_affairs = medians.loc[children_status]
        prop_any = props_any.loc[children_status]
        desc.append(
            f"- Children = {children_status}: n = {n}, "
            f"mean affairs = {mean_affairs:.3f}, median = {median_affairs:.1f}, "
            f"proportion with any affair = {prop_any:.3f}"
        )
    return "\n".join(desc)


def fit_models(df: pd.DataFrame):
    # Binary outcome: any affair vs none
    df = df.copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Logistic regression with and without controls
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)

    formula_full = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + C(gender) + occupation + rating"
    )
    logit_full = smf.logit(formula_full, data=df).fit(disp=False)

    # For intensity among those with any affairs, use Poisson
    df_positive = df[df["affairs"] > 0].copy()
    poisson_full = None
    if len(df_positive) > 0:
        poisson_full = smf.poisson(
            "affairs ~ C(children) + age + yearsmarried + religiousness "
            "+ education + C(gender) + occupation + rating",
            data=df_positive,
        ).fit(disp=False)

    return logit_simple, logit_full, poisson_full


def summarize_children_effect(logit_full, poisson_full) -> dict:
    summary = {}

    # Children factor coefficient (baseline will be one of the levels)
    params = logit_full.params
    pvalues = logit_full.pvalues
    children_terms = {k: (params[k], pvalues[k]) for k in params.index if "C(children)" in k}

    summary["logit_children_terms"] = {
        name: {"coef": float(coef), "pvalue": float(pval)}
        for name, (coef, pval) in children_terms.items()
    }

    if poisson_full is not None:
        p_params = poisson_full.params
        p_pvalues = poisson_full.pvalues
        p_children_terms = {k: (p_params[k], p_pvalues[k]) for k in p_params.index if "C(children)" in k}
        summary["poisson_children_terms"] = {
            name: {"coef": float(coef), "pvalue": float(pval)}
            for name, (coef, pval) in p_children_terms.items()
        }
    else:
        summary["poisson_children_terms"] = {}

    return summary


def map_to_likert(effect_info: dict, desc_text: str) -> (int, str):
    """
    Map statistical evidence to a 0-100 Likert scale for the statement:
    'Having children decreases engagement in extramarital affairs.'
    """
    logit_terms = effect_info["logit_children_terms"]
    poisson_terms = effect_info["poisson_children_terms"]

    explanation_lines = []
    explanation_lines.append("Research question: Does having children decrease engagement in extramarital affairs?")
    explanation_lines.append("")
    explanation_lines.append("Descriptive comparison by children status:")
    explanation_lines.append(desc_text)
    explanation_lines.append("")

    # Determine direction and significance
    # In statsmodels with C(children), one level is baseline, and others are differences.
    # Here there are only two levels ('yes' and 'no'), so there will be a single contrast term.
    logit_coefs = list(logit_terms.values())
    logit_names = list(logit_terms.keys())

    if logit_coefs:
        coef, pval = logit_coefs[0]["coef"], logit_coefs[0]["pvalue"]
        name = logit_names[0]
        explanation_lines.append(
            f"Logistic regression (any affair) coefficient for {name}: "
            f"{coef:.3f} (p = {pval:.4f})."
        )
        logit_effect = (coef, pval)
    else:
        explanation_lines.append("Logistic regression found no identifiable children term.")
        logit_effect = None

    poisson_effect = None
    if poisson_terms:
        p_coef = list(poisson_terms.values())[0]["coef"]
        p_pval = list(poisson_terms.values())[0]["pvalue"]
        p_name = list(poisson_terms.keys())[0]
        explanation_lines.append(
            f"Poisson regression (frequency among those with affairs) coefficient for {p_name}: "
            f"{p_coef:.3f} (p = {p_pval:.4f})."
        )
        poisson_effect = (p_coef, p_pval)
    else:
        explanation_lines.append("Poisson regression for frequency either was not fit or had no children term.")

    explanation_lines.append("")

    # Heuristic mapping:
    # - If coefficients point to higher risk with children and are significant, strong 'No'.
    # - If coefficients point to lower risk with children and significant, strong 'Yes'.
    # - If not significant or mixed, closer to neutral.
    def effect_direction(effect):
        if effect is None:
            return None
        coef, pval = effect
        if pval < 0.05:
            return "negative" if coef < 0 else "positive"
        return "nonsignificant"

    logit_dir = effect_direction(logit_effect)
    poisson_dir = effect_direction(poisson_effect)

    explanation_lines.append(f"Logistic effect classification: {logit_dir}.")
    explanation_lines.append(f"Poisson effect classification: {poisson_dir}.")

    # Default neutral
    response = 50
    narrative = ""

    if logit_dir == "negative" and (poisson_dir in ("negative", None, "nonsignificant")):
        # Evidence that children reduce probability of any affair
        response = 80
        narrative = (
            "The logistic model suggests that having children is associated with a significantly lower "
            "probability of having any extramarital affair, supporting a 'Yes' answer, although effect size "
            "and other model caveats prevent an absolute conclusion."
        )
    elif logit_dir == "positive" and (poisson_dir in ("positive", None, "nonsignificant")):
        # Evidence that children increase probability of affairs
        response = 20
        narrative = (
            "The logistic model indicates that having children is associated with a significantly higher "
            "probability of having an extramarital affair, contradicting the idea that children decrease such behaviour."
        )
    elif logit_dir == "nonsignificant" and (poisson_dir == "nonsignificant" or poisson_dir is None):
        response = 45
        narrative = (
            "Neither the probability of having an affair nor the frequency among those who have affairs shows a "
            "statistically clear relationship with having children once other factors are controlled, providing "
            "little evidence that children materially decrease affairs."
        )
    else:
        # Mixed signals
        response = 40
        narrative = (
            "The models provide mixed or weak evidence regarding the impact of having children on extramarital affairs, "
            "and overall do not robustly support the claim that children decrease such behaviour."
        )

    explanation_lines.append("")
    explanation_lines.append(narrative)

    return int(response), "\n".join(explanation_lines)


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "affairs.csv"

    df = load_data(csv_path)
    desc_text = describe_difference(df)
    logit_simple, logit_full, poisson_full = fit_models(df)
    effect_info = summarize_children_effect(logit_full, poisson_full)
    response, explanation = map_to_likert(effect_info, desc_text)

    conclusion = {"response": int(response), "explanation": explanation}
    output_path = base_dir / "conclusion.txt"
    # Write exactly one JSON object, no extra whitespace beyond what json.dumps produces
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

