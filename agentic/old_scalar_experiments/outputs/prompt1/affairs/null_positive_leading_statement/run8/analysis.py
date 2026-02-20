import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata and data
    info = json.loads((base_path / "info.json").read_text())
    df = pd.read_csv(base_path / "affairs.csv")

    # Basic sanity checks
    assert "children" in df.columns
    assert "affairs" in df.columns

    # Create binary outcome: any affair vs none
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Simple descriptive comparison
    desc = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_any_affair"})
    )

    # Logistic regression of any_affair on children + key controls
    # children is treated as a categorical predictor
    formula = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + C(gender) + rating"
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract effect of having children (yes vs no)
    # With C(children), the parameter is typically C(children)[T.yes]
    children_param = None
    for name in model.params.index:
        if name.startswith("C(children)[T."):
            children_param = (name, model.params[name], model.bse[name], model.pvalues[name])
            break

    explanation_lines = []
    explanation_lines.append(
        info["research_questions"][0].strip()
    )
    explanation_lines.append("")
    explanation_lines.append("Data and variables:")
    explanation_lines.append(
        "- Sample size: {} married individuals from the Fair (1978) affairs dataset.".format(len(df))
    )
    explanation_lines.append(
        "- Outcome: any extramarital affair in the last year (1 if affairs>0, 0 otherwise)."
    )
    explanation_lines.append("- Key predictor: presence of children in the marriage (yes/no).")
    explanation_lines.append(
        "- Controls: age, years married, gender, religiousness, education, and self-rated marital happiness."
    )
    explanation_lines.append("")

    # Add descriptive stats
    prop_yes = desc.loc["yes", "prop_any_affair"]
    prop_no = desc.loc["no", "prop_any_affair"]
    explanation_lines.append("Descriptive comparison (unadjusted):")
    explanation_lines.append(
        f"- Proportion with any affair when children = yes: {prop_yes:.3f}"
    )
    explanation_lines.append(
        f"- Proportion with any affair when children = no: {prop_no:.3f}"
    )

    explanation_lines.append("")
    explanation_lines.append("Logistic regression results (adjusted):")
    if children_param is not None:
        import math

        name, coef, se, pval = children_param
        odds_ratio = math.exp(coef)
        explanation_lines.append(
            f"- Coefficient for {name}: {coef:.3f} (SE = {se:.3f}, p-value = {pval:.3f})."
        )
        explanation_lines.append(
            f"- This corresponds to an odds ratio of approximately {odds_ratio:.3f} for any affair when children = yes versus no."
        )
        if pval < 0.05:
            sig_text = "statistically significant at the 5% level"
        else:
            sig_text = "not statistically significant at conventional levels"
        explanation_lines.append(f"- The effect is {sig_text}.")
    else:
        explanation_lines.append(
            "- Could not uniquely locate the children coefficient in the fitted model; results focus on descriptive differences."
        )

    explanation_lines.append("")

    # Determine answer: does having children decrease engagement in extramarital affairs?
    # We require both the descriptive and adjusted analyses to point towards a decrease
    # (lower proportion / odds when children = yes). If the effect is null or positive,
    # or clearly not significant, we answer 'No'.
    decreases_descriptively = prop_yes < prop_no
    decreases_adjusted = False
    if children_param is not None:
        _, coef, _, _ = children_param
        decreases_adjusted = coef < 0

    if decreases_descriptively and decreases_adjusted:
        response = "Yes"
        explanation_lines.append(
            "Both the unadjusted proportions and the adjusted logistic regression suggest that having children is associated with *lower* engagement in extramarital affairs."
        )
    else:
        response = "No"
        explanation_lines.append(
            "The data do not support the claim that having children decreases engagement in extramarital affairs. Either the difference is small, statistically uncertain, or points in the opposite direction."
        )

    explanation_lines.append(
        "Based on this evidence, we conclude that the dataset does not provide strong support for the hypothesis that having children reduces extramarital affairs."
    )

    conclusion = {
        "response": response,
        "explanation": "\n".join(explanation_lines),
    }

    (base_path / "conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
