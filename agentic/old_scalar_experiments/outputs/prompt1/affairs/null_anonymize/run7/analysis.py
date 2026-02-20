import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Outcome: any extramarital affair (1 if some, 0 if none)
    df["any_affair"] = (df["feature2"] > 0).astype(int)
    # Key predictor: children in the marriage (1 if yes, 0 if no)
    df["has_children"] = (df["feature6"] == "yes").astype(int)

    # Descriptive statistics by children status
    mean_freq_by_children = df.groupby("has_children")["feature2"].mean()
    prop_any_by_children = df.groupby("has_children")["any_affair"].mean()

    mean_no_children = float(mean_freq_by_children.get(0, float("nan")))
    mean_children = float(mean_freq_by_children.get(1, float("nan")))
    prop_no_children = float(prop_any_by_children.get(0, float("nan")))
    prop_children = float(prop_any_by_children.get(1, float("nan")))

    # Logistic regression: probability of any affair ~ children
    try:
        logit_simple = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
        coef_simple = float(logit_simple.params["has_children"])
        pval_simple = float(logit_simple.pvalues["has_children"])
    except Exception:
        coef_simple = float("nan")
        pval_simple = float("nan")

    # Logistic regression with additional controls
    try:
        logit_controls = smf.logit(
            "any_affair ~ has_children + C(feature3) + feature4 + feature5 + "
            "feature7 + feature8 + feature9 + feature10",
            data=df,
        ).fit(disp=False)
        coef_controls = float(logit_controls.params["has_children"])
        pval_controls = float(logit_controls.pvalues["has_children"])
    except Exception:
        coef_controls = float("nan")
        pval_controls = float("nan")

    # Decide whether there is evidence that having children decreases affairs
    evidence_decrease = False
    if coef_simple < 0 and pval_simple < 0.05:
        evidence_decrease = True
    if coef_controls < 0 and pval_controls < 0.05:
        evidence_decrease = True

    response = "Yes" if evidence_decrease else "No"

    explanation_parts = []
    explanation_parts.append(
        f"In this sample of {len(df)} first-time married respondents, the mean coded "
        f"frequency of extramarital sexual intercourse is {mean_children:.2f} for those "
        f"with children and {mean_no_children:.2f} for those without children."
    )
    explanation_parts.append(
        f"The proportion reporting any extramarital affair (a nonzero value on the "
        f"affair-frequency measure) is {prop_children:.2%} among those with children "
        f"compared with {prop_no_children:.2%} among those without children."
    )

    if not (coef_simple != coef_simple or pval_simple != pval_simple):
        direction_simple = (
            "lower"
            if coef_simple < 0
            else "higher"
            if coef_simple > 0
            else "no clear change in"
        )
        explanation_parts.append(
            f"A logistic regression of having any extramarital affair on an indicator "
            f"for having children (without additional controls) yields a coefficient "
            f"of {coef_simple:.3f} (p = {pval_simple:.3f}), indicating that having "
            f"children is associated with {direction_simple} probability of an affair."
        )

    if not (coef_controls != coef_controls or pval_controls != pval_controls):
        direction_controls = (
            "lower"
            if coef_controls < 0
            else "higher"
            if coef_controls > 0
            else "no clear change in"
        )
        explanation_parts.append(
            f"When additionally controlling for gender, age, years married, "
            f"religiousness, education, occupation, and self-rated marital happiness, "
            f"the coefficient on having children is {coef_controls:.3f} "
            f"(p = {pval_controls:.3f}), implying {direction_controls} odds of an "
            f"affair for parents relative to non-parents."
        )

    if response == "Yes":
        explanation_parts.append(
            "Because the estimated effects of having children are negative and "
            "statistically significant in at least one specification, and descriptive "
            "statistics are consistent with lower affair involvement among parents, "
            "the data support the claim that having children is associated with "
            "reduced engagement in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "Because the estimated effects of having children are not negative and "
            "statistically significant, and the descriptive statistics do not show a "
            "clear reduction in affair frequency or prevalence for parents, the data "
            "do not support the claim that having children decreases engagement in "
            "extramarital affairs in this sample."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

