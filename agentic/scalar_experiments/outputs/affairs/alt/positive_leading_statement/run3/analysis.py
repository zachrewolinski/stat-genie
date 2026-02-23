import json
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load research metadata
    with open("info.json", "r") as f:
        info = json.load(f)
    research_questions = info.get("research_questions", [])
    research_question = research_questions[0] if research_questions else ""

    # Load dataset
    df = pd.read_csv("affairs.csv")
    df = df.copy()

    # Define outcome and key predictor
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_binary"] = df["children"].map({"yes": 1, "no": 0})
    df = df.dropna(subset=["children_binary"])
    df["children_binary"] = df["children_binary"].astype(int)

    # Descriptive statistics by children status
    group = df.groupby("children")
    mean_affairs = group["affairs"].mean()
    prop_affair = group["has_affair"].mean()

    # Fit logistic regression for having any affair
    logit_result: Optional[object] = None
    model_description = ""
    coef_children: Optional[float] = None
    pval_children: Optional[float] = None
    odds_ratio: Optional[float] = None

    try:
        # Full model with standard controls used in prior analyses of this dataset
        formula = (
            "has_affair ~ children_binary + age + yearsmarried + "
            "religiousness + education + occupation + rating + C(gender)"
        )
        logit_model = smf.logit(formula, data=df)
        logit_result = logit_model.fit(disp=0)
        model_description = (
            "logistic regression of having any extramarital affair (yes/no) "
            "on having children, controlling for age, years married, gender, "
            "religiousness, education, occupation, and self-rated marriage quality."
        )
    except Exception:
        # Fallback: simple logistic regression with children only
        try:
            formula = "has_affair ~ children_binary"
            logit_model = smf.logit(formula, data=df)
            logit_result = logit_model.fit(disp=0)
            model_description = (
                "simple logistic regression of having any extramarital affair (yes/no) "
                "on having children only."
            )
        except Exception:
            logit_result = None

    if logit_result is not None:
        params = logit_result.params
        pvalues = logit_result.pvalues
        coef_children = float(params["children_binary"])
        pval_children = float(pvalues["children_binary"])
        odds_ratio = float(np.exp(coef_children))

        # Average predicted probabilities with and without children,
        # holding the rest of the covariate distribution as observed.
        base = df.copy()
        base_no_children = base.copy()
        base_no_children["children_binary"] = 0
        base_children = base.copy()
        base_children["children_binary"] = 1
        pred_no_children = float(logit_result.predict(base_no_children).mean())
        pred_children = float(logit_result.predict(base_children).mean())
    else:
        # Fall back to descriptive differences if regression fails completely
        if "no" in prop_affair.index:
            pred_no_children = float(prop_affair["no"])
        else:
            pred_no_children = float(df["has_affair"].mean())
        if "yes" in prop_affair.index:
            pred_children = float(prop_affair["yes"])
        else:
            pred_children = float(df["has_affair"].mean())

    diff_prob = pred_children - pred_no_children  # change with children vs no children

    raw_diff_prob: Optional[float] = None
    raw_diff_mean_affairs: Optional[float] = None
    if "yes" in prop_affair.index and "no" in prop_affair.index:
        raw_diff_prob = float(prop_affair["yes"] - prop_affair["no"])
    if "yes" in mean_affairs.index and "no" in mean_affairs.index:
        raw_diff_mean_affairs = float(mean_affairs["yes"] - mean_affairs["no"])

    # Map statistical evidence to a 0–100 Likert response
    if pval_children is None:
        base_certainty = 0.5
    else:
        if pval_children < 0.001:
            base_certainty = 0.9
        elif pval_children < 0.01:
            base_certainty = 0.8
        elif pval_children < 0.05:
            base_certainty = 0.7
        elif pval_children < 0.1:
            base_certainty = 0.6
        else:
            base_certainty = 0.5

    # diff_prob < 0 means children lower the predicted probability of any affair
    if diff_prob < 0:
        belief_yes = base_certainty
    elif diff_prob > 0:
        belief_yes = 1.0 - base_certainty
    else:
        belief_yes = 0.5

    response = int(round(100 * belief_yes))
    response = max(0, min(100, response))

    # Qualitative conclusion aligned with the Likert score
    if belief_yes > 0.6:
        qualitative = (
            "Yes: in this dataset, having children is associated with a lower "
            "likelihood of engaging in extramarital affairs."
        )
    elif belief_yes < 0.4:
        qualitative = (
            "No: in this dataset, having children is not associated with a lower "
            "likelihood of engaging in extramarital affairs and may even be "
            "associated with a higher likelihood."
        )
    else:
        qualitative = (
            "Uncertain: this dataset does not provide strong evidence that having "
            "children meaningfully changes the likelihood of engaging in "
            "extramarital affairs."
        )

    # Build explanation text
    lines = []
    if research_question:
        lines.append(f"Research question: {research_question.strip()}")
    lines.append(
        "Data: 601 first-marriage respondents from Ray Fair's extramarital affairs study, "
        "with 'affairs' coded as the frequency of extramarital sexual intercourse in the "
        "past year and 'children' recorded as 'yes' or 'no'."
    )
    lines.append(
        "Descriptive comparison: mean affair frequency by children status is "
        f"{mean_affairs.to_dict()}, and the proportion having any affair (affairs > 0) "
        f"is {prop_affair.to_dict()} by children status."
    )
    if raw_diff_prob is not None and raw_diff_mean_affairs is not None:
        lines.append(
            "Raw differences (children 'yes' minus 'no'): the probability of having any "
            f"affair differs by {raw_diff_prob:.3f}, and the mean affair frequency "
            f"differs by {raw_diff_mean_affairs:.3f}."
        )
    if logit_result is not None and coef_children is not None and pval_children is not None:
        lines.append(f"Model: {model_description}")
        lines.append(
            "In the regression, the coefficient on having children (coded 1 vs 0) is "
            f"{coef_children:.3f} with p-value {pval_children:.3g}, corresponding to an "
            f"odds ratio of {odds_ratio:.3f}."
        )
        lines.append(
            "Using the fitted model, the average predicted probability of having any "
            f"affair is {pred_no_children:.3f} for otherwise similar respondents without "
            f"children and {pred_children:.3f} for otherwise similar respondents with "
            f"children (difference {diff_prob:.3f})."
        )
    else:
        lines.append(
            "No regression model could be reliably fit; the conclusion relies on "
            "descriptive differences alone."
        )
    lines.append(f"Conclusion: {qualitative}")
    lines.append(
        "The Likert-scale response value of "
        f"{response} reflects the strength and direction of this evidence, combining the "
        "effect size and the statistical significance of the 'children' term: values "
        "closer to 100 indicate stronger support that having children decreases affairs, "
        "values closer to 0 indicate stronger evidence against that claim, and values "
        "near 50 reflect limited or inconclusive evidence."
    )

    explanation = " ".join(lines)

    output = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(output, f)

    # Print a human-readable summary for interactive inspection (not written to conclusion.txt).
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

