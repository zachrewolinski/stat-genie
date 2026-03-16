import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Key predictor: children in the marriage (1 = yes, 0 = no)
    df["has_children"] = (df["feature6"].astype(str).str.lower() == "yes").astype(int)

    # Basic descriptives by children status
    group = df.groupby("has_children", observed=True)
    desc = group["feature2"].agg(["mean", "std", "count"]).rename_axis("has_children")
    prop_affair = group["has_affair"].mean()

    # Logistic regression for any affair, controlling for available covariates
    formula = (
        "has_affair ~ has_children + feature4 + feature5 + "
        "C(feature3) + feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    child_coef = float(logit_model.params["has_children"])
    child_p = float(logit_model.pvalues["has_children"])
    child_or = float(np.exp(child_coef))

    # Map evidence to a 0–100 Likert score where
    # 0 = strong "No, children do NOT decrease affairs"
    # 100 = strong "Yes, children DO decrease affairs"
    if child_p < 0.01 and child_coef < 0:
        score = 90
    elif child_p < 0.05 and child_coef < 0:
        score = 80
    elif child_p < 0.1 and child_coef < 0:
        score = 65
    elif child_p >= 0.1 and child_coef < 0:
        score = 40
    elif child_p >= 0.1 and child_coef > 0:
        score = 20
    else:
        # Statistically significant coefficient, but in the opposite direction
        # of the research hypothesis (children associated with MORE affairs).
        score = 5

    # Adjust slightly based on effect size (odds ratio magnitude)
    if child_coef < 0:
        if child_or < 0.6:
            score += 5
        if child_or < 0.4:
            score += 5
    else:
        if child_or > 1.4:
            score -= 5
        if child_or > 2.0:
            score -= 5

    score = int(min(max(score, 0), 100))

    # Human-readable labels
    child_labels = {0: "no children", 1: "children"}

    mean_no_children = float(desc.loc[0, "mean"])
    mean_children = float(desc.loc[1, "mean"])
    prop_no_children = float(prop_affair.loc[0])
    prop_children = float(prop_affair.loc[1])

    # Build explanation
    direction = (
        "decrease"
        if child_coef < 0
        else "increase"
    )
    yes_no = "Yes" if score >= 50 and child_coef < 0 else "No"

    explanation = (
        f"Research question: Does having children decrease engagement in extramarital affairs?\n\n"
        f"Data and outcome: Using 601 first-marriage respondents from the Fair affairs dataset, "
        f"I defined engagement in extramarital affairs as reporting any non-zero frequency of extramarital "
        f"sexual intercourse during the past year (feature2 > 0). The key predictor is whether there are "
        f"children in the marriage (feature6).\n\n"
        f"Descriptive evidence: Among respondents with {child_labels[0]}, the mean affair-frequency code "
        f"is {mean_no_children:.2f} and {prop_no_children:.1%} report at least one extramarital encounter. "
        f"Among respondents with {child_labels[1]}, the mean affair-frequency code is {mean_children:.2f} and "
        f"{prop_children:.1%} report at least one encounter.\n\n"
        f"Model-based evidence: I fit a logistic regression for any affair (yes/no) on the presence of children, "
        f"controlling for age, years married, gender, religiousness, education, occupation, and self-rated marital "
        f"happiness. The coefficient on the children indicator is {child_coef:.3f}, corresponding to an odds ratio "
        f"of {child_or:.2f} with p-value {child_p:.3f}. This means that, holding the other factors constant, "
        f"having children is associated with a {direction} in the odds of engaging in an extramarital affair.\n\n"
        f"Conclusion and scale placement: Overall answer: {yes_no}—"
    )

    if yes_no == "Yes":
        explanation += (
            "there is evidence that having children is associated with lower engagement in extramarital affairs, "
            "although the effect size and statistical strength are reflected in the numeric score. "
        )
    else:
        if child_coef < 0:
            explanation += (
                "while the point estimate suggests fewer affairs among those with children, the association is "
                "not strongly supported by the data once other variables are taken into account. "
            )
        else:
            explanation += (
                "the data do not support the hypothesis that children reduce affairs; if anything, the estimated "
                "relationship points in the opposite direction or is too weak relative to sampling noise. "
            )

    explanation += (
        f"I therefore place my answer at {score} on a 0–100 Likert scale, where 0 means a strong 'No' and "
        f"100 means a strong 'Yes'. This score jointly reflects the direction of the estimated effect, the "
        f"magnitude of the odds ratio, and the statistical significance level."
    )

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

