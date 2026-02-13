import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["has_affair"] = (df["feature2"] > 0).astype(int)

    # Indicator for presence of children in the marriage (1 = yes, 0 = no)
    df["child"] = df["feature6"].str.lower().eq("yes").astype(int)

    model_cols = [
        "has_affair",
        "child",
        "feature3",
        "feature4",
        "feature5",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
    ]
    df_model = df[model_cols].dropna()

    # Descriptive comparison: affair rates by children status
    group = df_model.groupby("child")["has_affair"].agg(["mean", "sum", "count"])

    # Guard against any unexpected missing groups
    child_yes_mean = float(group.loc[1, "mean"]) if 1 in group.index else float("nan")
    child_no_mean = float(group.loc[0, "mean"]) if 0 in group.index else float("nan")
    total_n = int(group["count"].sum())

    # Logistic regression: any affair ~ children + controls
    model = smf.logit(
        "has_affair ~ child + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10",  # noqa: E501
        data=df_model,
    )
    result = model.fit(disp=0)

    coef_child = float(result.params["child"])
    p_child = float(result.pvalues["child"])
    or_child = float(np.exp(coef_child))

    # Map evidence onto 0–100 Likert-style response
    if coef_child < 0:
        if p_child < 0.05:
            response = 80
        elif p_child < 0.1:
            response = 65
        else:
            response = 55
    elif coef_child > 0:
        if p_child < 0.05:
            response = 20
        elif p_child < 0.1:
            response = 35
        else:
            response = 45
    else:
        response = 50

    response = int(min(100, max(0, response)))

    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs? "
    )
    explanation_parts.append(
        f"In the data (n={total_n}), {child_yes_mean * 100:.1f}% of spouses with children "
        f"reported at least one extramarital affair in the past year, compared with "
        f"{child_no_mean * 100:.1f}% of spouses without children. "
    )
    explanation_parts.append(
        "I fit a logistic regression predicting any affair from an indicator for having children, "
        "controlling for gender, age, years married, religiousness, education, occupation, and "
        "self-rated marriage quality. "
    )
    explanation_parts.append(
        f"The estimated coefficient for having children was {coef_child:.3f}, corresponding to an "
        f"odds ratio of {or_child:.3f} with p-value {p_child:.3f}. "
    )

    if coef_child < 0:
        if p_child < 0.05:
            explanation_parts.append(
                "This negative, statistically significant coefficient indicates that, after adjusting "
                "for other observed characteristics, having children is associated with a lower "
                "likelihood of engaging in extramarital affairs. "
            )
            explanation_parts.append(
                f"Accordingly, I answer that having children does decrease engagement in extramarital "
                f"affairs, with strength {response}/100 on the 0–100 scale."
            )
        elif p_child < 0.1:
            explanation_parts.append(
                "The coefficient is negative with marginal statistical evidence (p < 0.10) that having "
                "children is associated with fewer extramarital affairs, though uncertainty remains "
                "substantial. "
            )
            explanation_parts.append(
                f"I therefore lean toward the view that children decrease engagement in extramarital "
                f"affairs, with moderate strength {response}/100 on the 0–100 scale."
            )
        else:
            explanation_parts.append(
                "The coefficient is negative but not statistically distinguishable from zero at "
                "conventional levels, so evidence that children reduce extramarital affairs is weak. "
            )
            explanation_parts.append(
                f"I therefore give a slightly 'Yes'-leaning answer that having children decreases "
                f"engagement in extramarital affairs, with strength {response}/100 on the 0–100 scale."
            )
    elif coef_child > 0:
        if p_child < 0.05:
            explanation_parts.append(
                "This positive, statistically significant coefficient indicates that, after adjusting "
                "for other observed characteristics, having children is associated with a higher "
                "likelihood of engaging in extramarital affairs. "
            )
            explanation_parts.append(
                f"Accordingly, I answer that having children do not decrease engagement in "
                f"extramarital affairs (if anything, they are associated with more), with strength "
                f"{response}/100 on the 0–100 scale."
            )
        elif p_child < 0.1:
            explanation_parts.append(
                "The coefficient is positive with marginal statistical evidence (p < 0.10) that having "
                "children is associated with more extramarital affairs, though uncertainty remains "
                "substantial. "
            )
            explanation_parts.append(
                f"I therefore lean toward the view that children do not decrease engagement in "
                f"extramarital affairs, with moderate strength {response}/100 on the 0–100 scale."
            )
        else:
            explanation_parts.append(
                "The coefficient is positive but not statistically distinguishable from zero at "
                "conventional levels, so evidence that children reduce extramarital affairs is weak "
                "and, if anything, slightly in the opposite direction. "
            )
            explanation_parts.append(
                f"I therefore give a slightly 'No'-leaning answer that having children do not decrease "
                f"engagement in extramarital affairs, with strength {response}/100 on the 0–100 scale."
            )
    else:
        explanation_parts.append(
            "The estimated effect of having children on extramarital affairs is essentially zero, "
            "providing no clear evidence that children either increase or decrease engagement in such "
            "affairs. "
        )
        explanation_parts.append(
            f"I therefore give a neutral answer on whether children decrease engagement in extramarital "
            f"affairs, with strength {response}/100 on the 0–100 scale."
        )

    explanation = "".join(explanation_parts)

    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)


if __name__ == "__main__":
    main()

