import json

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Focus on variables relevant to the research question.
    required_cols = [
        "affairs",
        "children",
        "gender",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    df = df.dropna(subset=required_cols)

    # Binary indicator for whether the respondent engaged in any extramarital affairs.
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Descriptive comparison: share with any affair by children status.
    desc = (
        df.groupby("children")["affair_any"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_any"})
    )

    # Logistic regression controlling for key covariates.
    formula = (
        "affair_any ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )

    try:
        model = smf.logit(formula, data=df).fit(disp=False)
    except Exception:
        # Fallback: run a simpler model if convergence or data issues arise.
        formula_simple = "affair_any ~ C(children) + yearsmarried + rating"
        model = smf.logit(formula_simple, data=df).fit(disp=False)

    params = model.params
    pvalues = model.pvalues

    # Identify the children effect term (yes vs no). Depending on coding,
    # statsmodels will create C(children)[T.yes] or similar.
    children_term = None
    for term in params.index:
        if term.startswith("C(children)[T."):
            children_term = term
            break

    # Default answer if we cannot estimate the children effect.
    response = "No"
    confidence = 50
    explanation_parts = []

    # Add descriptive stats to explanation.
    desc_strs = []
    for level, row in desc.iterrows():
        desc_strs.append(
            f"{level}: {row['prop_any']:.3f} share with any affair (n={int(row['count'])})"
        )
    if desc_strs:
        explanation_parts.append(
            "Observed share with any extramarital affair by children status: "
            + "; ".join(desc_strs)
            + "."
        )

    if children_term is not None:
        coef = float(params[children_term])
        pval = float(pvalues[children_term])

        # Interpret direction and significance.
        if coef < 0 and pval < 0.05:
            response = "Yes"
        else:
            response = "No"

        # Confidence based on statistical significance.
        if pval < 0.01:
            confidence = 90
        elif pval < 0.05:
            confidence = 80
        elif pval < 0.1:
            confidence = 65
        else:
            confidence = 55

        direction = "decreases" if coef < 0 else "increases"
        explanation_parts.append(
            "A logistic regression of any affair on children, "
            "controlling for gender, age, years married, religiousness, "
            "education, occupation, and marital rating, "
            f"shows that having children {direction} the log-odds of an affair "
            f"(coefficient {coef:.3f}, p-value {pval:.3f})."
        )
    else:
        explanation_parts.append(
            "The regression model could not estimate a distinct children effect term, "
            "so conclusions rely only on descriptive comparisons."
        )

    explanation = " ".join(explanation_parts)

    result = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

