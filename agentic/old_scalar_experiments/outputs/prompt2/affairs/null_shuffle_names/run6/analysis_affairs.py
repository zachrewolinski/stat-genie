import json

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("affairs.csv")

    # According to info.json, the columns are shuffled:
    # - "age" actually encodes affair frequency (0 = none, >0 = some affairs).
    # - "religiousness" is a yes/no factor answering: "Are there children in the marriage?"
    df["affair_freq"] = df["age"]
    df["had_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop any rows with missing mappings, just in case
    df = df.dropna(subset=["had_affair", "has_children"])

    # Descriptive comparison of affair prevalence by children status
    group_stats = (
        df.groupby("has_children")["had_affair"]
        .agg(["mean", "count"])
        .rename(index={0: "no_children", 1: "has_children"})
    )

    prop_no_children = group_stats.loc["no_children", "mean"]
    prop_with_children = group_stats.loc["has_children", "mean"]

    # Logistic regression: had_affair ~ has_children
    X = sm.add_constant(df["has_children"])
    y = df["had_affair"]

    try:
        model = sm.Logit(y, X).fit(disp=False)
        coef_children = model.params["has_children"]
        p_value = float(model.pvalues["has_children"])
        odds_ratio = float(np.exp(coef_children))
    except Exception:
        # Fallback: if model fails, base conclusion on proportion difference only
        coef_children = float("nan")
        p_value = float("nan")
        odds_ratio = float("nan")

    # Determine answer to: "Does having children decrease engagement in extramarital affairs?"
    # Direction based on proportions and (when available) regression coefficient.
    children_lower_prop = prop_with_children < prop_no_children

    if not np.isnan(coef_children):
        # Use regression sign to determine direction when available
        children_lower_logodds = coef_children < 0
    else:
        children_lower_logodds = children_lower_prop

    if children_lower_prop and children_lower_logodds:
        response = "Yes"
    else:
        response = "No"

    # Confidence calibration based on p-value and magnitude of difference
    diff = abs(prop_no_children - prop_with_children)

    if np.isnan(p_value):
        # No model result; rely on effect size only
        if diff > 0.15:
            confidence = 70
        elif diff > 0.05:
            confidence = 55
        else:
            confidence = 45
    else:
        if p_value < 0.01 and diff > 0.10:
            confidence = 90
        elif p_value < 0.05 and diff > 0.05:
            confidence = 80
        elif p_value < 0.10 and diff > 0.03:
            confidence = 65
        else:
            confidence = 50

    # Build explanation string summarizing evidence
    explanation = (
        "Using the 601 married respondents, I treated the 'age' column as the coded "
        "frequency of extramarital intercourse (0 = none, higher values = more frequent) "
        "and the 'religiousness' column as a yes/no indicator of whether there are "
        "children in the marriage, as described in info.json. I defined a binary outcome "
        "for having any affairs in the past year and compared its prevalence between "
        "couples with and without children. The proportion reporting any affairs was "
        f"{prop_with_children:.3f} among those with children versus "
        f"{prop_no_children:.3f} among those without children. A logistic regression of "
        "having any affair on the children indicator "
    )

    if not np.isnan(p_value):
        explanation += (
            f"yielded an odds ratio of approximately {odds_ratio:.2f} for having children "
            f"(p-value ≈ {p_value:.3f}). "
        )
    else:
        explanation += "could not be reliably estimated, so I relied on the difference in proportions alone. "

    if response == "Yes":
        explanation += (
            "Both the descriptive comparison and the regression indicate that respondents "
            "with children are modestly less likely to report extramarital affairs than "
            "those without children, supporting the conclusion that having children is "
            "associated with lower engagement in extramarital affairs in this sample."
        )
    else:
        explanation += (
            "The descriptive comparison and the regression do not consistently show that "
            "respondents with children are less likely to report extramarital affairs; "
            "if anything, the data suggest similar or higher engagement among those with "
            "children. Therefore, this dataset does not provide strong evidence that "
            "having children decreases engagement in extramarital affairs."
        )

    result = {
        "response": response,
        "confidence": int(confidence),
        "explanation": explanation,
    }

    # Write the required JSON object with no extra text
    with open("conclusion.txt", "w") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    main()

