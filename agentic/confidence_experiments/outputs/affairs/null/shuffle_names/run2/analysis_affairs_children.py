import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Map semantics based on info.json documentation:
    # - Column "age" actually encodes frequency of extramarital intercourse in past year.
    # - Column "religiousness" is a yes/no indicator for whether there are children.
    df["affairs_freq"] = df["age"]
    df["any_affair"] = (df["affairs_freq"] > 0).astype(int)

    # Presence of children encoded as yes/no.
    children_raw = (
        df["religiousness"].astype(str).str.strip().str.lower()
    )
    df["has_children"] = children_raw.map({"yes": 1, "no": 0})
    df = df[df["has_children"].notna()].copy()

    # Descriptive statistics by children status.
    grouped = df.groupby("has_children", observed=True)
    mean_freq = grouped["affairs_freq"].mean()
    prop_any = grouped["any_affair"].mean()

    # Prepare data for regression: adjust for available covariates.
    reg_cols = [
        "any_affair",
        "has_children",
        "gender",
        "occupation",   # respondent age (coded midpoints)
        "children",     # years married
        "rating",       # religiousness level
        "yearsmarried", # education level (coded years)
        "affairs",      # marital satisfaction rating
    ]
    reg_df = df[reg_cols].dropna().copy()

    # Logistic regression: probability of any affair vs children and covariates.
    formula = (
        "any_affair ~ has_children + C(gender) + "
        "occupation + children + rating + yearsmarried + affairs"
    )
    model = smf.logit(formula=formula, data=reg_df).fit(disp=False)

    coef = model.params["has_children"]
    pvalue = float(model.pvalues["has_children"])
    or_ = float(np.exp(coef))
    ci_low, ci_high = model.conf_int().loc["has_children"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Significance strength description.
    if pvalue < 0.001:
        sig_strength = "very strong"
    elif pvalue < 0.01:
        sig_strength = "strong"
    elif pvalue < 0.05:
        sig_strength = "moderate"
    elif pvalue < 0.1:
        sig_strength = "weak"
    else:
        sig_strength = "no statistically significant"

    # Direction of association: OR < 1 means fewer affairs with children.
    if or_ < 1:
        direction = "decrease"
    elif or_ > 1:
        direction = "increase"
    else:
        direction = "no_change"

    # Likert scale mapping: 0 = strong "No", 100 = strong "Yes".
    # Question: "Does having children decrease engagement in extramarital affairs?"
    response = 50

    if sig_strength == "no statistically significant":
        if direction == "decrease":
            response = 60  # weak, non-significant support
        elif direction == "increase":
            response = 40  # weak, non-significant contradiction
        else:
            response = 50
    else:
        effect_mag = abs(float(np.log(or_)))
        if sig_strength == "weak":
            base_delta = 10
        elif sig_strength == "moderate":
            base_delta = 20
        elif sig_strength == "strong":
            base_delta = 30
        else:
            base_delta = 40
        delta = base_delta * min(effect_mag, 1.0)
        if direction == "decrease":
            response = int(round(50 + delta))
        elif direction == "increase":
            response = int(round(50 - delta))
        else:
            response = 50

    response = int(max(0, min(100, response)))

    # Pull descriptive values.
    # Index 1 = has children, 0 = no children.
    mean_freq_no_children = float(mean_freq.get(0, np.nan))
    mean_freq_with_children = float(mean_freq.get(1, np.nan))
    prop_any_no_children = float(prop_any.get(0, np.nan))
    prop_any_with_children = float(prop_any.get(1, np.nan))

    n_with_children = int((df["has_children"] == 1).sum())
    n_no_children = int((df["has_children"] == 0).sum())

    # Build explanation text.
    explanation_parts = []
    explanation_parts.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_parts.append(
        "Data come from the Fair (1978) extramarital affairs survey of 601 first-married individuals."
    )
    explanation_parts.append(
        "Following the provided metadata, the 'age' column is treated as the frequency of extramarital "
        "sexual intercourse in the past year (0 = none; larger values = more frequent affairs), and "
        "the 'religiousness' column is interpreted as a yes/no indicator for whether there are children "
        "in the marriage."
    )
    explanation_parts.append(
        f"After excluding rows with missing values in these variables and covariates, the analytic sample "
        f"contains {len(reg_df)} individuals: {n_with_children} with children and {n_no_children} without children."
    )
    if not np.isnan(prop_any_with_children) and not np.isnan(prop_any_no_children):
        explanation_parts.append(
            "Descriptively, "
            f"{prop_any_with_children:.1%} of individuals with children and "
            f"{prop_any_no_children:.1%} of individuals without children report at least one extramarital affair "
            "in the past year."
        )
    if not np.isnan(mean_freq_with_children) and not np.isnan(mean_freq_no_children):
        explanation_parts.append(
            "The mean affair-frequency score is "
            f"{mean_freq_with_children:.2f} with children versus "
            f"{mean_freq_no_children:.2f} without children (higher values indicate more frequent affairs)."
        )
    explanation_parts.append(
        "To formally assess the relationship, a logistic regression model was fit for the probability of having "
        "any extramarital affair (non-zero frequency) as a function of having children, adjusting for gender, "
        "age (coded midpoints), years married, religiousness, education level, and self-rated marital satisfaction."
    )
    explanation_parts.append(
        "In this model, the odds ratio for the 'has children' indicator is "
        f"{or_:.2f} with a 95% confidence interval of {or_ci_low:.2f} to {or_ci_high:.2f} "
        f"and a p-value of {pvalue:.3g}."
    )

    if sig_strength == "no statistically significant":
        explanation_parts.append(
            "Because this effect is not statistically significant at conventional thresholds, the data do not "
            "provide reliable evidence that having children meaningfully changes engagement in extramarital affairs."
        )
    else:
        if direction == "decrease":
            explanation_parts.append(
                "The odds ratio is below 1 and statistically significant, indicating that, after adjusting for "
                "other factors, individuals with children are less likely to report extramarital affairs. "
                "This supports the hypothesis that having children is associated with reduced engagement in "
                "extramarital affairs."
            )
        elif direction == "increase":
            explanation_parts.append(
                "The odds ratio is above 1 and statistically significant, indicating that, after adjusting for "
                "other factors, individuals with children are more likely to report extramarital affairs. "
                "This contradicts the hypothesis that having children decreases engagement and instead suggests "
                "an increase."
            )
        else:
            explanation_parts.append(
                "Although the model yields some association, its direction is effectively null, so there is no "
                "clear evidence that having children changes engagement in extramarital affairs."
            )

    if response >= 60:
        qualitative = "a modestly positive answer"
    elif response >= 75:
        qualitative = "a strong positive answer"
    elif response <= 40:
        qualitative = "a modestly negative answer"
    elif response <= 25:
        qualitative = "a strong negative answer"
    else:
        qualitative = "an essentially neutral answer"

    explanation_parts.append(
        "On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes' to the question "
        f"'Does having children decrease engagement in extramarital affairs?', the assigned score is {response}, "
        f"corresponding to {qualitative} given the observed effect size and statistical significance."
    )

    explanation = " ".join(explanation_parts)

    result = {"response": response, "explanation": explanation}

    with Path("conclusion.txt").open("w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

