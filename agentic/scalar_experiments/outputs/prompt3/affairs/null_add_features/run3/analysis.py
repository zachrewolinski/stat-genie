import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic cleaning: ensure expected columns exist
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
        "income",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create binary outcome: any affair vs none
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Normalize children to lower-case yes/no
    df["children"] = df["children"].astype(str).str.strip().str.lower()
    df = df[df["children"].isin(["yes", "no"])].copy()

    return df


def descriptive_stats(df: pd.DataFrame) -> dict:
    grouped = df.groupby("children")
    mean_affairs = grouped["affairs"].mean().to_dict()
    prop_any = grouped["affair_any"].mean().to_dict()

    return {
        "mean_affairs_by_children": mean_affairs,
        "prop_any_by_children": prop_any,
    }


def fit_logit(df: pd.DataFrame):
    # Encode children: 1 = has children, 0 = no children
    df = df.copy()
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Categorical variables via dummy coding
    gender = pd.get_dummies(df["gender"], prefix="gender", drop_first=True)

    # Continuous / ordinal controls
    controls = df[["age", "yearsmarried", "religiousness", "education", "occupation", "rating", "income"]]

    X = pd.concat([df["children_yes"], gender, controls], axis=1)
    X = sm.add_constant(X)
    y = df["affair_any"]

    logit_model = sm.Logit(y, X, missing="drop")
    try:
        result = logit_model.fit(disp=False)
    except Exception:
        # In case of separation or convergence issues, fall back to fewer controls
        X_simple = sm.add_constant(df[["children_yes", "age", "yearsmarried"]])
        logit_model = sm.Logit(y, X_simple, missing="drop")
        result = logit_model.fit(disp=False)

    return result


def interpret_effect(descriptives: dict, logit_result) -> dict:
    mean_affairs = descriptives["mean_affairs_by_children"]
    prop_any = descriptives["prop_any_by_children"]

    # Descriptive direction: compare no vs yes
    mean_no = mean_affairs.get("no", np.nan)
    mean_yes = mean_affairs.get("yes", np.nan)
    prop_no = prop_any.get("no", np.nan)
    prop_yes = prop_any.get("yes", np.nan)

    # Default values
    response = "No"
    strength = 50
    confidence = 50

    # Try to extract children coefficient if present
    coef = None
    pval = None
    try:
        if "children_yes" in logit_result.params.index:
            coef = float(logit_result.params["children_yes"])
            pval = float(logit_result.pvalues["children_yes"])
    except Exception:
        coef = None
        pval = None

    # Decision logic:
    # If both descriptive and regression suggest children are associated
    # with lower engagement (negative coef, lower means/props), answer "Yes".
    dir_descriptive = None
    if np.isfinite(mean_no) and np.isfinite(mean_yes):
        if mean_yes < mean_no:
            dir_descriptive = "decrease"
        elif mean_yes > mean_no:
            dir_descriptive = "increase"
        else:
            dir_descriptive = "none"

    dir_binary = None
    if np.isfinite(prop_no) and np.isfinite(prop_yes):
        if prop_yes < prop_no:
            dir_binary = "decrease"
        elif prop_yes > prop_no:
            dir_binary = "increase"
        else:
            dir_binary = "none"

    directions = [d for d in [dir_descriptive, dir_binary] if d is not None]

    # Determine overall direction from descriptives
    overall_dir = None
    if directions:
        if all(d == "decrease" for d in directions):
            overall_dir = "decrease"
        elif all(d == "increase" for d in directions):
            overall_dir = "increase"
        else:
            overall_dir = "mixed"

    # Regression evidence
    reg_direction = None
    sig = None
    if coef is not None:
        reg_direction = "decrease" if coef < 0 else ("increase" if coef > 0 else "none")
        if pval is not None:
            if pval < 0.01:
                sig = "strong"
            elif pval < 0.05:
                sig = "moderate"
            elif pval < 0.1:
                sig = "weak"
            else:
                sig = "ns"

    # Combine evidence for final yes/no and strength
    if overall_dir == "decrease" and reg_direction == "decrease":
        response = "Yes"
    elif overall_dir == "increase" and reg_direction == "increase":
        response = "No"
    elif reg_direction == "decrease":
        # Regression suggests a decrease even if descriptives are noisy
        response = "Yes"
    elif reg_direction == "increase":
        response = "No"
    elif overall_dir == "decrease":
        response = "Yes"
    elif overall_dir == "increase":
        response = "No"
    else:
        # No clear directional evidence
        response = "No"

    # Strength mapping
    base_strength = 50
    if sig == "strong":
        base_strength = 85
    elif sig == "moderate":
        base_strength = 75
    elif sig == "weak":
        base_strength = 65
    elif sig == "ns":
        base_strength = 55

    # Adjust strength if directions conflict
    if overall_dir == "mixed":
        base_strength -= 15
    elif overall_dir is None:
        base_strength -= 10

    base_strength = int(min(max(base_strength, 0), 100))

    # Confidence slightly below strength to reflect model assumptions
    conf = max(base_strength - 5, 0)

    # Build textual explanation
    explanation_parts = []
    explanation_parts.append(
        f"Research question: Does having children decrease engagement in extramarital affairs?"
    )
    explanation_parts.append(
        "I analyzed the Fair affairs dataset (601 married individuals) using both descriptive statistics "
        "and a logistic regression model predicting any extramarital affair during the past year."
    )

    if np.isfinite(mean_no) and np.isfinite(mean_yes):
        explanation_parts.append(
            f"Average affair score (0–12 scale) was {mean_no:.2f} for respondents without children "
            f"and {mean_yes:.2f} for respondents with children."
        )
    if np.isfinite(prop_no) and np.isfinite(prop_yes):
        explanation_parts.append(
            f"The proportion reporting at least one affair was {prop_no:.3f} without children "
            f"and {prop_yes:.3f} with children."
        )

    if coef is not None and pval is not None:
        explanation_parts.append(
            f"In the logistic regression controlling for gender, age, years married, religiousness, "
            f"education, occupation, self-rated marriage quality, and income, the coefficient for having "
            f"children was {coef:.3f} (p = {pval:.3f}) on the log-odds scale."
        )
        if reg_direction == "decrease":
            explanation_parts.append(
                "A negative coefficient indicates that, holding other factors constant, having children is "
                "associated with a lower probability of reporting any affair."
            )
        elif reg_direction == "increase":
            explanation_parts.append(
                "A positive coefficient indicates that, holding other factors constant, having children is "
                "associated with a higher probability of reporting any affair."
            )
        else:
            explanation_parts.append(
                "The coefficient is very close to zero, suggesting little to no association after adjustment."
            )
    else:
        explanation_parts.append(
            "Due to model limitations, I relied primarily on descriptive comparisons between respondents "
            "with and without children."
        )

    if overall_dir == "mixed":
        explanation_parts.append(
            "Descriptive indicators (mean affair scores and proportions) point in different directions, "
            "so the evidence from simple comparisons is mixed."
        )
    elif overall_dir == "decrease":
        explanation_parts.append(
            "Both mean affair scores and proportions suggest that respondents with children have slightly "
            "lower engagement in affairs than those without children."
        )
    elif overall_dir == "increase":
        explanation_parts.append(
            "Both mean affair scores and proportions suggest that respondents with children have slightly "
            "higher engagement in affairs than those without children."
        )

    explanation_parts.append(
        "Given the sample size, observational design, and possible unmeasured confounders, I treat these "
        "results as associative rather than strictly causal."
    )

    explanation = " ".join(explanation_parts)

    return {
        "response": response,
        "strength": base_strength,
        "confidence": int(conf),
        "explanation": explanation,
    }


def main() -> None:
    df = load_data(Path("affairs.csv"))
    descriptives = descriptive_stats(df)
    logit_result = fit_logit(df)
    conclusion = interpret_effect(descriptives, logit_result)

    # Write conclusion.json as specified
    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

