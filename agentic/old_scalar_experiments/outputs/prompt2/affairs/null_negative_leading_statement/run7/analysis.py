import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Key derived variables
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["has_children"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status
    grouped = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any=("has_affair", "mean"),
            n_obs=("affairs", "size"),
        )
        .reset_index()
    )

    # Map to convenient accessors
    desc = {
        row["children"]: {
            "mean_affairs": float(row["mean_affairs"]),
            "median_affairs": float(row["median_affairs"]),
            "prop_any": float(row["prop_any"]),
            "n_obs": int(row["n_obs"]),
        }
        for _, row in grouped.iterrows()
    }

    # Differences (parents minus non-parents)
    mean_diff = desc["yes"]["mean_affairs"] - desc["no"]["mean_affairs"]
    prop_diff = desc["yes"]["prop_any"] - desc["no"]["prop_any"]

    # Logistic regression: probability of any affair
    df_reg = df.copy()
    X = df_reg[
        [
            "has_children",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = df_reg["has_affair"]

    logit_model = sm.Logit(y, X, missing="drop")
    logit_res = logit_model.fit(disp=False)
    logit_coef = float(logit_res.params["has_children"])
    logit_pval = float(logit_res.pvalues["has_children"])
    logit_odds_ratio = float(np.exp(logit_coef))

    # Linear model for log(affairs + 1) as a rough intensity measure
    df_reg["log_affairs_plus1"] = np.log(df_reg["affairs"] + 1.0)
    y_intensity = df_reg["log_affairs_plus1"]
    ols_model = sm.OLS(y_intensity, X)
    ols_res = ols_model.fit()
    ols_coef = float(ols_res.params["has_children"])
    ols_pval = float(ols_res.pvalues["has_children"])

    # Decide on response based on direction and strength of evidence
    negative_direction_count = sum(
        [
            mean_diff < 0,
            prop_diff < 0,
            logit_coef < 0,
            ols_coef < 0,
        ]
    )

    strong_negative_signals = sum(
        [
            logit_coef < 0 and logit_pval < 0.05,
            ols_coef < 0 and ols_pval < 0.05,
        ]
    )

    if strong_negative_signals >= 1 and negative_direction_count >= 3:
        response = "Yes"
        confidence = 85
    elif strong_negative_signals >= 1 and negative_direction_count >= 2:
        response = "Yes"
        confidence = 75
    elif negative_direction_count >= 3:
        response = "Yes"
        confidence = 65
    else:
        response = "No"
        # Confidence depends on how consistently non-negative the evidence is
        non_negative_signals = sum(
            [
                mean_diff >= 0,
                prop_diff >= 0,
                logit_coef >= 0,
                ols_coef >= 0,
            ]
        )
        if (
            (logit_coef > 0 and logit_pval < 0.05)
            or (ols_coef > 0 and ols_pval < 0.05)
        ) and non_negative_signals >= 3:
            confidence = 80
        else:
            confidence = 60

    explanation = (
        "I examined whether having children is associated with lower engagement in extramarital affairs. "
        f"Descriptively, among those without children (n={desc['no']['n_obs']}), the mean affairs score was "
        f"{desc['no']['mean_affairs']:.2f} with {desc['no']['prop_any']*100:.1f}% reporting at least one affair, "
        f"whereas among those with children (n={desc['yes']['n_obs']}), the mean affairs score was "
        f"{desc['yes']['mean_affairs']:.2f} with {desc['yes']['prop_any']*100:.1f}% reporting any affair. "
        f"The differences (children minus no children) were {mean_diff:.2f} in mean affairs and "
        f"{prop_diff*100:.1f} percentage points in the probability of any affair. "
        "To adjust for potential confounders (age, years married, religiousness, education, occupation, and self-rated "
        "marital happiness), I fit a logistic regression for the probability of any affair with an indicator for having "
        f"children. The coefficient for having children was {logit_coef:.3f}, corresponding to an odds ratio of "
        f"{logit_odds_ratio:.3f} (p-value={logit_pval:.3f}). I also fit a linear regression for log(affairs+1) as an "
        f"intensity measure; the coefficient on having children was {ols_coef:.3f} (p-value={ols_pval:.3f}). "
        "Taken together, these descriptive and regression results were used to determine whether the evidence supports "
        "the claim that having children decreases engagement in extramarital affairs."
    )

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()

