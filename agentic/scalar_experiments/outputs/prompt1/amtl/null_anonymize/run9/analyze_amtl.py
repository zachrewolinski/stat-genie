import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Ensure valid denominator
    df = df[df["sockets"] > 0].copy()

    # Response as proportion with binomial denominator
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Fit binomial logistic regression with frequency weights = number of sockets
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_estimate + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = float(model.params["is_human"])
    pval = float(model.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = model.conf_int().loc["is_human"]
    ci_low = float(ci_low)
    ci_high = float(ci_high)
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    # Predicted probabilities at typical covariate values
    mean_age = float(df["age"].mean())
    mean_sex = float(df["sex_estimate"].mean())
    ref_class = str(df["tooth_class"].mode().iloc[0])

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "sex_estimate": [mean_sex, mean_sex],
            "tooth_class": [ref_class, ref_class],
        }
    )
    preds = model.predict(pred_df)
    prob_nonhuman = float(preds.iloc[0])
    prob_human = float(preds.iloc[1])

    # Decision rule: higher and statistically significant AMTL frequency in humans
    if coef > 0 and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Using a binomial logistic regression of the number of missing teeth "
        "out of observable sockets, with a binary indicator for modern humans "
        "versus non-human primates and covariates for estimated age at death, "
        "sex estimate, and tooth class, the coefficient for humans "
        f"was {coef:.3f} (odds ratio {odds_ratio:.2f}, 95% CI [{or_low:.2f}, {or_high:.2f}], "
        f"p = {pval:.3g}). At typical covariate values (age ≈ {mean_age:.1f} years, "
        f"tooth class = {ref_class}), the predicted proportion of teeth missing "
        f"was {prob_nonhuman:.3f} for non-human primates and {prob_human:.3f} for modern humans. "
        "Based on this model, modern humans "
        + (
            "have a significantly higher frequency of antemortem tooth loss than non-human primates."
            if response == "Yes"
            else "do not show a statistically higher frequency of antemortem tooth loss than non-human primates."
        )
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write required JSON-only output
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

