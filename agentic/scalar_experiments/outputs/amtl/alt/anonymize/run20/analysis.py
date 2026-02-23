import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in working directory.")

    df = pd.read_csv(data_path)

    # Basic cleaning: ensure valid denominators
    df = df[df["feature4"] > 0].copy()

    # Derived variables
    df["prop_missing"] = df["feature3"] / df["feature4"]
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class
    formula = "prop_missing ~ is_human + feature5 + feature7 + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    ).fit()

    coef_human = model.params["is_human"]
    pvalue_human = model.pvalues["is_human"]
    odds_ratio = float(np.exp(coef_human))

    # Average predicted AMTL probability for humans vs non-humans,
    # holding age, sex, and tooth class at their observed values.
    design = df.copy()
    design_human = design.copy()
    design_human["is_human"] = 1
    design_nonhuman = design.copy()
    design_nonhuman["is_human"] = 0

    pred_human = float(model.predict(design_human).mean())
    pred_nonhuman = float(model.predict(design_nonhuman).mean())
    delta = pred_human - pred_nonhuman

    # Simple mapping from statistical evidence to Likert scale (0-100).
    # Start from neutral 50 and adjust based on direction, effect size, and p-value.
    if pvalue_human >= 0.05:
        # No strong evidence that humans differ from non-human primates.
        if coef_human > 0:
            # Point estimate suggests higher AMTL in humans but not significant.
            response = 40
        else:
            # Point estimate suggests lower or similar AMTL in humans.
            response = 30
    else:
        # Statistically significant difference.
        if coef_human > 0:
            # Humans have higher AMTL than non-human primates.
            # Scale strength based on odds ratio magnitude.
            if odds_ratio >= 2.0:
                response = 90
            elif odds_ratio >= 1.5:
                response = 80
            else:
                response = 70
        else:
            # Humans have lower AMTL.
            if odds_ratio <= 0.5:
                response = 10
            elif odds_ratio <= 0.67:
                response = 20
            else:
                response = 30

    # Explanation string summarizing the evidence.
    direction = (
        "higher"
        if coef_human > 0
        else ("lower" if coef_human < 0 else "similar")
    )

    explanation = (
        "We fit a binomial regression model where the proportion of missing teeth "
        "(feature3 / feature4) was modeled as a function of a human indicator "
        "(Homo sapiens vs. non-human primates), estimated age at death (feature5), "
        "sex estimate (feature7), and tooth class (feature1), with the number of "
        "observable sockets (feature4) used as binomial trial weights. "
        f"The coefficient for the human indicator was {coef_human:.3f} on the log-odds scale "
        f"(odds ratio ≈ {odds_ratio:.2f}, p-value = {pvalue_human:.3g}), indicating "
        f"{direction} AMTL frequencies in modern humans compared to non-human primates "
        "after accounting for age, sex, and tooth class. "
        f"The model-based average predicted AMTL probability was {pred_human:.3f} for humans "
        f"and {pred_nonhuman:.3f} for non-human primates (difference = {delta:.3f}). "
        f"Based on this evidence, we map the strength of support for humans having higher "
        f"AMTL frequencies onto a 0–100 Likert scale as {response}."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON object to conclusion.txt
    output_path = Path("conclusion.txt")
    output_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

