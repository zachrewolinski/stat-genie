import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    cwd = Path(__file__).resolve().parent
    info = json.loads((cwd / "info.json").read_text())
    df = pd.read_csv(cwd / "amtl.csv")

    # Create outcome as proportion missing and a binomial response (successes, failures)
    df["missing"] = df["num_amtl"].astype(int)
    df["present"] = df["sockets"].astype(int) - df["missing"]

    # Filter to rows with at least one observable socket
    df = df[(df["sockets"] > 0) & (df["sockets"].notna())]

    # Define human indicator and keep only relevant genera
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Scale age for stability
    df["age_c"] = df["age"] - df["age"].mean()

    # Use prob_male as a continuous sex covariate
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Tooth class as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial GLM with logit link: missing ~ is_human + age + sex + tooth_class
    formula = "missing + present ~ is_human + age_c + prob_male_c + C(tooth_class)"

    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Extract coefficient and p-value for human indicator
    coef_human = model.params.get("is_human", np.nan)
    pvalue_human = model.pvalues.get("is_human", np.nan)

    # Compute estimated AMTL probabilities for a reference profile
    # Use mean age and sex, and most common tooth class
    ref_df = pd.DataFrame(
        {
            "is_human": [0, 1],  # non-human, human
            "age_c": [0, 0],
            "prob_male_c": [0, 0],
            "tooth_class": [df["tooth_class"].mode()[0]] * 2,
        }
    )

    # Use the fitted model to predict AMTL probabilities for humans vs non-humans
    prob = model.predict(ref_df)
    prob_nonhuman, prob_human = prob.tolist()

    # Determine conclusion: question asks if humans have higher AMTL
    # Null/leading belief is "No" (humans do not have higher AMTL).
    # We answer "Yes" if evidence suggests humans have higher rates (coef>0 and p<0.05 and higher predicted prob).
    if (
        np.isfinite(coef_human)
        and coef_human > 0
        and pvalue_human < 0.05
        and prob_human > prob_nonhuman
    ):
        response = "Yes"
    else:
        response = "No"

    explanation = {
        "research_question": info["research_questions"][0],
        "model_summary": str(model.summary()),
        "coef_is_human": float(coef_human) if np.isfinite(coef_human) else None,
        "pvalue_is_human": float(pvalue_human) if np.isfinite(pvalue_human) else None,
        "predicted_amtl_probability_nonhuman": float(prob_nonhuman),
        "predicted_amtl_probability_human": float(prob_human),
        "interpretation": (
            "Evidence for higher AMTL in humans (positive, significant human coefficient)"
            if response == "Yes"
            else "No strong evidence that humans have higher AMTL than non-human primates after adjusting for covariates"
        ),
    }

    result = {"response": response, "explanation": json.dumps(explanation)}

    (cwd / "conclusion.txt").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
