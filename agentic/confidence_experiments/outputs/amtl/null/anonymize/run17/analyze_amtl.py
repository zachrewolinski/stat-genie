import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def compute_likert(coef: float, pval: float, diff: float) -> tuple[int, str]:
    """
    Map the sign, size, and significance of the human effect
    onto a 0–100 Likert scale where higher values indicate
    stronger evidence that humans have higher AMTL than
    non-human primates.
    """
    if not (np.isfinite(coef) and np.isfinite(pval) and np.isfinite(diff)):
        explanation = (
            "The regression model did not yield finite estimates for the human effect; "
            "therefore the strength of evidence is assessed as indeterminate."
        )
        return 50, explanation

    # Default neutral setting
    response = 50
    explanation = ""

    # Strongest evidence: significant and effect direction clear
    if pval < 0.05:
        effect = max(0.0, min(abs(diff), 0.5))

        if coef > 0:
            # Humans have higher AMTL with statistical significance
            base = 70
            response = base + int(round(30 * (effect / 0.5)))
            explanation = (
                "The human indicator has a positive and statistically significant "
                "coefficient (p < 0.05), indicating higher AMTL in humans than in "
                "non-human primates after adjustment. The Likert score reflects both "
                "the statistical significance and the magnitude of the predicted "
                "difference in AMTL frequency."
            )
        else:
            # Significant evidence in the opposite direction
            base = 10
            response = base + int(round(20 * (effect / 0.5)))
            explanation = (
                "The human indicator has a negative and statistically significant "
                "coefficient (p < 0.05), indicating lower AMTL in humans than in "
                "non-human primates after adjustment. The Likert score near the "
                "low end reflects strong evidence against humans having higher AMTL."
            )
    else:
        # No conventional statistical significance
        if coef > 0:
            response = 40
            explanation = (
                "The human indicator coefficient is positive but not statistically "
                "significant (p ≥ 0.05). This suggests a possible tendency toward "
                "higher AMTL in humans, but the evidence is weak and compatible with "
                "no real difference. The Likert score is therefore slightly below "
                "the neutral midpoint, reflecting limited support for a 'Yes' answer."
            )
        elif coef < 0:
            response = 25
            explanation = (
                "The human indicator coefficient is negative and not statistically "
                "significant (p ≥ 0.05). This does not provide strong evidence that "
                "humans have higher AMTL than non-human primates; if anything, the "
                "point estimate leans in the opposite direction. The Likert score "
                "reflects a modest lean toward a 'No' answer."
            )
        else:
            response = 50
            explanation = (
                "The human indicator coefficient is effectively zero and not "
                "statistically significant, providing no evidence of a difference in "
                "AMTL between humans and non-human primates after adjustment. The "
                "Likert score is set to the neutral midpoint."
            )

    response = int(max(0, min(response, 100)))
    return response, explanation


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

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

    # Basic cleaning and validity checks
    df = df[(df["sockets"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets"])]
    df = df.dropna(
        subset=["missing", "sockets", "age", "sex_estimate", "tooth_class", "genus"]
    )

    # Proportion of missing teeth within each record
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure both humans and non-human primates are present
    if df["is_human"].nunique() < 2:
        explanation = (
            "After filtering invalid observations, the dataset does not contain both "
            "human and non-human primate specimens, so a comparative analysis of "
            "AMTL frequencies is not possible. The response is therefore set to a "
            "neutral value reflecting this limitation."
        )
        result = {"response": 50, "explanation": explanation}
        Path("conclusion.txt").write_text(json.dumps(result))
        return

    # Fit a binomial regression model for the proportion of missing teeth
    formula = "prop_missing ~ is_human + age + sex_estimate + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    fit_res = model.fit()

    coef = float(fit_res.params.get("is_human", np.nan))
    pval = float(fit_res.pvalues.get("is_human", np.nan))
    odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else float("nan")

    # Average predicted AMTL probabilities for humans vs non-humans,
    # holding the distribution of covariates fixed at the observed values.
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0
    df_human = df.copy()
    df_human["is_human"] = 1

    mean_pred_nonhuman = float(fit_res.predict(df_nonhuman).mean())
    mean_pred_human = float(fit_res.predict(df_human).mean())
    diff = mean_pred_human - mean_pred_nonhuman

    response_value, likert_expl = compute_likert(coef, pval, diff)

    explanation_parts = [
        (
            "We analyzed the antemortem tooth loss (AMTL) dataset using a binomial "
            "regression model. For each record, the response was the proportion of "
            "missing teeth (number of missing teeth divided by the number of "
            "observable sockets) and the model included predictors for whether the "
            "specimen was a modern human (Homo sapiens) versus a non-human primate, "
            "estimated age at death, estimated sex, and tooth class (anterior, "
            "posterior, or premolar). The model was fit using binomial GLM with "
            "the number of observable sockets as weights to reflect the number of "
            "teeth at risk in each observation."
        ),
        (
            f"The fitted coefficient for the human indicator (is_human) was "
            f"{coef:.3f}, corresponding to an odds ratio of approximately "
            f"{odds_ratio:.3f}. The p-value for this coefficient was {pval:.3g}."
        ),
        (
            f"Based on model predictions holding the distribution of covariates "
            f"constant, the average predicted probability of a tooth being "
            f"antemortemly lost was about {mean_pred_nonhuman:.3f} for non-human "
            f"primates and {mean_pred_human:.3f} for modern humans, a difference of "
            f"{diff:.3f} in absolute probability."
        ),
        likert_expl,
        (
            f"Interpreting the research question—whether modern humans have higher "
            f"frequencies of AMTL than non-human primates after accounting for age, "
            f"sex, and tooth class—the Likert-scale response of {response_value} "
            f"places our answer on a 0–100 scale where 0 represents a strong 'No' "
            f"and 100 a strong 'Yes'. Values above 50 indicate increasing support "
            f"for humans having higher AMTL, while values below 50 indicate "
            f"evidence against that claim."
        ),
    ]

    explanation = " ".join(explanation_parts)

    result = {"response": int(response_value), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(result))


if __name__ == "__main__":
    main()

