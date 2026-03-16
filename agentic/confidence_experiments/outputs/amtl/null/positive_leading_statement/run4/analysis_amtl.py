import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Create indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Basic sanity filters: keep rows with positive socket counts
    df = df[df["sockets"] > 0].copy()

    # Poisson regression for AMTL counts with log(sockets) offset
    # This models AMTL rates per tooth socket while allowing counts
    # to exceed the socket count in individual records.
    formula = "num_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["sockets"]),
    )
    result = model.fit()

    # Extract human coefficient and infer effect size
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    p_human = float(result.pvalues["is_human"])
    or_human = float(np.exp(coef_human))
    ci_low, ci_high = np.exp(result.conf_int().loc["is_human"].to_numpy())

    # Average marginal effect on AMTL rate: compare predicted rates
    base_design = df.copy()
    human_design = base_design.copy()
    nonhuman_design = base_design.copy()
    human_design["is_human"] = 1
    nonhuman_design["is_human"] = 0

    pred_human_counts = result.predict(human_design, offset=np.log(human_design["sockets"]))
    pred_nonhuman_counts = result.predict(
        nonhuman_design, offset=np.log(nonhuman_design["sockets"])
    )

    # Convert to per-socket AMTL rates
    pred_human = pred_human_counts / human_design["sockets"].to_numpy()
    pred_nonhuman = pred_nonhuman_counts / nonhuman_design["sockets"].to_numpy()

    # Weight by number of sockets to approximate per-tooth AMTL frequency
    weights = df["sockets"].to_numpy()
    avg_human = float(np.average(pred_human, weights=weights))
    avg_nonhuman = float(np.average(pred_nonhuman, weights=weights))
    abs_diff = avg_human - avg_nonhuman
    rel_ratio = avg_human / avg_nonhuman if avg_nonhuman > 0 else np.nan

    # Map evidence strength to Likert-style 0–100 response
    # Strong evidence for higher human AMTL: positive coefficient, OR>1, and small p-value.
    if (coef_human > 0) and (p_human < 1e-4) and (or_human > 1.5):
        # Strong "Yes"
        response = 90
    elif (coef_human > 0) and (p_human < 0.01) and (or_human > 1.2):
        # Moderate "Yes"
        response = 75
    elif (coef_human > 0) and (p_human < 0.05):
        # Weak but significant "Yes"
        response = 60
    elif p_human >= 0.05:
        # No clear evidence of a difference
        if coef_human > 0:
            # Directionally higher but not significant
            response = 40
        else:
            # Directionally lower or null
            response = 20
    else:
        # Fallback (should rarely trigger)
        response = 50

    # Build explanation string with key quantitative findings
    explanation_lines = [
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, "
        "Papio), after accounting for age, sex, and tooth class?",
        "",
        "Analysis:",
        "- I modeled the AMTL counts (num_amtl) per specimen-tooth-class record "
        "using a Poisson GLM with a log link and log(sockets) as an offset, so "
        "that the model estimates AMTL rates per tooth socket.",
        "- Predictors included a binary indicator for modern humans versus "
        "non-human primates (is_human), age at death (age), estimated probability "
        "of being male (prob_male), and tooth class as a categorical factor "
        "(anterior, posterior, premolar).",
        "- The log(sockets) offset ensures that each tooth socket contributes "
        "proportionally to the estimation as an exposure term.",
        "",
        "Key results for the human effect (is_human):",
        f"- Log-odds coefficient: {coef_human:.3f} (SE {se_human:.3f}), "
        f"p-value {p_human:.3g}.",
        f"- Odds ratio for AMTL in humans vs non-human primates: {or_human:.2f} "
        f"(95% CI {ci_low:.2f}–{ci_high:.2f}).",
        f"- Model-based average predicted AMTL frequency per tooth socket is "
        f"{avg_human:.3f} for humans and {avg_nonhuman:.3f} for non-human "
        f"primates, a difference of {abs_diff:.3f} (ratio {rel_ratio:.2f}).",
        "",
        "Interpretation:",
    ]

    if response >= 70:
        interpretation = (
            "The positive and statistically significant human coefficient, together "
            "with an odds ratio clearly greater than 1 and higher predicted AMTL "
            "frequencies for humans, indicates that modern humans do have higher "
            "rates of antemortem tooth loss than the non-human primates in this "
            "dataset, even after adjusting for age, sex, and tooth class."
        )
    elif response >= 50:
        interpretation = (
            "The human effect is positive and at least marginally significant, "
            "suggesting somewhat higher AMTL frequencies in humans than in "
            "non-human primates after adjustment, but the magnitude or certainty "
            "is moderate rather than overwhelming."
        )
    elif response >= 30:
        interpretation = (
            "Although the human coefficient is in the positive direction, the "
            "confidence interval includes no effect at conventional levels, so the "
            "data do not provide strong evidence that humans differ from non-human "
            "primates in AMTL frequencies once age, sex, and tooth class are "
            "accounted for."
        )
    else:
        interpretation = (
            "The estimated human effect is near zero or negative and not "
            "statistically significant, indicating no evidence that humans have "
            "higher AMTL frequencies than non-human primates after accounting for "
            "age, sex, and tooth class, and possibly even lower frequencies."
        )

    explanation_lines.append(interpretation)

    explanation = "\n".join(explanation_lines)

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
