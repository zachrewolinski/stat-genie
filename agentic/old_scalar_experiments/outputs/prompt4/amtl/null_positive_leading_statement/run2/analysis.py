import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    info_path = base_dir / "info.json"
    data_path = base_dir / "amtl.csv"
    conclusion_path = base_dir / "conclusion.txt"

    # Load research question (for inclusion in the explanation).
    with info_path.open() as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset.
    df = pd.read_csv(data_path)

    # Basic cleaning and feature construction.
    df = df.copy()
    df = df[df["sockets"] > 0].copy()
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["log_sockets"] = np.log(df["sockets"])

    # Descriptive summary by genus.
    genus_summary = (
        df.groupby("genus")
        .agg(total_amtl=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .assign(amtl_rate=lambda g: g["total_amtl"] / g["total_sockets"])
    )

    # Aggregate humans vs non-humans.
    if "Homo sapiens" in genus_summary.index:
        human_rate = float(genus_summary.loc["Homo sapiens", "amtl_rate"])
        nonhuman_tot_amtl = float(
            genus_summary.loc[genus_summary.index != "Homo sapiens", "total_amtl"].sum()
        )
        nonhuman_tot_sockets = float(
            genus_summary.loc[
                genus_summary.index != "Homo sapiens", "total_sockets"
            ].sum()
        )
        nonhuman_rate = nonhuman_tot_amtl / nonhuman_tot_sockets
    else:
        # Fallback in the unlikely event there are no human observations.
        human_rate = float("nan")
        nonhuman_rate = float("nan")

    # Fit Poisson regression with log link and offset for exposure (sockets).
    formula = "num_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_sockets"],
    )

    # Use robust (HC3) standard errors to reduce sensitivity to mild model
    # misspecification and overdispersion.
    result = model.fit(cov_type="HC3")

    # Extract human vs non-human effect.
    coef = float(result.params["is_human"])
    rr = float(np.exp(coef))
    ci_low_raw, ci_high_raw = result.conf_int().loc["is_human"].tolist()
    ci_low = float(np.exp(ci_low_raw))
    ci_high = float(np.exp(ci_high_raw))
    pvalue = float(result.pvalues["is_human"])

    # Map evidence into a 0–100 confidence score for a "Yes" answer.
    # 0 = strong "No", 100 = strong "Yes".
    if (rr > 1.0) and (pvalue < 0.001) and (ci_low > 1.0):
        response = 95
        qualitative_strength = "strong"
        qualitative_answer = "Yes"
    elif (rr > 1.0) and (pvalue < 0.05) and (ci_low > 1.0):
        response = 80
        qualitative_strength = "moderate"
        qualitative_answer = "Yes"
    elif (rr > 1.0) and (pvalue < 0.05):
        response = 65
        qualitative_strength = "weak"
        qualitative_answer = "Yes"
    elif (rr < 1.0) and (pvalue < 0.001) and (ci_high < 1.0):
        response = 5
        qualitative_strength = "strong"
        qualitative_answer = "No"
    elif (rr < 1.0) and (pvalue < 0.05) and (ci_high < 1.0):
        response = 20
        qualitative_strength = "moderate"
        qualitative_answer = "No"
    else:
        # Inconclusive evidence: center the response near 50 and
        # nudge based on point estimate direction.
        if rr > 1.0:
            response = 55
        elif rr < 1.0:
            response = 45
        else:
            response = 50
        qualitative_strength = "inconclusive"
        qualitative_answer = "Inconclusive"

    # Build human‑readable explanation.
    # Format key statistics for clarity.
    human_rate_str = (
        f"{human_rate:.3f}" if np.isfinite(human_rate) else "not available"
    )
    nonhuman_rate_str = (
        f"{nonhuman_rate:.3f}" if np.isfinite(nonhuman_rate) else "not available"
    )
    rr_str = f"{rr:.3f}"
    ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"
    if pvalue < 1e-4:
        p_str = f"{pvalue:.1e}"
    else:
        p_str = f"{pvalue:.4f}"

    explanation_parts = []
    explanation_parts.append(
        "Research question: "
        + research_question
    )
    explanation_parts.append(
        "Dataset and variables: The dataset contains counts of antemortem tooth "
        "loss (AMTL) by specimen and tooth class (anterior, posterior, premolar), "
        "along with the number of observable tooth sockets (exposure), estimated "
        "age at death, an estimate of sex (probability of being male), the genus "
        "of the specimen (Homo sapiens, Pan, Papio, Pongo), and population/region."
    )
    explanation_parts.append(
        "Descriptive comparison: Aggregating across all specimens, humans show an "
        f"overall AMTL rate of approximately {human_rate_str} missing teeth per "
        f"socket, whereas the combined non‑human primates (Pan, Papio, Pongo) have "
        f"an overall AMTL rate of approximately {nonhuman_rate_str}."
    )
    explanation_parts.append(
        "Modeling approach: To account for differing numbers of teeth at risk, I "
        "fit a Poisson regression with a log link and an offset for the log number "
        "of sockets. The outcome is the count of missing teeth, modeled as a rate "
        "per socket. Predictors include a binary indicator for humans versus "
        "non‑human primates (is_human), age, estimated sex (prob_male), and "
        "tooth_class as a categorical factor. Robust (HC3) standard errors are used "
        "to lessen the impact of mild overdispersion or model misspecification."
    )
    explanation_parts.append(
        "Key result: The coefficient for the human indicator corresponds to a rate "
        f"ratio of {rr_str} for humans versus non‑human primates "
        f"(95% CI {ci_str}, p = {p_str})."
    )

    if qualitative_answer == "Yes":
        interpretation = (
            "Because the estimated rate ratio is greater than 1 and the "
            "statistical test indicates "
        )
        if qualitative_strength == "strong":
            interpretation += (
                "strong evidence (very small p‑value with the entire confidence "
                "interval above 1) "
            )
        elif qualitative_strength == "moderate":
            interpretation += (
                "moderate evidence (p < 0.05 with the confidence interval above 1) "
            )
        else:
            interpretation += (
                "some evidence (p < 0.05, though the confidence interval only "
                "slightly exceeds 1) "
            )
        interpretation += (
            "that humans experience higher AMTL rates per socket than non‑human "
            "primates after adjusting for age, estimated sex, and tooth class."
        )
    elif qualitative_answer == "No":
        interpretation = (
            "Because the estimated rate ratio is less than 1 and the statistical "
            "test shows "
        )
        if qualitative_strength == "strong":
            interpretation += (
                "strong evidence (very small p‑value with the entire confidence "
                "interval below 1) "
            )
        else:
            interpretation += (
                "moderate evidence (p < 0.05 with the confidence interval below 1) "
            )
        interpretation += (
            "humans appear to have lower AMTL rates per socket than non‑human "
            "primates after adjusting for age, estimated sex, and tooth class."
        )
    else:
        interpretation = (
            "The confidence interval for the human rate ratio includes 1 or the "
            "p‑value is not conventionally significant, so the data do not provide "
            "clear evidence that humans differ from non‑human primates in AMTL "
            "rates after adjusting for age, estimated sex, and tooth class."
        )

    explanation_parts.append("Interpretation: " + interpretation)
    explanation_parts.append(
        f"On a 0–100 scale where 0 represents a strong 'No' and 100 represents a "
        f"strong 'Yes', I summarize the strength of evidence for the statement "
        f"'humans have higher AMTL frequencies than non‑human primates after "
        f"accounting for age, sex, and tooth class' as {response}."
    )

    explanation = "\n\n".join(explanation_parts)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

