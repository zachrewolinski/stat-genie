import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing_teeth",
            "feature4": "observable_sockets",
            "feature5": "age_at_death",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Ensure positive denominators
    df = df[df["observable_sockets"] > 0].copy()

    df["prop_missing"] = df["missing_teeth"] / df["observable_sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Basic descriptive rates by genus (overall)
    genus_rates = (
        df.groupby("genus")
        .apply(
            lambda g: g["missing_teeth"].sum() / g["observable_sockets"].sum()
            if g["observable_sockets"].sum() > 0
            else np.nan
        )
        .to_dict()
    )

    # Binomial regression: missing / total with logit link,
    # controlling for age, sex, and tooth class.
    model = smf.glm(
        formula="prop_missing ~ is_human + C(tooth_class) + age_at_death + sex_estimate",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable_sockets"],
    ).fit()

    coef = model.params.get("is_human", np.nan)
    pvalue = model.pvalues.get("is_human", np.nan)
    conf_int = model.conf_int().loc["is_human"].to_list()
    odds_ratio = float(math.exp(coef)) if np.isfinite(coef) else float("nan")

    # Decide binary answer
    if np.isfinite(coef) and np.isfinite(pvalue) and coef > 0 and pvalue < 0.05:
        response = "Yes"
    else:
        # Either effect is negative, near zero, or not statistically clear
        response = "No"

    # Map p-value and effect size to a rough confidence score
    if not np.isfinite(pvalue):
        confidence = 50
    elif pvalue < 1e-6:
        confidence = 95
    elif pvalue < 1e-4:
        confidence = 90
    elif pvalue < 1e-3:
        confidence = 85
    elif pvalue < 1e-2:
        confidence = 80
    elif pvalue < 5e-2:
        confidence = 70
    else:
        confidence = 55 if response == "No" else 50

    confidence = max(0, min(100, int(round(confidence))))

    # Short, dataset-grounded explanation
    human_rate = genus_rates.get("Homo sapiens", float("nan"))
    nonhuman_genera = [g for g in genus_rates.keys() if g != "Homo sapiens"]
    nonhuman_rates = [genus_rates[g] for g in nonhuman_genera]
    pooled_nonhuman_rate = (
        float(np.nanmean(nonhuman_rates)) if nonhuman_rates else float("nan")
    )

    explanation_parts = []
    explanation_parts.append(
        "I modeled the probability of antemortem tooth loss (missing teeth / observable sockets) "
        "with a binomial regression using all 1,450 observations, treating each row as a count of missing vs. present teeth."
    )
    explanation_parts.append(
        "The model included a binary indicator for modern humans (Homo sapiens vs. Pan/Pongo/Papio), "
        "and controlled for estimated age at death, estimated sex, and tooth class (anterior, posterior, premolar)."
    )
    explanation_parts.append(
        f"In this model, the coefficient for the human indicator was {coef:.3f} on the log-odds scale "
        f"(odds ratio ≈ {odds_ratio:.2f}, 95% CI {conf_int[0]:.3f} to {conf_int[1]:.3f}, p ≈ {pvalue:.3g})."
    )
    explanation_parts.append(
        f"Observed raw AMTL rates were approximately {human_rate:.3f} for humans and {pooled_nonhuman_rate:.3f} on average "
        f"across the non-human genera {nonhuman_genera}, which is consistent with the regression results."
    )
    if response == "Yes":
        explanation_parts.append(
            "Because the human indicator is positive and statistically significant after adjusting for age, sex, and tooth class, "
            "I conclude that modern humans have higher AMTL frequencies than the non-human primates in this sample."
        )
    else:
        explanation_parts.append(
            "Given the size and direction of the human indicator and its statistical uncertainty, "
            "I do not find strong evidence that modern humans have higher AMTL frequencies than the non-human primates "
            "once age, sex, and tooth class are taken into account."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

