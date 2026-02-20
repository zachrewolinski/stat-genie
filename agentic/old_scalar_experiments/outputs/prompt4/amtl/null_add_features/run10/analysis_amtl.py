import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Keep only variables needed for this analysis
    cols = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus", "specimen"]
    df = df[cols].dropna()

    # Basic filters and derived variables
    df = df[df["sockets"] > 0].copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    if df["human"].nunique() < 2:
        raise ValueError("Data must contain both human and non-human primate specimens.")

    # Descriptive summary by genus
    genus_summary = (
        df.groupby("genus", as_index=False)
        .agg(mean_prop=("prop_amtl", "mean"), n=("prop_amtl", "size"))
        .sort_values("genus")
    )
    print("Mean AMTL proportion (num_amtl / sockets) by genus:")
    print(genus_summary.to_string(index=False))

    # Binomial regression: proportion of missing teeth, weighted by number of sockets
    model = smf.glm(
        formula="prop_amtl ~ human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    print("\nBinomial regression results:")
    print(result.summary())

    # Extract human effect
    human_coef = float(result.params["human"])
    human_p = float(result.pvalues["human"])

    # Marginal predictions for humans vs non-humans at observed covariate values
    df_pred = df.copy()
    df_pred_human = df_pred.copy()
    df_pred_human["human"] = 1
    df_pred_nonhuman = df_pred.copy()
    df_pred_nonhuman["human"] = 0

    mean_prob_human = float(result.predict(df_pred_human).mean())
    mean_prob_nonhuman = float(result.predict(df_pred_nonhuman).mean())
    diff = mean_prob_human - mean_prob_nonhuman

    # Raw proportions by human vs non-human
    human_raw = float(df.loc[df["human"] == 1, "prop_amtl"].mean())
    nonhuman_raw = float(df.loc[df["human"] == 0, "prop_amtl"].mean())

    # Translate results into a 0–100 Likert scale response
    if diff > 0 and human_p < 0.001:
        if diff >= 0.05:
            score = 95
        elif diff >= 0.02:
            score = 85
        else:
            score = 75
    elif diff > 0 and human_p < 0.05:
        score = 70
    elif diff > 0:
        score = 60
    elif diff < 0 and human_p < 0.001:
        if diff <= -0.05:
            score = 5
        elif diff <= -0.02:
            score = 15
        else:
            score = 25
    elif diff < 0 and human_p < 0.05:
        score = 30
    elif diff < 0:
        score = 40
    else:
        score = 50

    explanation_parts = []
    explanation_parts.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class?"
    )
    explanation_parts.append(
        f"The analyzed dataset contains {len(df)} tooth-class observations across "
        f"{df['specimen'].nunique()} unique specimens and {df['genus'].nunique()} genera."
    )
    explanation_parts.append(
        f"Raw AMTL proportions (num_amtl / sockets) average {human_raw:.3f} for humans and "
        f"{nonhuman_raw:.3f} for non-human primates."
    )
    explanation_parts.append(
        "To adjust for covariates, I fit a binomial regression with a logit link, using the "
        "proportion of missing teeth as the response and weighting each observation by the "
        "number of observable sockets. Predictors were an indicator for humans versus "
        "non-human primates, age at death, probability of being male, and categorical tooth class."
    )
    explanation_parts.append(
        f"In this model, the human indicator has coefficient {human_coef:.3f} with p-value "
        f"{human_p:.3g}. On the model scale, the average predicted AMTL probability is "
        f"{mean_prob_human:.3f} for humans and {mean_prob_nonhuman:.3f} for non-human primates, "
        f"a difference of {diff * 100:.1f} percentage points in favor of humans."
    )

    if diff > 0:
        explanation_parts.append(
            "These results indicate that, after accounting for age, sex, and tooth class, "
            "modern humans do exhibit higher AMTL frequencies than the non-human primate genera "
            "considered here."
        )
    elif diff < 0:
        explanation_parts.append(
            "These results indicate that, once age, sex, and tooth class are controlled, "
            "modern humans exhibit lower AMTL frequencies than the non-human primate genera "
            "considered here."
        )
    else:
        explanation_parts.append(
            "These results indicate no meaningful difference in AMTL frequencies between humans "
            "and non-human primates after adjusting for age, sex, and tooth class."
        )

    if human_p < 0.001:
        explanation_parts.append(
            "The human effect is highly statistically significant (p < 0.001), so the observed "
            "difference is unlikely to be due to random sampling variation."
        )
    elif human_p < 0.05:
        explanation_parts.append(
            "The human effect is statistically significant at the 5% level, so the observed "
            "difference is unlikely to be due to random sampling variation."
        )
    else:
        explanation_parts.append(
            "However, the human effect is not statistically significant at conventional levels, "
            "so the evidence for a difference is weak and should be interpreted cautiously."
        )

    explanation = "\n\n".join(explanation_parts)

    conclusion = {"response": int(score), "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

