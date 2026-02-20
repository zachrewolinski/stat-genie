import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning and derived variables
    df = df[df["sockets"] > 0].copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Identify modern humans (Homo / Homo sapiens) vs non-human primates
    df["is_human"] = df["genus"].str.startswith("Homo").astype(int)

    # Drop rows with missing key covariates
    df = df.dropna(subset=["prop_amtl", "age", "prob_male", "tooth_class", "is_human"])

    # Descriptive: weighted mean AMTL proportion by genus
    def weighted_mean_prop(group: pd.DataFrame) -> float:
        return np.average(group["prop_amtl"], weights=group["sockets"])

    mean_by_genus = (
        df.groupby("genus", observed=True)
        .apply(weighted_mean_prop)
        .sort_values()
    )

    # Fit binomial logistic regression with sockets as binomial trials
    model = smf.glm(
        "prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])
    ci_low, ci_high = result.conf_int().loc["is_human"]

    median_age = float(df["age"].median())
    mean_prob_male = float(df["prob_male"].mean())
    common_tooth = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [median_age, median_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [common_tooth, common_tooth],
        }
    )
    pred_res = result.get_prediction(pred_df).summary_frame(alpha=0.05)
    nonhuman_prob = float(pred_res["mean"].iloc[0])
    human_prob = float(pred_res["mean"].iloc[1])

    if human_prob > nonhuman_prob and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    genus_summaries = ", ".join(
        f"{genus}: {prop:.3f}" for genus, prop in mean_by_genus.items()
    )

    human_effect_direction = "higher" if coef > 0 else "lower"

    explanation = (
        "I analyzed the antemortem tooth loss (AMTL) dataset using a binomial "
        "logistic regression model for the proportion of missing teeth "
        "(num_amtl / sockets), with the number of observable sockets used as "
        "binomial trials. The model included a binary indicator for modern "
        "humans (Homo) versus non-human primates (Pan, Papio, Pongo), along "
        "with covariates for age at death, estimated sex (prob_male), and "
        "tooth class.\n\n"
        f"The estimated coefficient for the human indicator was {coef:.3f} "
        f"(p = {pval:.3g}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]), indicating that, "
        f"after adjusting for age, sex, and tooth class, the odds of tooth loss are "
        f"{human_effect_direction} for humans relative to non-human primates. "
        f"For a typical specimen (age ≈ {median_age:.1f} years, prob_male ≈ "
        f"{mean_prob_male:.2f}, tooth class = {common_tooth}), the model predicts "
        f"an AMTL proportion of {nonhuman_prob:.3f} for non-human primates and "
        f"{human_prob:.3f} for humans.\n\n"
        f"Observed weighted mean AMTL proportions by genus (using sockets as "
        f"weights) were: {genus_summaries}. "
        "These descriptive patterns are consistent with the regression results. "
        "Based on the direction, magnitude, and statistical significance of the "
        "human effect in this model, I "
        + (
            "conclude that modern humans have higher AMTL frequencies than the non-human genera considered."
            if response == "Yes"
            else "do not find strong evidence that modern humans have higher AMTL frequencies than the non-human genera considered."
        )
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

