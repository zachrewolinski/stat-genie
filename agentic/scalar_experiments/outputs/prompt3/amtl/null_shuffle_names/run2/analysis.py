import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # The column names in this variant of the dataset are somewhat shuffled
    # relative to their semantic meaning. We remap them here based on the
    # metadata descriptions and observed values.
    #
    # Original semantic variables (from metadata):
    #   - genus: Homo sapiens, Pan, Papio, Pongo
    #   - tooth_class: Anterior, Posterior, Premolar
    #   - num_amtl: number of missing teeth
    #   - sockets: number of observable sockets
    #   - age: estimated age at death
    #   - prob_male: estimate of sex (probability male)
    #
    # In this CSV:
    #   - "tooth_class" actually holds genus labels (Homo sapiens, Pan, Papio, Pongo)
    #   - "sockets" holds tooth class (Anterior/Posterior/Premolar)
    #   - "genus" is a small integer count (interpreted as num_amtl)
    #   - "age" is a small integer count (interpreted as sockets)
    #   - "pop" is a continuous value in ~[8, 70] (interpreted as age at death)
    #   - "stdev_age" ranges from 0 to 1 in coarse steps (interpreted as prob_male)
    #
    # We construct clean semantic columns for analysis.

    df = df.copy()
    df["genus_label"] = df["tooth_class"]
    df["tooth_class_label"] = df["sockets"]

    # Counts for binomial model
    df["num_missing"] = df["genus"].astype(float)
    df["num_sockets"] = df["age"].astype(float)

    # Guard against any rows where counts are not sensible
    df = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0)]
    df = df[df["num_missing"] <= df["num_sockets"]]

    # Covariates
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)

    # Define human vs non-human indicator
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Tooth class as categorical
    df["tooth_class_cat"] = df["tooth_class_label"].astype("category")

    # Proportion of missing teeth per observation for descriptive summaries
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial logistic regression model:
        logit(p_missing) ~ is_human + age_at_death + prob_male + tooth_class_cat
    using num_sockets as binomial trials.
    """
    # Use proportion as response with weights = num_sockets. This is the
    # standard way to fit a binomial model with grouped data in statsmodels.
    df = df.copy()

    # Small numerical safeguard: clamp proportions into (0,1) for stability
    eps = 1e-6
    df["prop_missing_clamped"] = df["prop_missing"].clip(eps, 1 - eps)

    formula = "prop_missing_clamped ~ is_human + age_at_death + prob_male + C(tooth_class_cat)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    """
    Summarize whether humans have higher AMTL frequencies than non-human primates
    after accounting for covariates, based on the fitted model.
    """
    params = result.params
    conf_int = result.conf_int()

    # The coefficient for is_human is on the log-odds scale.
    coef_human = params.get("is_human", np.nan)
    ci_low, ci_high = conf_int.loc["is_human"]

    # Convert to odds ratio for interpretability
    odds_ratio = float(np.exp(coef_human))
    ci_low_or = float(np.exp(ci_low))
    ci_high_or = float(np.exp(ci_high))

    # Also compute simple descriptive stats: mean proportion missing
    mean_prop_by_genus = (
        df.groupby("genus_label")["prop_missing"].mean().sort_index()
    )
    human_mean = float(mean_prop_by_genus.get("Homo sapiens", np.nan))
    non_human_mean = float(
        df.loc[df["genus_label"] != "Homo sapiens", "prop_missing"].mean()
    )

    # Determine qualitative answer
    # We treat "Yes" as: coefficient > 0 and its 95% CI mostly above 0,
    # plus humans having higher mean proportion descriptively.
    if np.isnan(coef_human):
        response = "No"
        strength = 20
        confidence = 30
        reasoning = (
            "The regression model could not estimate a separate effect for humans, "
            "so there is insufficient evidence from this dataset to conclude that "
            "modern humans have higher AMTL frequencies than non-human primates."
        )
    else:
        # Significance and direction
        coef_positive = coef_human > 0
        ci_above_zero = ci_low > 0
        ci_below_zero = ci_high < 0

        humans_higher_descriptive = human_mean > non_human_mean

        if coef_positive and ci_above_zero and humans_higher_descriptive:
            response = "Yes"
            # Strong evidence: positive, significant, and consistent with descriptives
            strength = 85
            confidence = 80
        elif coef_positive and humans_higher_descriptive and not ci_below_zero:
            # Positive but CI overlaps zero: suggestive but not definitive
            response = "Yes"
            strength = 60
            confidence = 55
        else:
            response = "No"
            if coef_positive or humans_higher_descriptive:
                # Mixed signals
                strength = 45
                confidence = 50
            else:
                # Evidence points away from humans having higher AMTL
                strength = 70
                confidence = 75

        # Build human-readable reasoning
        reasoning = (
            "I fit a binomial logistic regression model predicting the proportion of missing "
            "teeth (AMTL) per observation from a binary indicator for modern humans versus "
            "non-human primates, age at death, estimated sex (probability male), and tooth "
            "class (anterior, posterior, premolar). The model treated the number of observable "
            "sockets as binomial trials and the number of missing teeth as successes. "
            f"The estimated log-odds coefficient for modern humans (relative to non-human primates) "
            f"was {coef_human:.3f}, corresponding to an odds ratio of approximately {odds_ratio:.2f} "
            f"with a 95% confidence interval from {ci_low_or:.2f} to {ci_high_or:.2f}. "
            f"Descriptively, the mean proportion of missing teeth was about {human_mean:.3f} for "
            f"modern humans and {non_human_mean:.3f} for non-human primates. "
            "These results were then used to decide whether there is evidence that modern humans "
            "have higher AMTL frequencies after accounting for age, sex, and tooth class."
        )

    return {
        "response": response,
        "strength": int(strength),
        "confidence": int(confidence),
        "explanation": reasoning,
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    result = fit_binomial_model(df)
    summary = summarize_effect(df, result)

    # Write JSON output to conclusion.txt with no extra text.
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()

