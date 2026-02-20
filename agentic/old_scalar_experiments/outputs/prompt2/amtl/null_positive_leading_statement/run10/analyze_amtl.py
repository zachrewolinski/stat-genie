import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Indicator for modern humans (Homo sapiens / Homo)
    df["is_human"] = df["genus"].str.contains("Homo", case=False, na=False).astype(int)

    # AMTL rate per row and basic counts
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    n_obs = len(df)
    n_human = int(df["is_human"].sum())
    n_nonhuman = int(n_obs - n_human)

    # Unadjusted AMTL rates
    human_mask = df["is_human"] == 1
    nonhuman_mask = ~human_mask

    human_rate = (
        df.loc[human_mask, "num_amtl"].sum()
        / df.loc[human_mask, "sockets"].sum()
    )
    nonhuman_rate = (
        df.loc[nonhuman_mask, "num_amtl"].sum()
        / df.loc[nonhuman_mask, "sockets"].sum()
    )

    # Binomial regression with sockets as frequency weights and
    # cluster-robust SEs by specimen.
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    coef_human = float(model.params["is_human"])
    se_human = float(model.bse["is_human"])
    p_human = float(model.pvalues["is_human"])

    # Predicted AMTL probabilities for humans vs non-humans at average covariates
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": ["Anterior", "Anterior"],
        }
    )

    pred_means = model.get_prediction(new_data).predicted_mean
    pred_nonhuman = float(pred_means[0])
    pred_human = float(pred_means[1])

    # Decide on binary answer based on sign and significance of human effect
    if coef_human > 0 and p_human < 0.05:
        response = "Yes"
    else:
        # Either effect is non-positive or not statistically convincing
        response = "No"

    # Map p-value and effect size to an intuitive confidence score
    z_human = coef_human / se_human if se_human > 0 else 0.0
    abs_z = abs(z_human)

    if p_human < 1e-4 and coef_human != 0:
        base_conf = 95
    elif p_human < 1e-3 and coef_human != 0:
        base_conf = 90
    elif p_human < 0.01 and coef_human != 0:
        base_conf = 80
    elif p_human < 0.05 and coef_human != 0:
        base_conf = 70
    else:
        base_conf = 55

    # Adjust slightly for effect size (on log-odds scale)
    effect_adjust = int(min(10, max(-10, (abs_z - 2.0) * 3)))
    confidence = int(np.clip(base_conf + effect_adjust, 0, 100))

    explanation = (
        "Using  binomial logistic regression on the AMTL dataset, I modeled the AMTL rate "
        "(num_amtl divided by sockets, with sockets as frequency weights) as a function of an indicator "
        "for modern humans versus non-human primates, age at death, estimated probability of being male, "
        "and tooth class. The dataset contains "
        f"{n_obs} specimen-tooth-class observations, of which {n_human} are humans and {n_nonhuman} are "
        "non-human primates (Pan, Pongo, and Papio). The unadjusted AMTL rate is "
        f"{human_rate:.3f} for humans and {nonhuman_rate:.3f} for non-human primates. After adjustment, "
        f"the human indicator has a log-odds coefficient of {coef_human:.3f} (standard error {se_human:.3f}, "
        f"p-value {p_human:.3g}), corresponding to predicted AMTL probabilities of "
        f"{pred_nonhuman:.3f} for non-human primates and {pred_human:.3f} for humans at average age and sex "
        "and for anterior teeth. "
        "Based on the sign and statistical significance of the human coefficient in this model, "
        f"I answer '{response}' to the question of whether modern humans have higher AMTL frequencies than "
        "non-human primates after accounting for age, sex, and tooth class, while acknowledging that the "
        "conclusion depends on this particular regression specification and does not account for additional "
        "sources of uncertainty such as unmodeled population or specimen-level heterogeneity."
    )

    result = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
