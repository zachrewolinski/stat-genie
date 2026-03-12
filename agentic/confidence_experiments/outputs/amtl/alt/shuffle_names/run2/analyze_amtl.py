import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    # Load metadata (not strictly required for modeling, but documents context)
    info_path = base_path / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    data_path = base_path / "amtl.csv"
    df = pd.read_csv(data_path)

    # The column names in this run are shuffled; use the descriptions in info.json
    # and the observed values to map them back to their semantic meaning.
    #
    # sockets:      categorical, values Anterior/Posterior/Premolar -> tooth class
    # prob_male:    specimen identifier string
    # genus:        number of missing teeth of given class
    # age:          number of observable sockets that could be scored
    # pop:          estimated age at death
    # num_amtl:     uncertainty (stdev) of age at death
    # stdev_age:    estimate of sex (probability of male, 0–1)
    # tooth_class:  specimen genus (Homo sapiens, Pan, Papio, Pongo)
    # specimen:     region / population label

    df = df.copy()
    df["tooth_class_cat"] = df["sockets"]
    df["specimen_id"] = df["prob_male"]
    df["num_missing"] = df["genus"]
    df["n_sockets"] = df["age"]
    df["age_at_death"] = df["pop"]
    df["age_uncertainty"] = df["num_amtl"]
    df["prob_male_numeric"] = df["stdev_age"]
    df["genus_label"] = df["tooth_class"]
    df["region"] = df["specimen"]

    # Basic cleaning: drop rows with non-positive sockets or missing key values
    df = df[df["n_sockets"] > 0].copy()
    df = df.dropna(
        subset=[
            "num_missing",
            "n_sockets",
            "age_at_death",
            "prob_male_numeric",
            "tooth_class_cat",
            "genus_label",
        ]
    )

    # Create outcome: proportion of missing teeth in each specimen/tooth-class cell
    df["amtl_rate"] = df["num_missing"] / df["n_sockets"]

    # Sanity clamp: restrict to [0, 1]
    df = df[(df["amtl_rate"] >= 0.0) & (df["amtl_rate"] <= 1.0)].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus_label"] == "Homo sapiens").astype(int)

    # Observed (unadjusted) AMTL frequencies by group
    human_mask = df["is_human"] == 1
    nonhuman_mask = df["is_human"] == 0

    human_missing = df.loc[human_mask, "num_missing"].sum()
    human_sockets = df.loc[human_mask, "n_sockets"].sum()
    non_missing = df.loc[nonhuman_mask, "num_missing"].sum()
    non_sockets = df.loc[nonhuman_mask, "n_sockets"].sum()

    obs_rate_human = float(human_missing / human_sockets) if human_sockets > 0 else np.nan
    obs_rate_non = float(non_missing / non_sockets) if non_sockets > 0 else np.nan

    # Fit binomial regression: AMTL proportion as a function of
    # human vs non-human, tooth class, age, and sex (probability male),
    # with the number of sockets as binomial trial weights.
    formula = "amtl_rate ~ is_human + C(tooth_class_cat) + age_at_death + prob_male_numeric"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()

    # Extract effect of being human
    coef_human = float(result.params.get("is_human", np.nan))
    se_human = float(result.bse.get("is_human", np.nan))
    p_human = float(result.pvalues.get("is_human", np.nan))

    # Predicted AMTL probabilities at mean covariate values and most common tooth class
    mean_age = float(df["age_at_death"].mean())
    mean_prob_male = float(df["prob_male_numeric"].mean())
    common_tooth_class = df["tooth_class_cat"].mode().iloc[0]

    pred_df = pd.DataFrame(
        {
            "is_human": [0, 1],
            "tooth_class_cat": [common_tooth_class, common_tooth_class],
            "age_at_death": [mean_age, mean_age],
            "prob_male_numeric": [mean_prob_male, mean_prob_male],
        }
    )
    pred_probs = result.predict(pred_df)
    pred_nonhuman = float(pred_probs.iloc[0])
    pred_human = float(pred_probs.iloc[1])

    # Map statistical evidence to a 0–100 Likert scale where
    # 0 = strong "No, humans do NOT have higher AMTL"
    # 100 = strong "Yes, humans DO have higher AMTL".
    if np.isnan(coef_human) or np.isnan(p_human):
        scale = 50
        qualitative = "inconclusive evidence about differences in AMTL frequency"
        answer_text = "There is no clear evidence that modern humans have higher AMTL frequencies than non-human primates."
    else:
        if coef_human > 0:
            # Humans estimated to have higher AMTL
            if p_human < 0.001:
                scale = 95
                strength = "very strong"
            elif p_human < 0.01:
                scale = 90
                strength = "strong"
            elif p_human < 0.05:
                scale = 80
                strength = "moderate"
            elif p_human < 0.1:
                scale = 65
                strength = "weak"
            else:
                scale = 55
                strength = "very weak"
            qualitative = f"{strength} evidence that humans have higher AMTL frequencies than non-human primates"
            answer_text = "The model favors a 'Yes' answer: modern humans appear to have higher AMTL frequencies than non-human primates after adjusting for covariates."
        else:
            # Humans estimated to have similar or lower AMTL
            if p_human < 0.001:
                scale = 5
                strength = "very strong"
            elif p_human < 0.01:
                scale = 10
                strength = "strong"
            elif p_human < 0.05:
                scale = 20
                strength = "moderate"
            elif p_human < 0.1:
                scale = 35
                strength = "weak"
            else:
                scale = 45
                strength = "very weak"
            qualitative = f"{strength} evidence that humans do not have higher AMTL frequencies than non-human primates"
            answer_text = "The model favors a 'No' answer: modern humans do not appear to have higher AMTL frequencies than non-human primates after adjusting for covariates."

    scale_int = int(round(scale))

    # Build textual explanation summarizing data, model, and interpretation.
    explanation_parts = [
        info["research_questions"][0],
        f"The analysis used {len(df)} specimen–tooth-class observations from the provided dataset, ",
        f"with a total of {int(human_sockets)} observed tooth sockets for modern humans and {int(non_sockets)} sockets for non-human primates.",
        f"Unadjusted AMTL frequencies (missing teeth divided by observable sockets) were {obs_rate_human:.3f} for humans and {obs_rate_non:.3f} for non-human primates.",
        "To account for age, sex, and tooth class, I fit a binomial logistic regression of the AMTL proportion on an indicator for modern humans (Homo sapiens) versus non-human primates (Pan, Papio, Pongo),",
        "including covariates for estimated age at death, probability of being male, and tooth class (anterior, posterior, premolar),",
        f"and using the number of observable sockets as binomial trial weights.",
        f"In this model, the coefficient for the human indicator was {coef_human:.3f} (standard error {se_human:.3f}, p-value {p_human:.3g}).",
        f"At the mean covariate values and the most common tooth class ({common_tooth_class}), the adjusted predicted AMTL frequencies were {pred_nonhuman:.3f} for non-human primates and {pred_human:.3f} for humans.",
        f"These results provide {qualitative}.",
        answer_text,
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', this evidence corresponds to a value of {scale_int}.",
    ]

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": scale_int,
        "explanation": explanation,
    }

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)

    # Also print a brief summary to stdout for interactive inspection.
    print("Human coefficient:", coef_human, "p-value:", p_human)
    print("Observed AMTL rates - human:", obs_rate_human, "non-human:", obs_rate_non)
    print("Predicted AMTL rates at mean covariates - human:", pred_human, "non-human:", pred_nonhuman)
    print("Likert response (0-100):", scale_int)


if __name__ == "__main__":
    main()

