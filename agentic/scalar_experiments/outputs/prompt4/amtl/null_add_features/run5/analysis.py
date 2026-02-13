import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Keep only rows with valid values for the variables we use
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "genus",
            "tooth_class",
        ]
    ).copy()

    # Exclude rows with non-positive socket counts
    df = df[df["sockets"] > 0].copy()

    # Define outcome as proportion of teeth missing in the tooth class
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure tooth_class is treated as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    # If there are no non-human primates or no humans, we cannot answer the question
    if df["is_human"].nunique() < 2:
        response = 50
        explanation = (
            "The dataset does not contain both modern humans (Homo sapiens) and "
            "non-human primates, so it is not possible to compare AMTL frequencies "
            "between them. I therefore report a neutral response of 50 on the 0–100 scale."
        )
        with open("conclusion.txt", "w") as f:
            json.dump({"response": response, "explanation": explanation}, f)
        return

    # Binomial regression: prop_missing with sockets as the number of trials
    # Adjusting for age, sex (prob_male), and tooth class.
    model = smf.glm(
        "prop_missing ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = model.params.get("is_human", np.nan)
    se = model.bse.get("is_human", np.nan)

    if np.isnan(coef) or not np.isfinite(se) or se == 0:
        # Fall back to neutral if the model cannot estimate the human effect
        response_value = 50
        explanation = (
            "A binomial regression model adjusting for age, sex, and tooth class "
            "could not reliably estimate the effect of being a modern human on AMTL "
            "frequency (the coefficient was undefined or had zero standard error). "
            "Because of this modeling instability, I return a neutral response of 50 "
            "on the 0–100 scale."
        )
    else:
        z = coef / se
        # Approximate probability that the human effect on AMTL is positive
        prob_yes = float(norm.cdf(z))
        response_value = int(round(prob_yes * 100))

        # Compute descriptive statistics: overall AMTL rates for humans vs. non-humans
        def weighted_mean_prop(group: pd.DataFrame) -> float:
            sockets_sum = group["sockets"].sum()
            if sockets_sum <= 0:
                return float("nan")
            return float(group["num_amtl"].sum() / sockets_sum)

        human = df[df["is_human"] == 1]
        nonhuman = df[df["is_human"] == 0]
        mean_human = weighted_mean_prop(human)
        mean_nonhuman = weighted_mean_prop(nonhuman)

        odds_ratio = float(np.exp(coef))
        p_value = float(2 * (1 - norm.cdf(abs(z))))

        # Predicted probabilities at a typical reference profile
        ref_age = float(df["age"].median())
        ref_prob_male = 0.5
        ref_tooth_class = df["tooth_class"].mode().iat[0]

        ref_df = pd.DataFrame(
            {
                "is_human": [1, 0],
                "age": [ref_age, ref_age],
                "prob_male": [ref_prob_male, ref_prob_male],
                "tooth_class": [ref_tooth_class, ref_tooth_class],
            }
        )
        pred_probs = model.predict(ref_df)
        pred_human = float(pred_probs.iloc[0])
        pred_nonhuman = float(pred_probs.iloc[1])
        diff_pred = pred_human - pred_nonhuman

        explanation = (
            "Research question: Do modern humans (Homo sapiens) have higher frequencies "
            "of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio), "
            "after accounting for age, sex, and tooth class?\n\n"
            "Using the dataset in amtl.csv, I modeled the proportion of teeth missing "
            "in each tooth class (num_amtl / sockets) with a binomial regression. "
            "The outcome was the probability that a tooth socket was missing, and predictors "
            "included an indicator for modern humans vs. non-human primates (is_human), "
            "age at death, the probability of being male (prob_male), and categorical "
            "tooth class.\n\n"
            f"In this model, the coefficient for being a modern human (is_human) was "
            f"{coef:.3f} (standard error {se:.3f}), corresponding to an odds ratio of "
            f"{odds_ratio:.2f} for AMTL in humans relative to non-human primates when "
            "holding age, sex, and tooth class constant. The associated two-sided "
            f"p-value was approximately {p_value:.3g}, and the z-statistic was {z:.2f}.\n\n"
            f"Descriptively, the overall AMTL rate (missing teeth divided by observable "
            f"sockets) was about {mean_human:.3f} for modern humans and {mean_nonhuman:.3f} "
            "for non-human primates. For a typical individual at the median age in the "
            f"sample (about {ref_age:.1f} years), with prob_male = 0.5 and the most common "
            f"tooth class ({ref_tooth_class}), the model predicted an AMTL probability of "
            f"{pred_human:.3f} for modern humans and {pred_nonhuman:.3f} for non-human "
            f"primates, a difference of {diff_pred:.3f}.\n\n"
            f"Interpreting the z-statistic via a normal approximation, the implied "
            f"probability that the true human effect on AMTL is positive (i.e., humans "
            f"have higher AMTL than non-human primates, after adjustment) is about "
            f"{prob_yes:.2f}. Mapping this probability onto a 0–100 Likert scale yields "
            f"a response value of {response_value}, where higher values support the "
            "statement that humans have higher AMTL frequencies.\n\n"
            "Overall, the regression results and descriptive statistics together indicate "
            "the direction and strength of evidence regarding whether modern humans exhibit "
            "higher adjusted AMTL frequencies than non-human primates."
        )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": int(response_value), "explanation": explanation}, f)


if __name__ == "__main__":
    main()

