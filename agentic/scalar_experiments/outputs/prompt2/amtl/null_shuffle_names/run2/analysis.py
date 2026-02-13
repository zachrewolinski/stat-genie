import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # The column names in this anonymized version are shuffled relative to their semantics.
    # Based on `info.json` descriptions and inspection of the CSV:
    # - `tooth_class` column actually holds the genus (Homo sapiens, Pan, Papio, Pongo)
    # - `sockets` is the tooth class (Anterior, Posterior, Premolar)
    # - `genus` is the count of missing teeth of that class
    # - `age` is the number of observable sockets
    # - `pop` is estimated age at death
    # - `stdev_age` is the estimated probability the specimen is male
    df = df.copy()
    df.rename(columns={"tooth_class": "species"}, inplace=True)
    df["tooth_class"] = df["sockets"]
    df["n_missing"] = df["genus"].astype(float)
    df["n_sockets"] = df["age"].astype(float)
    df["age_death"] = df["pop"].astype(float)
    df["sex_prob_male"] = df["stdev_age"].astype(float)

    # Basic filtering to ensure a valid binomial structure.
    df = df[df["n_sockets"] > 0].copy()
    df = df[df["n_missing"] >= 0].copy()
    df = df[df["n_missing"] <= df["n_sockets"]].copy()

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = (df["species"] == "Homo sapiens").astype(int)

    # Proportion of teeth missing in each row.
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]
    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression on the proportion of missing teeth, with sockets as frequency weights.
    model = smf.glm(
        formula="prop_missing ~ is_human + age_death + sex_prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    # Extract coefficient for humans vs non-humans.
    if "is_human" not in result.params:
        # Fallback: if model failed to include the term, treat as no detectable effect.
        coef = 0.0
        pvalue = 1.0
        ci_low, ci_high = 0.0, 0.0
    else:
        coef = float(result.params["is_human"])
        pvalue = float(result.pvalues["is_human"])
        ci_low, ci_high = map(float, result.conf_int().loc["is_human"])

    # Predicted average AMTL probability if everyone were human vs non-human,
    # holding age, sex, and tooth class at their observed values.
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0

    pred_human = float(result.predict(df_human).mean())
    pred_nonhuman = float(result.predict(df_nonhuman).mean())

    # Simple mapping from p-value and effect direction to a 0–100 confidence.
    if pvalue < 1e-4:
        base_conf = 97
    elif pvalue < 1e-3:
        base_conf = 93
    elif pvalue < 1e-2:
        base_conf = 88
    elif pvalue < 5e-2:
        base_conf = 75
    elif pvalue < 1e-1:
        base_conf = 60
    else:
        base_conf = 45

    if coef == 0.0:
        direction = "none"
    elif coef > 0:
        direction = "human_higher"
    else:
        direction = "human_lower"

    if direction == "human_higher":
        response = "Yes"
        confidence = base_conf
    elif direction == "human_lower":
        response = "No"
        confidence = base_conf
    else:
        # No clear effect estimated.
        response = "No"
        confidence = min(base_conf, 50)

    # Clip confidence to [0, 100].
    confidence = max(0, min(100, int(round(confidence))))

    # Descriptive group-level means (unadjusted) for context.
    group_means = (
        df.assign(amtl_rate=df["n_missing"] / df["n_sockets"])
        .groupby("species")["amtl_rate"]
        .mean()
        .to_dict()
    )

    explanation = (
        "I analyzed the AMTL dataset using a binomial regression model where the outcome was the "
        "proportion of missing teeth (number of missing teeth divided by observable sockets) for each "
        "specimen and tooth class. The key predictor was whether the specimen was a modern human "
        "(Homo sapiens) versus a non-human primate (Pan, Pongo, Papio), and I controlled for estimated "
        "age at death, estimated probability of being male, and tooth class (anterior, posterior, premolar). "
        f"The estimated coefficient for the human indicator in the logit model was {coef:.3f} "
        f"with a 95% confidence interval from {ci_low:.3f} to {ci_high:.3f} and p-value {pvalue:.3g}. "
        f"Under the fitted model, the average predicted AMTL probability if all observations were humans was "
        f"{pred_human:.3f}, compared to {pred_nonhuman:.3f} if all observations were non-human primates. "
        "Unadjusted mean AMTL rates by genus (proportion of missing teeth) were: "
        + ", ".join(
            f"{species}: {rate:.3f}" for species, rate in sorted(group_means.items())
        )
        + ". "
        "Because the human coefficient is "
        + ("positive" if direction == "human_higher" else "negative or near zero")
        + f" and the associated p-value is {pvalue:.3g}, I "
        + ("conclude that modern humans have higher AMTL frequencies than the non-human genera, "
           "after accounting for age, sex, and tooth class."
           if direction == "human_higher"
           else "do not find evidence that modern humans have higher AMTL frequencies than the non-human genera "
                "once age, sex, and tooth class are taken into account.")
    )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    result = fit_model(df)
    summary = summarize_effect(df, result)

    # Write the required JSON object to conclusion.txt with no extra text.
    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "response": summary["response"],
                "confidence": summary["confidence"],
                "explanation": summary["explanation"],
            },
            f,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()

