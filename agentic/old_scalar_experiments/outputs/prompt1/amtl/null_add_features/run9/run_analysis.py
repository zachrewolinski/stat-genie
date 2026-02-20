import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data():
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Keep only the genera relevant to the research question
    target_genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Define human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial response: proportion of missing teeth and corresponding denominators
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Drop any rows with missing values in key fields
    df = df.dropna(subset=["amtl_prop", "is_human", "age", "prob_male", "tooth_class", "sockets"])

    return df


def fit_model(df: pd.DataFrame):
    # Binomial GLM with logit link, using sockets as denominators (var_weights)
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(result, df: pd.DataFrame):
    coef = float(result.params["is_human"])
    pval = float(result.pvalues["is_human"])

    # Odds ratio for humans vs non-humans
    odds_ratio = float(np.exp(coef))

    # 95% CI for odds ratio
    conf_int = result.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(conf_int.iloc[0]))
    or_ci_high = float(np.exp(conf_int.iloc[1]))

    # Average predicted AMTL probabilities for humans vs non-humans,
    # holding age, sex, and tooth class at their observed values.
    df_all = df.copy()

    df_human = df_all.copy()
    df_human["is_human"] = 1
    mean_pred_human = float(result.predict(df_human).mean())

    df_non = df_all.copy()
    df_non["is_human"] = 0
    mean_pred_non = float(result.predict(df_non).mean())

    return {
        "coef": coef,
        "pval": pval,
        "odds_ratio": odds_ratio,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "mean_pred_human": mean_pred_human,
        "mean_pred_non": mean_pred_non,
        "n_rows": int(df.shape[0]),
    }


def build_conclusion(summary: dict):
    coef = summary["coef"]
    pval = summary["pval"]
    odds_ratio = summary["odds_ratio"]
    or_ci_low = summary["or_ci_low"]
    or_ci_high = summary["or_ci_high"]
    mean_pred_human = summary["mean_pred_human"]
    mean_pred_non = summary["mean_pred_non"]
    n_rows = summary["n_rows"]

    # Decision rule: humans have higher AMTL if the human indicator coefficient
    # is positive and statistically significant at alpha = 0.05.
    if coef > 0 and pval < 0.05:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "I fit a binomial regression model for the proportion of missing teeth "
        "(num_amtl / sockets) with a logit link, using the number of observable "
        "sockets as the binomial denominator. The model included a human indicator "
        "(Homo sapiens vs. Pan/Papio/Pongo), age at death, probability of being male, "
        "and tooth class (anterior, posterior, premolar) as predictors, based on "
        f"{n_rows} observations from the AMTL dataset. "
        f"The estimated coefficient for the human indicator was {coef:.3f}, with a "
        f"p-value of {pval:.3g}. This corresponds to an odds ratio of "
        f"{odds_ratio:.2f} for AMTL in humans relative to non-human primates "
        f"(95% CI {or_ci_low:.2f}–{or_ci_high:.2f}). "
        f"Using the fitted model, the average predicted probability of antemortem "
        f"tooth loss per observable socket was {mean_pred_human:.3f} for humans and "
        f"{mean_pred_non:.3f} for non-human primates when holding age, sex, and "
        "tooth class at their observed values. "
        "Based on the sign and statistical significance of the human indicator, "
        "I therefore concluded that modern humans "
        + ("do" if response == "Yes" else "do not")
        + " have higher frequencies of antemortem tooth loss than the non-human primate "
        "genera considered, after accounting for age, sex, and tooth class."
    )

    return {"response": response, "explanation": explanation}


def main():
    df = load_data()
    result = fit_model(df)
    summary = summarize_effect(result, df)
    conclusion = build_conclusion(summary)

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

