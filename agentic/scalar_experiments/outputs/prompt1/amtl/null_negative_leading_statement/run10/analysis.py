import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data():
    info_path = Path("info.json")
    data_path = Path("amtl.csv")

    with info_path.open() as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    return info, df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning: drop rows with missing key variables and ensure positive socket counts
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]
    ).copy()
    df = df[df["sockets"] > 0].copy()

    # Proportion of missing teeth in each tooth class for each specimen
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression on AMTL proportion with number of sockets as trial weights
    model = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result) -> dict:
    coef = result.params["is_human"]
    pval = result.pvalues["is_human"]
    conf_int = result.conf_int().loc["is_human"]

    odds_ratio = float(np.exp(coef))
    or_low, or_high = np.exp(conf_int.to_numpy())

    # Descriptive, socket-weighted AMTL frequencies
    df = df.copy()
    df["rate"] = df["num_amtl"] / df["sockets"]
    human_mask = df["is_human"] == 1
    nonhuman_mask = df["is_human"] == 0

    human_rate = float(
        np.average(df.loc[human_mask, "rate"], weights=df.loc[human_mask, "sockets"])
    )
    nonhuman_rate = float(
        np.average(
            df.loc[nonhuman_mask, "rate"], weights=df.loc[nonhuman_mask, "sockets"]
        )
    )

    n_human_rows = int(human_mask.sum())
    n_nonhuman_rows = int(nonhuman_mask.sum())

    total_human_sockets = int(df.loc[human_mask, "sockets"].sum())
    total_nonhuman_sockets = int(df.loc[nonhuman_mask, "sockets"].sum())

    total_human_missing = int(df.loc[human_mask, "num_amtl"].sum())
    total_nonhuman_missing = int(df.loc[nonhuman_mask, "num_amtl"].sum())

    return {
        "coef": float(coef),
        "pval": float(pval),
        "odds_ratio": odds_ratio,
        "or_low": float(or_low),
        "or_high": float(or_high),
        "human_rate": human_rate,
        "nonhuman_rate": nonhuman_rate,
        "n_human_rows": n_human_rows,
        "n_nonhuman_rows": n_nonhuman_rows,
        "total_human_sockets": total_human_sockets,
        "total_nonhuman_sockets": total_nonhuman_sockets,
        "total_human_missing": total_human_missing,
        "total_nonhuman_missing": total_nonhuman_missing,
    }


def determine_response(effect: dict, alpha: float = 0.05) -> str:
    # Positive coefficient means humans have higher AMTL odds than non-human primates
    if effect["coef"] > 0.0 and effect["pval"] < alpha:
        return "Yes"
    return "No"


def build_explanation(info: dict, effect: dict, response: str) -> str:
    question = info.get("research_questions", [""])[0]

    direction = (
        "higher"
        if effect["coef"] > 0
        else "lower" if effect["coef"] < 0 else "no clear difference in"
    )

    explanation_parts = [
        "Research question:",
        question,
        "",
        "Data and outcome:",
        "- The dataset contains AMTL counts (number of missing teeth) and the number of observable tooth sockets",
        "  for modern humans (Homo sapiens) and three non-human primate genera (Pan, Papio, Pongo).",
        f"- After basic cleaning, the analysis used {effect['n_human_rows']} human rows and {effect['n_nonhuman_rows']} non-human rows.",
        f"- These rows correspond to {effect['total_human_sockets']} human sockets with {effect['total_human_missing']} missing teeth",
        f"  and {effect['total_nonhuman_sockets']} non-human sockets with {effect['total_nonhuman_missing']} missing teeth.",
        f"- Socket-weighted mean AMTL frequency is {effect['human_rate']:.3f} for humans and {effect['nonhuman_rate']:.3f} for non-human primates.",
        "",
        "Modeling approach:",
        "- I fit a binomial regression (GLM with logit link) to model the proportion of missing teeth (num_amtl / sockets).",
        "- Each row's contribution was weighted by the number of sockets so that specimens with more observable teeth have more influence.",
        "- The predictors included an indicator for humans vs. non-human primates (is_human), age at death, sex estimate (prob_male),",
        "  and tooth class (anterior, posterior, premolar) as categorical covariates. This directly addresses the request to account",
        "  for age, sex, and tooth class when comparing AMTL frequencies.",
        "",
        "Key result for humans vs. non-human primates:",
        f"- The coefficient for the human indicator (is_human) in the regression is {effect['coef']:.3f}, corresponding to an odds ratio",
        f"  of {effect['odds_ratio']:.2f} (95% CI: {effect['or_low']:.2f}–{effect['or_high']:.2f}).",
        f"- The p-value for this coefficient is {effect['pval']:.3g}.",
    ]

    if response == "Yes":
        explanation_parts.extend(
            [
                "- Because the human coefficient is positive and statistically significant (p-value below 0.05), the model indicates that,",
                "  after controlling for age, sex, and tooth class, modern humans have higher odds of antemortem tooth loss than the",
                "  combined group of non-human primates.",
                "",
                "Conclusion:",
                "Based on the binomial regression and the descriptive AMTL frequencies, there is strong evidence that modern humans",
                "have higher AMTL frequencies than non-human primates after accounting for age, sex, and tooth class. Therefore, the",
                "answer to the research question is 'Yes'.",
            ]
        )
    else:
        explanation_parts.extend(
            [
                f"- The human coefficient is {direction} but not statistically significant at the 0.05 level, so the model does not provide",
                "  strong evidence that humans differ in AMTL frequency from non-human primates after adjusting for age, sex, and tooth class.",
                "",
                "Conclusion:",
                "Given the regression results and descriptive AMTL frequencies, there is not sufficient evidence to conclude that modern humans",
                "have higher AMTL frequencies than non-human primates after accounting for age, sex, and tooth class. Therefore, the answer",
                "to the research question is 'No'.",
            ]
        )

    return "\n".join(explanation_parts)


def main():
    info, df = load_data()
    df_prepared = prepare_data(df)
    result = fit_model(df_prepared)
    effect = summarize_effect(df_prepared, result)
    response = determine_response(effect)
    explanation = build_explanation(info, effect, response)

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

