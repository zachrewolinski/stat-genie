import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summarize_data(df: pd.DataFrame) -> None:
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nGenus value counts:")
    print(df["genus"].value_counts())
    print("\nTooth class value counts:")
    print(df["tooth_class"].value_counts())
    print("\nBasic AMTL stats by genus:")
    summary = (
        df.assign(prop_amtl=df["num_amtl"] / df["sockets"].replace(0, np.nan))
        .groupby("genus")["prop_amtl"]
        .agg(["mean", "std", "count"])
    )
    print(summary)


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop rows with non-positive sockets to avoid invalid proportions.
    df = df[df["sockets"] > 0].copy()

    # Create proportion and ensure it is in (0, 1) for binomial modeling.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Encode sex proxy using prob_male (0–1). We keep as numeric covariate.
    # Center age to improve model stability.
    df["age_c"] = df["age"] - df["age"].mean()

    # Focus on comparison of humans vs non-human primates.
    df["is_human"] = (df["genus"].str.contains("Homo", case=False, na=False)).astype(int)

    # Keep only rows where genus is clearly one of the target taxa.
    target_mask = (
        df["genus"].str.contains("Homo", case=False, na=False)
        | df["genus"].str.contains("Pan", case=False, na=False)
        | df["genus"].str.contains("Pongo", case=False, na=False)
        | df["genus"].str.contains("Papio", case=False, na=False)
    )
    df = df[target_mask].copy()

    # Ensure categorical variables.
    df["tooth_class"] = df["tooth_class"].astype("category")

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Use a binomial GLM with the proportion as response and sockets as weights.
    # This corresponds to modeling counts of missing teeth out of total sockets.
    formula = "prop_amtl ~ is_human + age_c + prob_male + tooth_class"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def evaluate_human_effect(result, df: pd.DataFrame) -> dict:
    # Extract coefficient, standard error, and p-value for is_human.
    coef = result.params.get("is_human", np.nan)
    se = result.bse.get("is_human", np.nan)
    pval = result.pvalues.get("is_human", np.nan)

    # Approximate 95% CI on log-odds scale and convert to odds ratios.
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    odds_ratio = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    # Compute predicted probabilities for a “typical” specimen.
    # Use mean age and prob_male, and most common tooth_class.
    mean_age_c = 0.0  # centered
    mean_prob_male = float(df["prob_male"].mean())
    mode_tooth_class = df["tooth_class"].mode().iat[0]

    base_row = {
        "age_c": mean_age_c,
        "prob_male": mean_prob_male,
        "tooth_class": mode_tooth_class,
    }

    human_df = pd.DataFrame([{**base_row, "is_human": 1}])
    nonhuman_df = pd.DataFrame([{**base_row, "is_human": 0}])

    human_prob = float(result.predict(human_df)[0])
    nonhuman_prob = float(result.predict(nonhuman_df)[0])
    diff_prob = human_prob - nonhuman_prob

    return {
        "coef": float(coef),
        "se": float(se),
        "pval": float(pval),
        "odds_ratio": odds_ratio,
        "or_low": or_low,
        "or_high": or_high,
        "human_prob": human_prob,
        "nonhuman_prob": nonhuman_prob,
        "diff_prob": diff_prob,
    }


def map_effect_to_likert(effect: dict) -> int:
    pval = effect["pval"]
    odds_ratio = effect["odds_ratio"]
    diff_prob = effect["diff_prob"]

    if np.isnan(pval) or np.isnan(odds_ratio):
        return 50

    if pval < 0.001:
        base = 90
    elif pval < 0.01:
        base = 80
    elif pval < 0.05:
        base = 70
    elif pval < 0.1:
        base = 60
    else:
        base = 40

    magnitude = abs(np.log(odds_ratio))
    if magnitude > 1.0:
        base += 8
    elif magnitude > 0.5:
        base += 4
    elif magnitude < 0.1:
        base -= 5

    if diff_prob < 0:
        base = 100 - base

    base = max(0, min(100, int(round(base))))
    return base


def build_explanation(
    research_question: str, effect: dict, response_score: int
) -> str:
    lines = []
    lines.append(f"Research question: {research_question}")
    lines.append(
        "I modeled the proportion of antemortem tooth loss (num_amtl / sockets) "
        "using a binomial regression with a logit link, treating the number of sockets "
        "as the binomial denominator."
    )
    lines.append(
        "Predictors included an indicator for modern humans vs non-human primates "
        "(is_human), centered age, a continuous proxy for sex (prob_male), and "
        "tooth_class as a categorical covariate."
    )
    lines.append(
        f"The estimated log-odds coefficient for modern humans (is_human) was "
        f"{effect['coef']:.3f} (SE {effect['se']:.3f}, p = {effect['pval']:.3g}), "
        f"corresponding to an odds ratio of {effect['odds_ratio']:.2f} "
        f"(95% CI [{effect['or_low']:.2f}, {effect['or_high']:.2f}])."
    )
    direction = "higher" if effect["diff_prob"] > 0 else "lower"
    lines.append(
        f"For a typical individual (mean age and prob_male, most common tooth class), "
        f"the model predicts an AMTL proportion of {effect['human_prob']:.3f} for "
        f"modern humans and {effect['nonhuman_prob']:.3f} for non-human primates "
        f"(difference of {effect['diff_prob']:.3f}, humans {direction} on average)."
    )
    if effect["pval"] < 0.05 and effect["diff_prob"] > 0:
        qualitative = (
            "These results provide statistically significant evidence that modern humans "
            "have higher frequencies of AMTL than the non-human primate genera studied, "
            "even after controlling for age, sex, and tooth class."
        )
    elif effect["pval"] < 0.05 and effect["diff_prob"] < 0:
        qualitative = (
            "These results provide statistically significant evidence that modern humans "
            "actually have lower frequencies of AMTL than the non-human primate genera "
            "studied, after controlling for age, sex, and tooth class."
        )
    else:
        qualitative = (
            "The estimated human effect is not statistically reliable at conventional "
            "levels once age, sex, and tooth class are controlled, so the data do not "
            "provide clear evidence that humans differ from the non-human primates in "
            "AMTL frequency."
        )
    lines.append(qualitative)
    lines.append(
        f"On a 0–100 Likert scale, I summarize this as a response of {response_score}, "
        "where values near 0 represent a strong 'No' (no evidence that humans have "
        "higher AMTL) and values near 100 represent a strong 'Yes'."
    )
    return " ".join(lines)


def main() -> None:
    info = load_metadata(Path("info.json"))
    research_question = info["research_questions"][0]
    df_raw = load_data(Path("amtl.csv"))

    summarize_data(df_raw)

    df_model = prepare_model_data(df_raw)
    print("\nModeling data shape:", df_model.shape)

    result = fit_binomial_model(df_model)
    print(result.summary())

    effect = evaluate_human_effect(result, df_model)
    response_score = map_effect_to_likert(effect)
    explanation = build_explanation(research_question, effect, response_score)

    conclusion = {"response": int(response_score), "explanation": explanation}
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    print("\nWrote conclusion.txt with response:", response_score)


if __name__ == "__main__":
    main()
