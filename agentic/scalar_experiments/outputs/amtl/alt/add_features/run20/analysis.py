import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity filtering for the binomial model
    df = df.copy()
    # Keep only rows with valid counts and required covariates
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] <= df["sockets"]]
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])
    # Focus on the four genera relevant to the question, in case other rows appear
    genera_of_interest = {"Homo sapiens", "Pan", "Pongo", "Papio"}
    df = df[df["genus"].isin(genera_of_interest)]
    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    # Ensure categorical coding for tooth class
    df["tooth_class"] = df["tooth_class"].astype("category")
    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression using successes and failures counts
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    design = dmatrix(
        "is_human + C(tooth_class) + age + prob_male",
        df,
        return_type="dataframe",
    )

    model = sm.GLM(endog, design, family=sm.families.Binomial())
    result = model.fit()
    return result


def summarize_human_effect(result) -> dict:
    # Extract coefficient and p-value for the human indicator
    coef = result.params.get("is_human", float("nan"))
    se = result.bse.get("is_human", float("nan"))
    pval = result.pvalues.get("is_human", float("nan"))
    # Compute odds ratio and 95% CI when possible
    if pd.notnull(coef) and pd.notnull(se):
        or_est = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
    else:
        or_est = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "coef": float(coef),
        "pval": float(pval),
        "odds_ratio": or_est,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def map_to_likert(human_effect: dict) -> int:
    coef = human_effect["coef"]
    pval = human_effect["pval"]

    # If estimate or p-value is missing, return neutral
    if not pd.notnull(coef) or not pd.notnull(pval):
        return 50

    # Determine direction and strength based on p-value and effect size
    if pval >= 0.10:
        # Little evidence either way
        return 50

    # Evidence that humans differ from non-human primates
    if coef > 0:
        # Humans have higher AMTL frequencies
        if pval < 1e-6:
            return 95
        if pval < 1e-3:
            return 85
        if pval < 0.01:
            return 75
        # 0.01 <= p < 0.10, modest evidence
        return 65
    else:
        # Humans have lower AMTL frequencies
        if pval < 1e-6:
            return 5
        if pval < 1e-3:
            return 15
        if pval < 0.01:
            return 25
        return 35


def build_explanation(metadata: dict, human_effect: dict, response_score: int) -> str:
    question = metadata["research_questions"][0]
    coef = human_effect["coef"]
    pval = human_effect["pval"]
    or_est = human_effect["odds_ratio"]
    ci_low = human_effect["ci_low"]
    ci_high = human_effect["ci_high"]

    direction = (
        "higher"
        if pd.notnull(coef) and coef > 0
        else "lower"
        if pd.notnull(coef) and coef < 0
        else "no clear difference in"
    )

    yes_no = (
        "Yes"
        if response_score > 50
        else "No"
        if response_score < 50
        else "Uncertain"
    )

    explanation = (
        f"Research question: {question}\n"
        f"Answer on 0–100 scale: {response_score} ({yes_no}).\n"
        "I fit a binomial regression model with the proportion of antemortem tooth loss "
        "(num_amtl / sockets) as the outcome, and included an indicator for modern humans "
        "vs. non-human primates (Pan, Pongo, Papio), tooth class (anterior, posterior, premolar), "
        "age at death, and probability of being male as predictors, weighting by the number of "
        "observable tooth sockets per specimen and tooth class.\n"
    )

    if pd.notnull(coef) and pd.notnull(pval):
        explanation += (
            f"The coefficient for the human indicator was {coef:.3f}, corresponding to an odds ratio "
            f"of approximately {or_est:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f}), with p-value {pval:.2e}. "
        )

        if response_score > 50:
            explanation += (
                "This positive and statistically significant effect indicates that, after adjusting "
                "for age, sex, and tooth class, modern humans have "
                f"{direction} odds of AMTL than the non-human primate genera considered. "
            )
        elif response_score < 50:
            explanation += (
                "This negative and statistically significant effect indicates that, after adjusting "
                "for age, sex, and tooth class, modern humans have "
                f"{direction} odds of AMTL than the non-human primate genera considered. "
            )
        else:
            explanation += (
                "Because the effect is small and/or not statistically significant at conventional "
                "levels, the data do not provide clear evidence that modern humans differ in AMTL "
                "frequency from these non-human primates once age, sex, and tooth class are taken "
                "into account. "
            )
    else:
        explanation += (
            "The model could not reliably estimate the effect of modern humans versus non-human "
            "primates on AMTL frequencies, leading to an essentially uncertain conclusion. "
        )

    explanation += (
        "The Likert-style score reflects both the statistical significance (p-value) and the size "
        "of the human effect estimate: values closer to 0 represent strong evidence against humans "
        "having higher AMTL frequencies, values near 50 represent ambiguous or null evidence, and "
        "values closer to 100 represent strong evidence that humans have higher AMTL frequencies "
        "than non-human primates after adjusting for the covariates."
    )

    return explanation


def main():
    base = Path(__file__).parent
    metadata = load_metadata(base / "info.json")
    df = load_data(base / "amtl.csv")

    result = fit_model(df)
    human_effect = summarize_human_effect(result)
    response_score = map_to_likert(human_effect)

    explanation = build_explanation(metadata, human_effect, response_score)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt
    conclusion_path = base / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
