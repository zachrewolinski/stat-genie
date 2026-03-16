import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Total teeth originally present in this class (missing + observable sockets)
    df["total_teeth"] = df["num_amtl"] + df["sockets"]
    return df


def expand_to_tooth_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the specimen-level counts (num_amtl, sockets) into per-tooth rows.

    Each row is repeated total_teeth times, with `amtl` indicating whether that
    tooth was lost antemortem (1) or present (0).
    """
    # Repeat each row total_teeth times
    df_rep = df.loc[df.index.repeat(df["total_teeth"])].copy()
    df_rep.reset_index(drop=True, inplace=True)

    # Build corresponding AMTL indicator: first num_amtl ones, then sockets zeros
    amtl_flags = []
    for _, row in df.iterrows():
        num_amtl = int(row["num_amtl"])
        sockets = int(row["sockets"])
        amtl_flags.extend([1] * num_amtl + [0] * sockets)

    if len(amtl_flags) != len(df_rep):
        raise ValueError("Length mismatch when expanding to tooth-level data.")

    df_rep["amtl"] = amtl_flags

    # Indicator for modern human vs non-human primate genera
    df_rep["is_human"] = (df_rep["genus"] == "Homo sapiens").astype(int)

    return df_rep


def fit_model(df_teeth: pd.DataFrame):
    """
    Fit a logistic regression model for AMTL at the tooth level.

    Outcome: amtl (1 = missing tooth, 0 = present)
    Predictor of interest: is_human (1 = Homo sapiens, 0 = Pan/Papio/Pongo)
    Covariates: age, prob_male, tooth_class
    """
    formula = "amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.logit(formula=formula, data=df_teeth)
    result = model.fit(disp=False)
    return result


def summarize_human_effect(result):
    coef = result.params["is_human"]
    pval = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = result.conf_int().loc["is_human"]
    ci_low = float(np.exp(ci_low))
    ci_high = float(np.exp(ci_high))
    return {
        "coef": float(coef),
        "pval": pval,
        "odds_ratio": odds_ratio,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def map_to_likert(effect_summary: dict) -> int:
    """
    Map the direction and strength of evidence about `is_human` to a 0-100 scale.

    0  -> very strong evidence that humans have *lower* AMTL than non-humans
    50 -> no clear evidence either way
    100 -> very strong evidence that humans have *higher* AMTL
    """
    coef = effect_summary["coef"]
    pval = effect_summary["pval"]
    oratio = effect_summary["odds_ratio"]

    # Strong evidence humans have higher AMTL
    if coef > 0 and pval < 0.01:
        if oratio > 1.5:
            return 90
        return 75

    # Moderate/weak evidence humans have higher AMTL
    if coef > 0 and pval < 0.05:
        return 65
    if coef > 0 and pval < 0.1:
        return 60

    # Near null effect or clearly non-significant
    if abs(coef) < 0.05 or pval >= 0.2:
        return 50

    # Evidence humans have *lower* AMTL
    if coef < 0 and pval < 0.01:
        return 10
    if coef < 0 and pval < 0.05:
        return 20
    if coef < 0 and pval < 0.1:
        return 30

    # Fallback (should rarely be hit)
    return 50


def build_explanation(effect_summary: dict, response: int) -> str:
    direction = (
        "higher" if effect_summary["coef"] > 0 else "lower"
        if effect_summary["coef"] < 0
        else "similar"
    )

    explanation = (
        "I analyzed the AMTL dataset by expanding each specimen and tooth class "
        "record into per-tooth data, treating each potential tooth as either present "
        "or lost antemortem. I then fit a logistic (binomial) regression model for "
        "the probability that an individual tooth was lost, with a binary indicator "
        "for modern humans (Homo sapiens vs. Pan/Papio/Pongo) as the main predictor, "
        "while adjusting for age at death, estimated sex (prob_male), and tooth "
        "class (anterior, posterior, premolar).\n\n"
        f"In this model, the coefficient for the human indicator corresponds to an "
        f"odds ratio of approximately {effect_summary['odds_ratio']:.2f} "
        f"(95% CI {effect_summary['ci_low']:.2f}–{effect_summary['ci_high']:.2f}, "
        f"p-value {effect_summary['pval']:.3g}), indicating that—after accounting for "
        f"age, sex, and tooth class—modern humans have {direction} odds of "
        "antemortem tooth loss compared to the pooled non-human primates. "
    )

    if response < 50:
        conclusion = (
            "Because the estimated effect is negative or at most very small and "
            "statistically inconsistent with substantially higher odds in humans, "
            "I conclude that the data do not support the claim that modern humans "
            "have higher frequencies of AMTL than non-human primates once age, sex, "
            "and tooth class are controlled."
        )
    elif response > 50:
        conclusion = (
            "Because the estimated effect is positive and the odds ratio is "
            "statistically above one, there is evidence that modern humans have "
            "higher frequencies of AMTL than non-human primates after controlling "
            "for age, sex, and tooth class."
        )
    else:
        conclusion = (
            "Because the estimated effect is close to null and statistically "
            "uncertain, the data provide no clear evidence that modern humans have "
            "different AMTL frequencies than non-human primates after controlling "
            "for age, sex, and tooth class."
        )

    return explanation + conclusion


def main() -> None:
    base_dir = Path(__file__).parent
    csv_path = base_dir / "amtl.csv"

    df = load_data(csv_path)
    df_teeth = expand_to_tooth_level(df)

    result = fit_model(df_teeth)
    effect_summary = summarize_human_effect(result)
    response = map_to_likert(effect_summary)

    explanation = build_explanation(effect_summary, response)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

