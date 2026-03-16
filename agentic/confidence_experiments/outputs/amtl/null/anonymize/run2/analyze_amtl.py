import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Proportion of missing teeth (AMTL rate) per row
    df = df.copy()
    df["prop_missing"] = df["feature3"] / df["feature4"]
    # Drop any rows with invalid counts
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["prop_missing", "feature5", "feature7"])
    return df


def fit_binomial_glm(df: pd.DataFrame):
    # Binomial model with logit link on counts of missing vs present teeth
    # feature3 = missing teeth, feature4 = observable sockets
    # Predictors: genus (feature8), age (feature5), sex estimate (feature7), tooth class (feature1)
    df = df.copy()
    df["non_missing"] = df["feature4"] - df["feature3"]
    # Ensure counts are valid
    df = df[(df["feature3"] >= 0) & (df["non_missing"] >= 0)]

    # Use GLM with Binomial family and weights = total sockets
    # We model the log-odds of tooth loss as a function of predictors
    formula = "prop_missing ~ C(feature8) + feature5 + feature7 + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()
    return result


def summarize_human_effect(result) -> dict:
    # Reference category for genus will be the first in alphabetical order unless otherwise specified.
    # We construct contrasts for Homo sapiens vs each non-human genus from the fitted parameters.
    params = result.params
    bse = result.bse

    # Identify genus coefficients (C(feature8)[T.x])
    genus_terms = {name: (coef, bse[name]) for name, coef in params.items() if name.startswith("C(feature8)[T.")}

    # The baseline genus is the one omitted from the dummy set.
    # We infer which genus labels appear, then deduce the baseline from the metadata / observed categories.
    # For interpretation, we care whether coefficients for Homo sapiens are higher than non-human genera.

    # Since the reference level is baked into the intercept, we approximate by comparing Homo sapiens effect
    # relative to the average of non-human coefficients when available.
    homo_key = None
    for key in genus_terms:
        if "Homo sapiens" in key or "Homo" in key:
            homo_key = key
            break

    # If Homo is the reference, it will not appear in genus_terms; handle this separately.
    info = {
        "homo_vs_nonhuman_logit_diff": None,
        "homo_vs_nonhuman_se": None,
        "p_value_approx": None,
        "interpretation": "",
    }

    if homo_key is not None:
        homo_coef, homo_se = genus_terms[homo_key]
        nonhuman_coefs = [coef for key, (coef, _) in genus_terms.items() if key != homo_key]
        if nonhuman_coefs:
            mean_nonhuman = float(np.mean(nonhuman_coefs))
            diff = homo_coef - mean_nonhuman
            # Approximate SE by assuming independence
            nonhuman_ses = [bse[key] for key in genus_terms if key != homo_key]
            mean_nonhuman_se = float(np.sqrt(np.mean(np.square(nonhuman_ses))))
            se_diff = float(np.sqrt(homo_se ** 2 + mean_nonhuman_se ** 2))
            z = diff / se_diff if se_diff > 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            info.update(
                {
                    "homo_vs_nonhuman_logit_diff": float(diff),
                    "homo_vs_nonhuman_se": float(se_diff),
                    "p_value_approx": float(p) if not np.isnan(p) else None,
                }
            )
    else:
        # Homo sapiens is likely the reference level (absorbed in intercept).
        # Compare each non-human genus coefficient to 0 (Homo baseline).
        diffs = []
        ses = []
        zs = []
        ps = []
        for key, (coef, se) in genus_terms.items():
            diffs.append(-coef)  # Homo - genus = 0 - coef
            ses.append(se)
            z = (-coef) / se if se > 0 else np.nan
            zs.append(z)
            p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ps.append(p)

        if diffs:
            mean_diff = float(np.mean(diffs))
            mean_se = float(np.sqrt(np.mean(np.square(ses))))
            # For conservatism, take max p-value among comparisons
            max_p = float(np.nanmax(ps)) if any(not np.isnan(p) for p in ps) else None
            info.update(
                {
                    "homo_vs_nonhuman_logit_diff": mean_diff,
                    "homo_vs_nonhuman_se": mean_se,
                    "p_value_approx": max_p,
                }
            )

    # High-level verbal interpretation stub; detailed interpretation will be built outside.
    if info["homo_vs_nonhuman_logit_diff"] is not None:
        if info["homo_vs_nonhuman_logit_diff"] > 0:
            direction = "higher"
        else:
            direction = "lower"
        info["interpretation"] = f"Model suggests Homo sapiens have {direction} AMTL odds than non-human genera on average."

    return info


def map_to_likert(effect_info: dict) -> int:
    diff = effect_info.get("homo_vs_nonhuman_logit_diff")
    p = effect_info.get("p_value_approx")

    if diff is None or p is None:
        # Inconclusive evidence; lean slightly toward no strong difference
        return 45

    # Use p-value and effect size to map to Likert score
    abs_diff = abs(diff)

    if p < 0.001 and abs_diff >= 0.5:
        base = 95
    elif p < 0.01 and abs_diff >= 0.4:
        base = 85
    elif p < 0.05 and abs_diff >= 0.3:
        base = 75
    elif p < 0.05 and abs_diff >= 0.15:
        base = 65
    elif p < 0.1 and abs_diff >= 0.15:
        base = 60
    elif p < 0.1:
        base = 55
    else:
        base = 40

    if diff < 0:
        # Evidence goes against humans having higher AMTL
        score = 100 - base
    else:
        score = base

    score = int(max(0, min(100, round(score))))
    return score


def build_explanation(metadata: dict, result, effect_info: dict, likert_score: int) -> str:
    research_q = metadata["research_questions"][0]
    summary = result.summary2().as_text()

    parts = []
    parts.append("Research question: " + research_q)
    parts.append(
        "Modeling approach: Fit a binomial generalized linear model (logit link) to the proportion of antemortem tooth loss (missing teeth / observable sockets) per specimen, with genus (Homo, Pan, Papio, Pongo) as the primary predictor and age at death, sex estimate, and tooth class as covariates, using socket counts as frequency weights."
    )

    diff = effect_info.get("homo_vs_nonhuman_logit_diff")
    p = effect_info.get("p_value_approx")
    if diff is not None and p is not None:
        direction = "higher" if diff > 0 else "lower"
        parts.append(
            f"Human vs non-human effect: The average log-odds difference for Homo sapiens relative to non-human genera is {diff:.3f} with an approximate p-value of {p:.3g}, indicating that humans have {direction} AMTL odds after adjusting for age, sex, and tooth class."
        )
    else:
        parts.append(
            "Human vs non-human effect: The model could not yield a stable, fully interpretable contrast for Homo sapiens versus non-human genera, so evidence for a difference is weak or inconclusive."
        )

    if likert_score >= 60:
        parts.append(
            f"Conclusion (Likert {likert_score}/100): Yes, the data provide statistical evidence that modern humans have higher frequencies of antemortem tooth loss than non-human primates once age, sex, and tooth class are accounted for, with the Likert score reflecting both the effect size and strength of the evidence."
        )
    elif likert_score <= 40:
        parts.append(
            f"Conclusion (Likert {likert_score}/100): No, the data do not support higher AMTL frequencies in modern humans compared to non-human primates once age, sex, and tooth class are controlled; the Likert score reflects that the estimated differences are small, inconsistent, or not statistically robust."
        )
    else:
        parts.append(
            f"Conclusion (Likert {likert_score}/100): The evidence for higher AMTL frequencies in modern humans versus non-human primates is mixed or marginal; while some model terms suggest a difference, the overall uncertainty keeps the conclusion closer to equivocal than to a strong yes or no."
        )

    parts.append(
        "Key model output (abbreviated):\n" + "\n".join(summary.splitlines()[:25])
    )

    return "\n\n".join(parts)


def main():
    base = Path(".")
    metadata = load_metadata(base / "info.json")
    df = pd.read_csv(base / "amtl.csv")

    df_prep = prepare_data(df)
    result = fit_binomial_glm(df_prep)
    effect_info = summarize_human_effect(result)
    likert_score = map_to_likert(effect_info)
    explanation = build_explanation(metadata, result, effect_info, likert_score)

    conclusion = {"response": int(likert_score), "explanation": explanation}

    with (base / "conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
