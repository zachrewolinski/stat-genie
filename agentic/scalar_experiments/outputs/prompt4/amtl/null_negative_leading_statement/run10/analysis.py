import json
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment  # noqa: F401  - used in formula evaluation


def fit_model(df: pd.DataFrame):
    """
    Fit a binomial GLM for AMTL proportion with genus (humans as reference),
    age, sex estimate, and tooth class as predictors.
    """
    # Keep only rows with valid counts
    valid = (df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])
    df = df.loc[valid].copy()

    # Proportion of missing teeth per tooth class
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Genus as categorical with Homo sapiens as reference; tooth_class as categorical
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result, df


def extract_genus_effects(result) -> Dict[str, Dict[str, float]]:
    """
    Extract coefficients, standard errors, and p-values for non-human genera
    relative to Homo sapiens (the reference group).
    """
    prefix = "C(genus, Treatment(reference='Homo sapiens'))"
    genus_effects: Dict[str, Dict[str, float]] = {}
    for name in result.params.index:
        if not name.startswith(prefix):
            continue
        # Name looks like: C(genus, Treatment(reference='Homo sapiens'))[T.Pan]
        genus_label = name.split("[T.")[-1].rstrip("]")
        coef = float(result.params[name])
        se = float(result.bse[name])
        pval = float(result.pvalues[name])
        z = coef / se if se != 0 else np.nan
        genus_effects[genus_label] = {
            "coef": coef,
            "se": se,
            "z": z,
            "p": pval,
        }
    return genus_effects


def compute_support_from_effects(genus_effects: Dict[str, Dict[str, float]]) -> float:
    """
    Convert the set of genus effects into a scalar support in [0, 1]
    that humans have higher AMTL than non-human genera.

    Each coefficient is (genus - Homo) on the log-odds scale, so
    negative values mean the non-human genus has *lower* AMTL than humans.
    We work with z-scores and flip the sign so that positive values favor humans.
    """
    z_scores: List[float] = []
    for eff in genus_effects.values():
        z = eff["z"]
        if not np.isfinite(z):
            continue
        # Flip sign so positive values mean humans > non-human genus
        z_scores.append(-z)

    if not z_scores:
        return 0.5  # neutral if we cannot estimate anything

    mean_z = float(np.mean(z_scores))
    # Map average z-score through a logistic transform to get a support probability
    support = 1.0 / (1.0 + np.exp(-mean_z))
    return float(support)


def predicted_probs_by_genus(result, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute predicted AMTL probabilities for each genus at typical covariate values:
    mean age, mean sex estimate, and the modal tooth class.
    """
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())

    tooth_mode_series = df["tooth_class"].mode()
    if not tooth_mode_series.empty:
        tooth_mode = tooth_mode_series.iloc[0]
    else:
        tooth_mode = df["tooth_class"].iloc[0]

    genera = sorted(df["genus"].unique())

    new_data = pd.DataFrame(
        {
            "genus": genera,
            "age": [mean_age] * len(genera),
            "prob_male": [mean_prob_male] * len(genera),
            "tooth_class": [tooth_mode] * len(genera),
        }
    )

    pred = result.get_prediction(new_data)
    pred_summary = pred.summary_frame(alpha=0.05)

    new_data = new_data.assign(
        pred_prob=pred_summary["mean"],
        pred_lower=pred_summary["mean_ci_lower"],
        pred_upper=pred_summary["mean_ci_upper"],
    )
    return new_data


def build_explanation(
    genus_effects: Dict[str, Dict[str, float]],
    pred_table: pd.DataFrame,
    support: float,
    score: int,
) -> str:
    """
    Create a textual explanation summarizing the model, key coefficients, and conclusion.
    """
    lines: List[str] = []
    lines.append(
        "I fit a binomial regression with logit link for the proportion of missing teeth "
        "(num_amtl / sockets) using genus (with Homo sapiens as the reference), age at death, "
        "estimated sex (prob_male), and tooth class as predictors, weighting each observation "
        "by the number of observable sockets."
    )

    if genus_effects:
        for genus, eff in sorted(genus_effects.items()):
            coef = eff["coef"]
            se = eff["se"]
            z = eff["z"]
            p = eff["p"]
            lines.append(
                f" For genus {genus} relative to Homo sapiens, the coefficient on the log-odds scale "
                f"was {coef:.3f} (SE {se:.3f}, z = {z:.2f}, p = {p:.3g}), where negative values mean "
                "lower AMTL frequencies than in humans after adjusting for age, sex, and tooth class."
            )

    if not pred_table.empty:
        # Use the same typical covariate values for description
        typical_age = pred_table["age"].iloc[0]
        typical_tooth_class = pred_table["tooth_class"].iloc[0]
        lines.append(
            f" I then compared predicted AMTL probabilities by genus at a typical profile "
            f"(age ≈ {typical_age:.1f} years, mean sex estimate, tooth class {typical_tooth_class})."
        )
        for _, row in pred_table.iterrows():
            lines.append(
                f" The predicted AMTL proportion for {row['genus']} was "
                f"{row['pred_prob']:.3f} (95% CI {row['pred_lower']:.3f}–{row['pred_upper']:.3f})."
            )

    if support > 0.5:
        qualitative = (
            "These results provide overall evidence that, after accounting for age, sex, and tooth "
            "class, modern humans tend to have higher AMTL frequencies than the non-human primate "
            "genera in this sample."
        )
    elif support < 0.5:
        qualitative = (
            "These results provide overall evidence against the claim that, after accounting for "
            "age, sex, and tooth class, modern humans have higher AMTL frequencies than the "
            "non-human primate genera in this sample."
        )
    else:
        qualitative = (
            "Overall, the results are effectively neutral regarding whether modern humans have higher "
            "AMTL frequencies than the non-human primate genera after accounting for age, sex, "
            "and tooth class."
        )

    lines.append(
        f" I summarized the human-versus-non-human genus effects by averaging the z-scores for the "
        f"differences (with positive values favoring higher AMTL in humans) and mapping that through "
        f"a logistic transform to obtain a support value of {support:.2f} for the statement that "
        f"humans have higher AMTL frequencies. This support value corresponds to a Likert-scale "
        f"response of {score} on a 0–100 scale, where larger values indicate stronger support for "
        f"the statement."
    )
    lines.append(f" {qualitative}")

    explanation = " ".join(lines)
    return explanation


def main() -> None:
    df = pd.read_csv("amtl.csv")

    result, cleaned_df = fit_model(df)
    genus_effects = extract_genus_effects(result)
    support = compute_support_from_effects(genus_effects)
    score = int(round(100 * support))
    score = max(0, min(100, score))

    pred_table = predicted_probs_by_genus(result, cleaned_df)
    explanation = build_explanation(genus_effects, pred_table, support, score)

    conclusion = {
        "response": score,
        "explanation": explanation,
    }

    # Write the required JSON-only conclusion file
    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)

    # Also print a human-readable version to stdout for inspection
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()

