import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks: drop rows with missing key fields
    key_cols = ["num_amtl", "sockets", "genus", "tooth_class", "age", "prob_male"]
    df = df.dropna(subset=key_cols)
    # Ensure counts are valid
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]
    return df


def fit_binomial_model(df: pd.DataFrame):
    # Proportion of missing teeth as response with sockets as frequency weights
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Explicitly set Homo sapiens as the reference genus
    # Age and prob_male enter as linear covariates; tooth_class as categorical
    formula = (
        "amtl_prop ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + C(tooth_class) + age + prob_male"
    )
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_genus_effects(result) -> dict:
    """
    Return a summary of how each non-human genus compares to Homo sapiens.
    Parameters are on the log-odds scale; negative values mean *lower* AMTL
    frequency than Homo sapiens (since Homo sapiens is the reference).
    """
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues

    genus_effects = {}
    for genus in ["Pan", "Pongo", "Papio"]:
        term = f"C(genus, Treatment(reference='Homo sapiens'))[T.{genus}]"
        if term not in params.index:
            continue
        coef = float(params[term])
        ci_low, ci_high = conf_int.loc[term].astype(float)
        pval = float(pvalues[term])

        # If coef < 0, then non-human genus has lower AMTL than humans.
        homo_higher = coef < 0
        # Statistically significant at conventional 0.05 level
        significant = pval < 0.05 and not (ci_low <= 0.0 <= ci_high)

        genus_effects[genus] = {
            "log_odds_diff_vs_human": coef,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": pval,
            "human_higher": homo_higher,
            "significant": significant,
        }

    return genus_effects


def compute_descriptive_rates(df: pd.DataFrame) -> dict:
    """Compute simple observed AMTL rates by genus as a descriptive check."""
    rates = {}
    for genus, sub in df.groupby("genus"):
        total_missing = sub["num_amtl"].sum()
        total_sockets = sub["sockets"].sum()
        if total_sockets > 0:
            rate = total_missing / total_sockets
        else:
            rate = np.nan
        rates[genus] = {
            "total_missing": int(total_missing),
            "total_sockets": int(total_sockets),
            "amtl_rate": float(rate),
        }
    return rates


def map_to_likert(genus_effects: dict) -> int:
    """
    Map the pattern of results to a 0–100 Likert score where
    0 = strong 'No', 100 = strong 'Yes' (humans clearly higher).
    """
    # Count in how many pairwise comparisons humans are significantly higher.
    n_genus = 0
    n_sig_higher = 0
    n_higher_any = 0
    for genus, info in genus_effects.items():
        n_genus += 1
        if info["human_higher"]:
            n_higher_any += 1
            if info["significant"]:
                n_sig_higher += 1

    if n_genus == 0:
        # Fallback if something is odd with the data/model
        return 50

    # Strong yes: humans significantly higher than all non-human genera
    if n_sig_higher == n_genus:
        return 90
    # Mixed but generally higher: at least one significant and all effects positive for humans
    if n_sig_higher >= 1 and n_higher_any == n_genus:
        return 75
    # Some evidence but weak / inconsistent
    if n_higher_any >= 1:
        return 60
    # Essentially no evidence that humans are higher
    return 20


def build_explanation(
    genus_effects: dict, descriptive_rates: dict, likert_score: int
) -> str:
    lines = []
    lines.append(
        "I fit a binomial regression model for AMTL frequency "
        "using the proportion of missing teeth (num_amtl / sockets) "
        "as the outcome and including genus, tooth class, age, and sex "
        "(proxied by prob_male) as predictors, with Homo sapiens as the "
        "reference genus."
    )

    # Add model-based genus comparisons
    human_higher_all = True
    for genus in ["Pan", "Pongo", "Papio"]:
        info = genus_effects.get(genus)
        if info is None:
            continue
        coef = info["log_odds_diff_vs_human"]
        ci_low = info["ci_low"]
        ci_high = info["ci_high"]
        pval = info["p_value"]
        direction = "lower" if info["human_higher"] else "higher"
        human_higher_all = human_higher_all and info["human_higher"]
        lines.append(
            f"Compared with Homo sapiens, {genus} shows {direction} AMTL log-odds "
            f"(coefficient = {coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
            f"p = {pval:.3f})."
        )

    # Add descriptive rates
    if "Homo sapiens" in descriptive_rates:
        human_rate = descriptive_rates["Homo sapiens"]["amtl_rate"]
        lines.append(
            f"Observed AMTL rates (num_amtl / sockets) are approximately "
            f"{human_rate:.3f} for Homo sapiens and:"
        )
        for genus in ["Pan", "Pongo", "Papio"]:
            if genus in descriptive_rates:
                rate = descriptive_rates[genus]["amtl_rate"]
                lines.append(f"- {genus}: {rate:.3f}")

    if human_higher_all:
        qualitative = "These patterns indicate that humans have higher AMTL frequencies than all sampled non-human genera after adjusting for age, sex, and tooth class."
    else:
        qualitative = "These patterns provide only mixed evidence that humans have higher AMTL frequencies than all sampled non-human genera after adjusting for age, sex, and tooth class."

    lines.append(qualitative)
    lines.append(
        f"On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"I would place the answer at about {likert_score}, reflecting the overall "
        f"strength and consistency of the model-based and descriptive evidence."
    )

    return " ".join(lines)


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    result = fit_binomial_model(df)
    genus_effects = summarize_genus_effects(result)
    descriptive_rates = compute_descriptive_rates(df)
    likert_score = map_to_likert(genus_effects)
    explanation = build_explanation(genus_effects, descriptive_rates, likert_score)

    conclusion = {
        "response": int(likert_score),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

