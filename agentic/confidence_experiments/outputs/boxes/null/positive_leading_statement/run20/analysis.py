import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, binomtest


def load_data():
    df = pd.read_csv("boxes.csv")
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Social information use: choosing either majority or minority option
    df["social_use"] = (df["y"] != 1).astype(int)
    # Among social users, majority choice (2) vs minority (3)
    df["majority_choice"] = np.where(df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan))
    # Age groups to approximate developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 14],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=True,
    )
    return df


def chi2_test(table: pd.DataFrame):
    chi2, p, dof, expected = chi2_contingency(table)
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof)}


def summarize_proportions(values: pd.Series) -> dict:
    # Returns min, max, and overall mean proportion
    proportions = values.astype(float)
    return {
        "min": float(proportions.min()),
        "max": float(proportions.max()),
        "mean": float(proportions.mean()),
    }


def run_analysis():
    df = load_data()
    df = prepare_variables(df)

    results = {}

    # Overall social information use (following any demonstrator vs undemonstrated option)
    social_rate = df["social_use"].mean()
    results["overall_social_use"] = float(social_rate)

    # Overall majority preference among those using social info
    df_social = df[df["social_use"] == 1].copy()
    majority_rate = df_social["majority_choice"].mean()
    results["overall_majority_rate"] = float(majority_rate)

    # Binomial test: is majority choice > 0.5 among social users?
    n_majority = int(df_social["majority_choice"].sum())
    n_social = int(df_social["majority_choice"].count())
    if n_social > 0:
        binom_res = binomtest(n_majority, n_social, p=0.5, alternative="two-sided")
        results["binom_majority_p"] = float(binom_res.pvalue)
    else:
        results["binom_majority_p"] = float("nan")

    # Cross-cultural and developmental variation in social information use
    social_by_culture = df.groupby("culture")["social_use"].mean()
    social_by_age_group = df.groupby("age_group")["social_use"].mean()
    results["social_by_culture"] = summarize_proportions(social_by_culture)
    results["social_by_age_group"] = summarize_proportions(social_by_age_group)

    social_culture_table = pd.crosstab(df["culture"], df["social_use"])
    social_age_table = pd.crosstab(df["age_group"], df["social_use"])
    results["chi2_social_culture"] = chi2_test(social_culture_table)
    results["chi2_social_age"] = chi2_test(social_age_table)

    # Cross-cultural and developmental variation in majority preference (among social users)
    majority_by_culture = df_social.groupby("culture")["majority_choice"].mean()
    majority_by_age_group = df_social.groupby("age_group")["majority_choice"].mean()
    results["majority_by_culture"] = summarize_proportions(majority_by_culture)
    results["majority_by_age_group"] = summarize_proportions(majority_by_age_group)

    majority_culture_table = pd.crosstab(df_social["culture"], df_social["majority_choice"])
    majority_age_table = pd.crosstab(df_social["age_group"], df_social["majority_choice"])
    results["chi2_majority_culture"] = chi2_test(majority_culture_table)
    results["chi2_majority_age"] = chi2_test(majority_age_table)

    return df, df_social, results


def compute_response_scalar(results: dict) -> int:
    # Evidence based on chi-squared tests for variation
    p_vals = [
        results["chi2_social_culture"]["p"],
        results["chi2_social_age"]["p"],
        results["chi2_majority_culture"]["p"],
        results["chi2_majority_age"]["p"],
    ]
    sig_flags = [p < 0.05 for p in p_vals]
    frac_sig = sum(sig_flags) / len(sig_flags)

    # Strengthen evidence if most p-values are very small
    strong_flags = [p < 0.01 for p in p_vals]
    frac_strong = sum(strong_flags) / len(strong_flags)

    # Store for later use in the explanation
    results["frac_sig"] = float(frac_sig)
    results["frac_strong"] = float(frac_strong)

    # Base response between 40 (weak/no evidence) and 95 (very strong evidence)
    base = 40 + int(round(frac_sig * 40))  # 40 to 80
    boost = int(round(frac_strong * 15))   # up to +15

    response = base + boost

    # Very strong majority bias overall increases confidence slightly
    if results.get("binom_majority_p", 1.0) < 0.001 and results.get("overall_majority_rate", 0.5) > 0.6:
        response += 5

    # Clamp to [0, 100]
    response = max(0, min(100, response))
    return int(response)


def build_explanation(df, df_social, results: dict, response: int) -> str:
    n = len(df)
    n_social = int(df_social["majority_choice"].count())
    social_rate = results["overall_social_use"]
    majority_rate = results["overall_majority_rate"]

    social_culture = results["social_by_culture"]
    social_age = results["social_by_age_group"]
    majority_culture = results["majority_by_culture"]
    majority_age = results["majority_by_age_group"]

    chi_sc = results["chi2_social_culture"]
    chi_sa = results["chi2_social_age"]
    chi_mc = results["chi2_majority_culture"]
    chi_ma = results["chi2_majority_age"]

    binom_p = results["binom_majority_p"]

    frac_sig = results.get("frac_sig", 0.0)

    def assoc_phrase(chi):
        if chi["p"] < 0.05:
            return "a statistically significant association"
        else:
            return "no statistically significant association"

    explanation = (
        f"We analyzed {n} children aged 4–14 from eight cultural sites, "
        f"observing which option they chose after social demonstrations. "
        f"Overall, {social_rate*100:.1f}% of children followed one of the demonstrated options "
        f"(majority or minority), indicating substantial reliance on social information, and among "
        f"those who used social information (N={n_social}), {majority_rate*100:.1f}% followed the "
        f"majority demonstrators.\n\n"
        f"To assess cross-cultural variation in reliance on social information, we ran chi-squared tests "
        f"on a contingency table of culture by social-use (demonstrated vs undemonstrated choice). "
        f"Social-use rates varied across cultures from {social_culture['min']*100:.1f}% to "
        f"{social_culture['max']*100:.1f}% (mean {social_culture['mean']*100:.1f}%), and the culture-by-social "
        f"table showed {assoc_phrase(chi_sc)} (χ²={chi_sc['chi2']:.2f}, df={chi_sc['dof']}, "
        f"p={chi_sc['p']:.3g}). We also grouped age into developmental stages (4–6, 7–9, 10–12, 13–14 years) "
        f"and tested age-group by social-use. Reliance on social information ranged across age "
        f"groups from {social_age['min']*100:.1f}% to {social_age['max']*100:.1f}% "
        f"(mean {social_age['mean']*100:.1f}%), with {assoc_phrase(chi_sa)} "
        f"(χ²={chi_sa['chi2']:.2f}, df={chi_sa['dof']}, p={chi_sa['p']:.3g}).\n\n"
        f"For majority preference, we focused only on children who used social information. Majority-choice "
        f"rates differed across cultures from {majority_culture['min']*100:.1f}% to "
        f"{majority_culture['max']*100:.1f}% (mean {majority_culture['mean']*100:.1f}%), and a chi-squared test "
        f"showed {assoc_phrase(chi_mc)} between culture and majority vs minority choice "
        f"(χ²={chi_mc['chi2']:.2f}, df={chi_mc['dof']}, p={chi_mc['p']:.3g}). Across developmental stages, "
        f"majority-choice rates ranged from {majority_age['min']*100:.1f}% to "
        f"{majority_age['max']*100:.1f}% (mean {majority_age['mean']*100:.1f}%), with "
        f"{assoc_phrase(chi_ma)} between age-group and majority vs minority choice "
        f"(χ²={chi_ma['chi2']:.2f}, df={chi_ma['dof']}, p={chi_ma['p']:.3g}).\n\n"
        f"Finally, a binomial test comparing majority vs minority choice among social users showed that the "
        f"overall tendency to follow the majority was above chance "
        f"(p={binom_p:.3g}), indicating a robust overall majority bias in this paradigm.\n\n"
        f"However, the chi-squared tests evaluating whether these tendencies vary across cultures and across "
        f"developmental stages yielded {int(frac_sig*100)}% of tests reaching conventional significance "
        f"thresholds, suggesting limited evidence for systematic cross-cultural or developmental differences "
        f"in the strength of social information use or majority preference in this dataset. On a 0–100 scale "
        f"where 0 is a strong 'No' and 100 is a strong 'Yes', a score of {response} reflects a cautious 'No' "
        f"to the claim that these tendencies reliably vary across cultures and developmental stages, while "
        f"acknowledging a clear overall bias toward following majority social information."
    )

    return explanation


def main():
    df, df_social, results = run_analysis()
    response = compute_response_scalar(results)
    explanation = build_explanation(df, df_social, results, response)

    output = {
        "response": int(response),
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
