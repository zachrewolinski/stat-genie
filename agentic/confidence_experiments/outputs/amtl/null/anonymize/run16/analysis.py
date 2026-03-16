import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_observable",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning: keep only rows with positive observable sockets
    df = df[df["n_observable"] > 0].copy()
    # Clip obvious numerical issues
    df["n_missing"] = df["n_missing"].clip(lower=0)
    df = df[df["n_missing"] <= df["n_observable"]]

    # Proportion of missing teeth (AMTL frequency)
    df["prop_missing"] = df["n_missing"] / df["n_observable"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression for AMTL frequency:
        prop_missing ~ is_human + age + sex_estimate + C(tooth_class)
    using n_observable as the binomial denominator via freq_weights.
    """
    formula = "prop_missing ~ is_human + age + sex_estimate + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_observable"],
    )
    result = model.fit()
    return result


def summarize_results(df: pd.DataFrame, result) -> dict:
    # Genus-level descriptive statistics
    genus_summary = (
        df.assign(
            prop_missing=lambda d: d["n_missing"] / d["n_observable"],
        )
        .groupby("genus")
        .agg(
            mean_prop_missing=("prop_missing", "mean"),
            sd_prop_missing=("prop_missing", "std"),
            total_missing=("n_missing", "sum"),
            total_observable=("n_observable", "sum"),
            n_specimens=("specimen_id", "nunique"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    # Extract human effect from the GLM
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues

    if "is_human" in params.index:
        coef = float(params["is_human"])
        ci_low, ci_high = map(float, conf_int.loc["is_human"])
        pval = float(pvalues["is_human"])
    else:
        coef = np.nan
        ci_low, ci_high = np.nan, np.nan
        pval = np.nan

    # Compute approximate odds ratio and its CI
    if np.isfinite(coef):
        odds_ratio = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))
    else:
        odds_ratio = np.nan
        or_ci_low, or_ci_high = np.nan, np.nan

    return {
        "genus_summary": genus_summary,
        "human_effect": {
            "log_odds_coef": coef,
            "log_odds_ci": [ci_low, ci_high],
            "p_value": pval,
            "odds_ratio": odds_ratio,
            "odds_ratio_ci": [or_ci_low, or_ci_high],
        },
        "model_deviance": float(result.deviance),
        "model_df_resid": int(result.df_resid),
    }


def derive_likert_response(human_effect: dict) -> int:
    """
    Map statistical evidence about the human effect onto a 0–100 Likert scale,
    where 0 = strong "No" (no higher frequency in humans),
    and 100 = strong "Yes".
    """
    coef = human_effect["log_odds_coef"]
    pval = human_effect["p_value"]
    odds_ratio = human_effect["odds_ratio"]

    if not np.isfinite(coef) or not np.isfinite(pval) or not np.isfinite(odds_ratio):
        return 50

    # Strong positive and highly significant effect
    if coef > 0 and pval < 0.001:
        if odds_ratio >= 2.0:
            return 95
        return 85

    # Moderately strong evidence
    if coef > 0 and pval < 0.01:
        return 75

    # Weak but positive evidence
    if coef > 0 and pval < 0.05:
        return 65

    # No clear difference
    if pval >= 0.05:
        return 40 if coef > 0 else 35

    # Fallback neutral
    return 50


def build_explanation(summary: dict, likert: int) -> str:
    human_effect = summary["human_effect"]
    genus_summary = summary["genus_summary"]

    # Find human and non-human summaries
    human_entry = next((g for g in genus_summary if g["genus"] == "Homo sapiens"), None)
    nonhuman_entries = [g for g in genus_summary if g["genus"] != "Homo sapiens"]

    lines = []
    lines.append(
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) "
        "after accounting for age, sex, and tooth class?"
    )

    # Descriptive comparison
    if human_entry is not None and nonhuman_entries:
        human_mean = human_entry["mean_prop_missing"]
        human_n = human_entry["n_specimens"]

        total_nonhuman_missing = sum(e["total_missing"] for e in nonhuman_entries)
        total_nonhuman_observable = sum(e["total_observable"] for e in nonhuman_entries)
        nonhuman_mean = total_nonhuman_missing / total_nonhuman_observable

        lines.append(
            f"Descriptively, humans show an average AMTL proportion of "
            f"{human_mean:.3f} across {human_n} specimens, while the pooled "
            f"non-human genera show an average AMTL proportion of "
            f"{nonhuman_mean:.3f}."
        )

    coef = human_effect["log_odds_coef"]
    ci_low, ci_high = human_effect["log_odds_ci"]
    pval = human_effect["p_value"]
    odds_ratio = human_effect["odds_ratio"]
    or_ci_low, or_ci_high = human_effect["odds_ratio_ci"]

    if np.isfinite(coef) and np.isfinite(pval) and np.isfinite(odds_ratio):
        direction = "higher" if coef > 0 else "lower"
        lines.append(
            "To formally test the hypothesis, I fit a binomial regression model "
            "for the proportion of missing teeth with predictors for a human "
            "indicator (Homo sapiens vs all non-human genera), age at death, "
            "estimated sex, and tooth class (anterior, premolar, posterior). "
            "The model uses the number of observable sockets as the binomial "
            "denominator so that each specimen contributes according to the number "
            "of teeth that could be scored."
        )
        lines.append(
            f"In this model, the coefficient for the human indicator corresponds "
            f"to a change in log-odds of AMTL of {coef:.3f} "
            f"(95% CI {ci_low:.3f} to {ci_high:.3f}, p = {pval:.3g}). "
            f"This translates to an odds ratio of {odds_ratio:.2f} "
            f"(95% CI {or_ci_low:.2f} to {or_ci_high:.2f}), meaning that, "
            f"after adjusting for age, sex, and tooth class, humans have "
            f"{direction} odds of AMTL than non-human primates."
        )
    else:
        lines.append(
            "The binomial regression model could not reliably estimate the effect "
            "of being human on AMTL frequency, so the evidence from the model "
            "is inconclusive."
        )

    # Map Likert score into qualitative interpretation
    if likert >= 85:
        strength = "very strong"
        answer = "Yes"
    elif likert >= 70:
        strength = "strong"
        answer = "Yes"
    elif likert >= 60:
        strength = "moderate"
        answer = "Yes"
    elif likert >= 40:
        strength = "weak"
        answer = "No"
    else:
        strength = "strong"
        answer = "No"

    lines.append(
        f"Overall, based on the direction and statistical significance of the "
        f"human effect, I answer the research question as '{answer}'. "
        f"The Likert-scale response of {likert} reflects a {strength} level "
        f"of evidence for this conclusion, where 0 is a strong 'No' and 100 is "
        f"a strong 'Yes'."
    )

    lines.append(
        "Caveats: The analysis assumes a simple binomial model with linear "
        "effects of age and sex and does not include hierarchical structure "
        "for different primate genera or regions. Unequal sample sizes across "
        "groups and potential unmeasured confounders could influence the "
        "estimated differences in AMTL frequencies."
    )

    return " ".join(lines)


def main():
    df = load_data(Path("amtl.csv"))
    result = fit_binomial_model(df)
    summary = summarize_results(df, result)
    likert = derive_likert_response(summary["human_effect"])
    explanation = build_explanation(summary, likert)

    conclusion = {"response": int(likert), "explanation": explanation}

    # Write conclusion to the required file
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

