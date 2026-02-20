import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Rename columns to more meaningful internal names based on info.json descriptions
    df = df.rename(
        columns={
            "genus": "num_missing",  # count of missing teeth of the given class
            "age": "num_sockets",  # observable sockets
            "pop": "age_est",  # estimated age at death
            "stdev_age": "prob_male_cont",  # continuous estimate of probability male
            "sockets": "tooth_class",  # Anterior / Posterior / Premolar
            "tooth_class": "species_label",  # e.g., Homo sapiens, Pan, Papio, Pongo
        }
    )

    # Drop rows with non-positive socket counts or impossible counts
    df = df[df["num_sockets"] > 0].copy()
    df = df[(df["num_missing"] >= 0) & (df["num_missing"] <= df["num_sockets"])].copy()

    # Create genus/species indicators
    df["species_label"] = df["species_label"].astype(str)
    df["is_human"] = (df["species_label"].str.contains("Homo sapiens")).astype(int)

    # It is possible that humans are encoded simply as "Homo"; fall back to that if needed.
    if df["is_human"].sum() == 0:
        df["is_human"] = df["species_label"].str.contains("Homo").astype(int)

    # Tooth class categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Sex proxy: probability of being male (0–1). Clamp to sensible bounds.
    df["prob_male_cont"] = df["prob_male_cont"].clip(0.0, 1.0)

    # Proportion of missing teeth for binomial regression
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    model = smf.glm(
        "prop_missing ~ is_human + age_est + prob_male_cont + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    ).fit()
    return model


def summarize_results(df: pd.DataFrame, model) -> dict:
    # Overall descriptive rates
    df["is_human_bool"] = df["is_human"] == 1

    summary = {}
    for label, mask in [("human", df["is_human_bool"]), ("non_human", ~df["is_human_bool"])]:
        sub = df[mask]
        total_sockets = float(sub["num_sockets"].sum())
        total_missing = float(sub["num_missing"].sum())
        rate = total_missing / total_sockets if total_sockets > 0 else np.nan
        summary[label] = {
            "n_rows": int(len(sub)),
            "total_sockets": int(total_sockets),
            "total_missing": int(total_missing),
            "missing_rate": rate,
        }

    # Extract coefficient and standard error for the human effect
    coef = model.params.get("is_human", np.nan)
    se = model.bse.get("is_human", np.nan)
    pvalue = model.pvalues.get("is_human", np.nan)

    # Odds ratio and an approximate 95% CI
    if np.isfinite(coef) and np.isfinite(se):
        or_est = float(np.exp(coef))
        ci_low = float(np.exp(coef - 1.96 * se))
        ci_high = float(np.exp(coef + 1.96 * se))
    else:
        or_est = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        "descriptive": summary,
        "coef_is_human": float(coef) if np.isfinite(coef) else None,
        "se_is_human": float(se) if np.isfinite(se) else None,
        "pvalue_is_human": float(pvalue) if np.isfinite(pvalue) else None,
        "or_is_human": or_est if np.isfinite(or_est) else None,
        "or_ci_low": ci_low if np.isfinite(ci_low) else None,
        "or_ci_high": ci_high if np.isfinite(ci_high) else None,
    }


def map_result_to_likert(coef: float | None, pvalue: float | None) -> int:
    # Default to uncertain / neutral if anything is missing
    if coef is None or pvalue is None or not np.isfinite(coef) or not np.isfinite(pvalue):
        return 50

    # Positive coefficient means higher AMTL in humans; negative means lower.
    if coef > 0:
        if pvalue < 0.001:
            return 95
        if pvalue < 0.01:
            return 90
        if pvalue < 0.05:
            return 80
        return 65
    else:
        if pvalue < 0.001:
            return 5
        if pvalue < 0.01:
            return 10
        if pvalue < 0.05:
            return 20
        return 45


def build_explanation(summary: dict) -> str:
    desc = summary["descriptive"]
    human = desc["human"]
    non_human = desc["non_human"]

    coef = summary["coef_is_human"]
    pvalue = summary["pvalue_is_human"]
    or_est = summary["or_is_human"]
    ci_low = summary["or_ci_low"]
    ci_high = summary["or_ci_high"]

    parts = []

    parts.append(
        "I analyzed the AMTL dataset (1450 rows) using a binomial regression "
        "that models the proportion of missing teeth in each specimen/tooth-class "
        "combination as a function of whether the specimen is a modern human "
        "(Homo sapiens), estimated age at death, estimated probability of being male, "
        "and tooth class (anterior, posterior, premolar). Counts of missing teeth "
        "and observable sockets were taken from the `genus` and `age` columns, "
        "respectively, as described in the metadata."
    )

    parts.append(
        f"Descriptively, modern humans contributed {human['n_rows']} rows with "
        f"{human['total_missing']} missing teeth out of {human['total_sockets']} sockets "
        f"(raw AMTL rate ≈ {human['missing_rate']:.3f}), while non-human primates "
        f"contributed {non_human['n_rows']} rows with {non_human['total_missing']} missing teeth "
        f"out of {non_human['total_sockets']} sockets "
        f"(raw AMTL rate ≈ {non_human['missing_rate']:.3f})."
    )

    if coef is not None and pvalue is not None and or_est is not None:
        parts.append(
            "In the adjusted binomial regression, the coefficient for the human indicator "
            f"was {coef:.3f} (log-odds scale), corresponding to an odds ratio of "
            f"{or_est:.2f} for AMTL in modern humans compared to non-human primates, "
            f"with an approximate 95% confidence interval from {ci_low:.2f} to {ci_high:.2f} "
            f"and p-value {pvalue:.3g}."
        )
    else:
        parts.append(
            "The binomial regression for the human indicator did not yield a stable "
            "coefficient estimate (likely due to numerical issues), so inference relies "
            "primarily on descriptive comparisons."
        )

    if coef is None or pvalue is None:
        parts.append(
            "Because the model-based human effect could not be reliably estimated, "
            "I treat the evidence as inconclusive and place the response near the neutral point."
        )
    elif coef > 0:
        if pvalue < 0.05:
            parts.append(
                "The positive and statistically significant human coefficient indicates that, "
                "after accounting for age, sex, and tooth class, modern humans have higher "
                "frequencies of antemortem tooth loss than the non-human primate genera in "
                "this sample."
            )
        else:
            parts.append(
                "The human coefficient is positive but not statistically significant at "
                "conventional levels, so while point estimates suggest higher AMTL in modern "
                "humans after adjustment, the evidence is modest and compatible with no real "
                "difference."
            )
    else:
        if pvalue < 0.05:
            parts.append(
                "The negative and statistically significant human coefficient indicates that, "
                "after accounting for age, sex, and tooth class, modern humans actually show "
                "lower frequencies of antemortem tooth loss than the non-human primate genera "
                "in this sample."
            )
        else:
            parts.append(
                "The human coefficient is negative but not statistically significant, so the "
                "adjusted model does not support higher AMTL frequencies in modern humans; "
                "if anything, the point estimates suggest slightly lower rates, but the "
                "evidence is weak."
            )

    return " ".join(parts)


def main():
    df = load_and_prepare_data("amtl.csv")
    model = fit_binomial_model(df)
    summary = summarize_results(df, model)

    response_value = map_result_to_likert(
        summary["coef_is_human"], summary["pvalue_is_human"]
    )
    explanation_text = build_explanation(summary)

    conclusion = {
        "response": int(response_value),
        "explanation": explanation_text,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

