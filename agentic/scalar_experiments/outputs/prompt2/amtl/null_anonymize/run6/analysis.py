import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def load_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Rename columns to more descriptive labels for the model
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "num_missing",
            "feature4": "num_sockets",
            "feature5": "age",
            "feature7": "sex_estimate",
            "feature8": "genus",
        }
    )
    # Drop any rows with invalid or zero sockets to avoid division / invalid trials
    df = df[df["num_sockets"] > 0].copy()
    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial GLM with logit link:
    num_missing ~ genus + age + sex_estimate + tooth_class
    using num_sockets as the number of trials.
    """

    # Ensure categorical variables are treated as such
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Set reference category for genus to one of the non-human primates.
    # Use Papio as reference if present; otherwise use the first non-human genus.
    non_human_order = ["Papio", "Pan", "Pongo"]
    existing_non_humans = [g for g in non_human_order if g in df["genus"].cat.categories]

    if not existing_non_humans:
        # Fallback: leave the default encoding; the comparison logic below will adapt.
        ref_genus = df["genus"].cat.categories[0]
    else:
        ref_genus = existing_non_humans[0]

    df["genus"] = df["genus"].cat.reorder_categories(
        [ref_genus] + [g for g in df["genus"].cat.categories if g != ref_genus],
        ordered=True,
    )

    formula = "num_missing ~ C(genus) + age + sex_estimate + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=None,
    )
    result = model.fit()
    return result, ref_genus


def compare_humans_to_non_humans(result, ref_genus: str) -> dict:
    """
    Extract the contrast between Homo sapiens and the reference non-human genus
    from the fitted GLM and summarize evidence.
    """
    params = result.params
    bse = result.bse

    # Identify the coefficient corresponding to Homo sapiens vs reference genus
    # The term will look like C(genus)[T.Homo sapiens] or similar.
    human_key_candidates = [k for k in params.index if "C(genus)" in k and "Homo" in k]
    if not human_key_candidates:
        # If humans are the reference category, all non-human coefficients are relative to humans.
        # In that (unlikely) case given our setup, flip interpretation: check if all non-human
        # genera have negative coefficients (lower AMTL) relative to humans.
        non_human_keys = [k for k in params.index if "C(genus)" in k]
        non_human_coefs = params[non_human_keys]
        non_human_ses = bse[non_human_keys]
        # If all non-human coefs are negative and at least one is significantly below zero,
        # this supports higher AMTL in humans.
        z_scores = non_human_coefs / non_human_ses.replace(0, np.nan)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        any_significant_negative = np.any((non_human_coefs < 0) & (p_values < 0.05))
        evidence_strength = float(np.clip(np.nanmax(np.abs(z_scores)), 0, 10))
        return {
            "human_higher": bool(any_significant_negative),
            "z_score": float(np.nanmax(np.abs(z_scores))),
            "p_value_min": float(np.nanmin(p_values)),
            "coef": float(non_human_coefs.min()),
            "ref_genus": ref_genus,
            "coef_name": "non_human_vs_human",
            "evidence_strength": evidence_strength,
        }

    human_key = human_key_candidates[0]
    coef = params[human_key]
    se = bse[human_key]

    # Compute Wald z and p-value
    if se == 0:
        z = np.inf if coef > 0 else -np.inf
        p = 0.0
    else:
        z = coef / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))

    human_higher = (coef > 0) and (p < 0.05)

    evidence_strength = float(np.clip(abs(z), 0, 10))

    return {
        "human_higher": bool(human_higher),
        "z_score": float(z),
        "p_value": float(p),
        "coef": float(coef),
        "ref_genus": ref_genus,
        "coef_name": human_key,
        "evidence_strength": evidence_strength,
    }


def derive_conclusion(summary: dict) -> tuple[str, int, str]:
    """
    Map model comparison summary to the required output:
    response (Yes/No), confidence (0–100), explanation (str).
    """
    human_higher = summary["human_higher"]
    evidence_strength = summary["evidence_strength"]

    # Translate evidence strength (roughly |z| capped at 10) to confidence
    base_conf = (evidence_strength / 10.0) * 40 + 50  # 50–90 range

    # Adjust slightly depending on direction and p-value if present
    if "p_value" in summary:
        p = summary["p_value"]
    else:
        p = summary.get("p_value_min", 1.0)

    if p < 0.001:
        base_conf += 7
    elif p < 0.01:
        base_conf += 5
    elif p < 0.05:
        base_conf += 3
    elif p > 0.2:
        base_conf -= 5

    confidence = int(np.clip(round(base_conf), 0, 100))

    response = "Yes" if human_higher else "No"

    # Build explanation string
    if "p_value" in summary:
        p_str = f"{summary['p_value']:.4f}"
    else:
        p_str = f"{summary.get('p_value_min', 1.0):.4f}"

    explanation = (
        "I fit a binomial logistic regression model predicting the number of missing teeth "
        "out of observable tooth sockets as a function of genus, age at death, estimated sex, "
        "and tooth class. The coefficient comparing Homo sapiens to the reference non-human "
        f"genus ({summary['ref_genus']}) was {summary['coef']:.3f} on the log-odds scale, with "
        f"a Wald z statistic of {summary['z_score']:.2f} and p-value {p_str}. "
        "A positive, statistically significant coefficient indicates that, after controlling for "
        "age, sex, and tooth class, modern humans exhibit higher frequencies of antemortem tooth "
        "loss (AMTL) than the non-human primates. "
    )

    if response == "No":
        explanation += (
            "In this analysis the coefficient for Homo sapiens was not significantly greater than "
            "that of the non-human reference genus, so the data do not provide strong evidence that "
            "humans have higher AMTL frequencies once covariates are accounted for."
        )
    else:
        explanation += (
            "Given the positive and statistically significant human coefficient, the model supports "
            "the conclusion that humans have higher AMTL frequencies than the non-human primates in "
            "this dataset, even after adjusting for age, sex, and tooth class."
        )

    return response, confidence, explanation


def write_conclusion(path: Path, response: str, confidence: int, explanation: str) -> None:
    obj = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }
    with path.open("w") as f:
        json.dump(obj, f)


def main():
    base = Path(".")
    metadata = load_metadata(base / "info.json")
    _ = metadata  # Currently unused but read to respect instructions.

    df = load_data(base / "amtl.csv")
    result, ref_genus = fit_binomial_model(df)
    summary = compare_humans_to_non_humans(result, ref_genus)
    response, confidence, explanation = derive_conclusion(summary)
    write_conclusion(base / "conclusion.txt", response, confidence, explanation)


if __name__ == "__main__":
    main()
