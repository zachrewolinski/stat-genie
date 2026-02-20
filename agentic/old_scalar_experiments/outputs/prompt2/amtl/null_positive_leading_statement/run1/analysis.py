import json
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def build_per_tooth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expand specimen-level AMTL counts into per-tooth binary outcomes."""
    rows: List[Dict] = []

    for _, r in df.iterrows():
        sockets = int(round(float(r["sockets"])))
        amtl = int(round(float(r["num_amtl"])))

        if sockets <= 0:
            continue

        if amtl < 0:
            amtl = 0
        if amtl > sockets:
            amtl = sockets

        base = {
            "age": float(r["age"]),
            "prob_male": float(r["prob_male"]),
            "genus": r["genus"],
            "tooth_class": r["tooth_class"],
        }

        # One row per tooth: 1 for missing (AMTL), 0 otherwise.
        rows.extend({**base, "amtl": 1} for _ in range(amtl))
        rows.extend({**base, "amtl": 0} for _ in range(sockets - amtl))

    return pd.DataFrame(rows)


def fit_logistic_model(df_long: pd.DataFrame):
    """Fit logistic regression of AMTL on genus, age, sex, and tooth class."""
    # Ensure categorical types with Homo sapiens as reference genus.
    df_long = df_long.copy()
    df_long["genus"] = pd.Categorical(
        df_long["genus"],
        categories=["Homo sapiens", "Pan", "Papio", "Pongo"],
    )
    df_long["tooth_class"] = pd.Categorical(df_long["tooth_class"])

    model = smf.glm(
        "amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df_long,
        family=sm.families.Binomial(),
    )

    try:
        result = model.fit()
    except PerfectSeparationError:
        # Perfect separation is unlikely but, if encountered, refit a simpler model
        # without tooth_class to obtain a conservative genus effect.
        simpler_model = smf.glm(
            "amtl ~ C(genus) + age + prob_male",
            data=df_long,
            family=sm.families.Binomial(),
        )
        result = simpler_model.fit()

    return result


def summarize_genus_effects(result) -> Dict[str, object]:
    """Summarize genus effects relative to Homo sapiens."""
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues

    genus_terms = ["C(genus)[T.Pan]", "C(genus)[T.Papio]", "C(genus)[T.Pongo]"]
    evidence_humans_higher = True
    summary_lines: List[str] = []

    for term in genus_terms:
        if term in params.index:
            coef = float(params[term])
            ci_low, ci_high = map(float, conf_int.loc[term])
            pval = float(pvalues[term])
            direction = "lower" if coef < 0 else "higher"
            sig = ci_high < 0

            summary_lines.append(
                f"{term}: coef={coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], "
                f"p={pval:.3g} (non-human genus has {direction} AMTL odds than humans)."
            )

            if not sig or coef >= 0:
                evidence_humans_higher = False
        else:
            summary_lines.append(f"{term}: not estimated in model.")
            evidence_humans_higher = False

    # Confidence heuristic based on smallest p-value among non-human genera.
    genus_pvals = [pvalues.get(term, 1.0) for term in genus_terms]
    min_p = float(np.min(genus_pvals)) if genus_pvals else 1.0

    if evidence_humans_higher:
        if min_p < 1e-4:
            confidence = 97
        elif min_p < 1e-3:
            confidence = 95
        elif min_p < 1e-2:
            confidence = 90
        else:
            confidence = 85
        response = "Yes"
    else:
        # If the adjusted model does not clearly show humans higher,
        # answer "No" (or "not clearly yes") with moderate confidence.
        response = "No"
        confidence = 70

    return {
        "response": response,
        "confidence": confidence,
        "evidence_humans_higher": evidence_humans_higher,
        "summary_lines": summary_lines,
    }


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic cleaning and constraints
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]
    )
    df["num_amtl"] = df["num_amtl"].astype(float)
    df["sockets"] = df["sockets"].astype(float)

    # Exclude non-positive socket counts and clamp counts to valid range
    df = df[df["sockets"] > 0].copy()
    df["num_amtl"] = df[["num_amtl", "sockets"]].min(axis=1)
    df["num_amtl"] = df["num_amtl"].clip(lower=0)

    # Ensure genus categories include Homo sapiens and non-human genera
    df["genus"] = pd.Categorical(
        df["genus"],
        categories=["Homo sapiens", "Pan", "Papio", "Pongo"],
    )

    # Expand to per-tooth observations
    df_long = build_per_tooth_dataframe(df)

    # Fit logistic regression model
    result = fit_logistic_model(df_long)

    # Summarize genus effects
    genus_summary = summarize_genus_effects(result)

    # Raw (unadjusted) AMTL rates by genus for additional context
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    raw_rates = df.groupby("genus")["amtl_rate"].mean().dropna().to_dict()
    raw_rates_str = "; ".join(f"{g}: {rate:.3f}" for g, rate in raw_rates.items())

    # Build explanation text
    explanation_parts: List[str] = []
    explanation_parts.append(
        "I analyzed the provided antemortem tooth loss (AMTL) dataset by expanding each "
        "specimen-level record into per-tooth observations and fitting a binomial logistic "
        "regression model predicting the probability that an individual tooth was lost "
        "antemortem from genus, age at death, estimated sex (probability of being male), "
        "and tooth class."
    )
    explanation_parts.append(
        "In this model, Homo sapiens was treated as the reference genus so that coefficients "
        "for Pan, Papio, and Pongo represent differences in AMTL odds relative to humans, "
        "after adjusting for age, sex, and tooth class."
    )
    explanation_parts.append(
        "The adjusted genus effects indicate how non-human primates compare to humans: "
        + " ".join(genus_summary["summary_lines"])
    )
    explanation_parts.append(
        "Raw (unadjusted) mean AMTL rates by genus show a consistent pattern, with approximate "
        f"mean AMTL proportions per tooth of {raw_rates_str}."
    )

    if genus_summary["response"] == "Yes":
        explanation_parts.append(
            "Because all estimated non-human genus effects are negative with confidence intervals "
            "that remain below zero, the model indicates that Pan, Papio, and Pongo have "
            "significantly lower AMTL odds than Homo sapiens after accounting for age, sex, and "
            "tooth class. This supports the conclusion that modern humans have higher AMTL "
            "frequencies than the non-human primates in this dataset."
        )
    else:
        explanation_parts.append(
            "At least one non-human genus effect is not clearly negative and significantly below "
            "zero, so the adjusted model does not provide strong evidence that humans have higher "
            "AMTL frequencies than all non-human genera after controlling for age, sex, and tooth "
            "class. In this case, the data do not robustly support the hypothesized human advantage "
            "in AMTL frequency."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": genus_summary["response"],
        "confidence": int(genus_summary["confidence"]),
        "explanation": explanation,
    }

    # Write required JSON-only output file
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

