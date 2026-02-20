import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent

    # Load data
    df = pd.read_csv(base_path / "amtl.csv")

    # Basic sanity checks matching metadata expectations
    assert {"tooth_class", "specimen", "num_amtl", "sockets", "age", "stdev_age", "prob_male", "genus", "pop"}.issubset(
        df.columns
    ), "Unexpected columns in amtl.csv"

    # Create human indicator (Homo sapiens vs non-human primates Pan, Pongo, Papio)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive: overall AMTL proportion by genus
    genus_summary = (
        df.groupby("genus")
        .agg(total_missing=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .assign(prop_missing=lambda x: x["total_missing"] / x["total_sockets"])
    )

    # Expand to per-tooth records for a straightforward logistic regression
    records = []
    for _, row in df.iterrows():
        n_missing = int(row["num_amtl"])
        n_present = int(row["sockets"] - row["num_amtl"])

        covariates = {
            "is_human": int(row["is_human"]),
            "age": float(row["age"]),
            "prob_male": float(row["prob_male"]),
            "tooth_class": row["tooth_class"],
        }

        # 1 = tooth was lost antemortem, 0 = tooth present
        records.extend({**covariates, "amtl_loss": 1} for _ in range(n_missing))
        records.extend({**covariates, "amtl_loss": 0} for _ in range(n_present))

    long_df = pd.DataFrame.from_records(records)

    # Fit logistic regression: AMTL (per tooth) ~ human vs non-human + age + sex + tooth class
    model = smf.logit("amtl_loss ~ is_human + age + prob_male + C(tooth_class)", data=long_df)
    result = model.fit(disp=False)

    # Extract key statistics for the human effect
    coef_human = float(result.params["is_human"])
    se_human = float(result.bse["is_human"])
    pval_human = float(result.pvalues["is_human"])

    odds_ratio_human = float(np.exp(coef_human))
    ci_low, ci_high = result.conf_int().loc["is_human"]
    or_ci_low = float(np.exp(ci_low))
    or_ci_high = float(np.exp(ci_high))

    # Estimated marginal probabilities for humans vs non-humans,
    # averaging over the observed covariate distribution.
    base_covariates = long_df.copy()

    human_cov = base_covariates.copy()
    human_cov["is_human"] = 1
    pred_human = result.predict(human_cov).mean()

    nonhuman_cov = base_covariates.copy()
    nonhuman_cov["is_human"] = 0
    pred_nonhuman = result.predict(nonhuman_cov).mean()

    prob_diff = float(pred_human - pred_nonhuman)

    # Map evidence to a 0-100 Likert response.
    # Start from 50 (uncertain), then adjust by sign, effect size, and p-value.
    if coef_human > 0:
        # Positive effect: humans show higher AMTL frequency
        # Strength grows with larger odds ratios and smaller p-values.
        base = 60.0
        # Cap effect-based increment between 0 and 25
        effect_increment = min(25.0, max(0.0, (odds_ratio_human - 1.0) * 10.0))
        # Cap significance-based increment between 0 and 15
        if pval_human <= 0:
            sig_increment = 15.0
        else:
            sig_increment = min(15.0, max(0.0, -10.0 * np.log10(pval_human)))
        response_scalar = base + effect_increment + sig_increment
    else:
        # Negative or null effect: evidence against humans having higher AMTL
        base = 40.0
        effect_decrement = min(25.0, max(0.0, (1.0 - odds_ratio_human) * 10.0))
        if pval_human <= 0:
            sig_decrement = 15.0
        else:
            sig_decrement = min(15.0, max(0.0, -10.0 * np.log10(pval_human)))
        response_scalar = base - effect_decrement - sig_decrement

    # Clamp to [0, 100] and convert to int
    response_int = int(round(min(100.0, max(0.0, response_scalar))))

    # Build explanation summarizing evidence
    explanation_parts = []
    explanation_parts.append(
        "I fit a logistic regression model at the per-tooth level, "
        "with antemortem tooth loss (AMTL) coded as a binary outcome and predictors including "
        "a Homo sapiens indicator, age at death, estimated sex (probability of male), and tooth class."
    )
    explanation_parts.append(
        "Descriptively, the genus-level AMTL proportions (number of missing teeth divided by observable sockets) were: "
        + ", ".join(f"{genus}: {row.prop_missing:.3f}" for genus, row in genus_summary.iterrows())
        + "."
    )
    explanation_parts.append(
        "In the regression model, the coefficient for the Homo sapiens indicator corresponds to an odds ratio "
        f"of {odds_ratio_human:.2f} (95% CI {or_ci_low:.2f}–{or_ci_high:.2f}, p = {pval_human:.3g})."
    )
    explanation_parts.append(
        f"After averaging over the observed age, sex, and tooth-class distribution, the predicted probability of AMTL "
        f"per tooth was {pred_human:.3f} for humans and {pred_nonhuman:.3f} for non-human primates, "
        f"a difference of {prob_diff:.3f}."
    )

    if coef_human > 0:
        explanation_parts.append(
            "Because the human effect is positive and the odds ratio exceeds 1, "
            "the data support the claim that modern humans have higher AMTL frequencies than the non-human primate genera, "
            "after accounting for age, sex, and tooth class."
        )
    else:
        explanation_parts.append(
            "Because the human effect is not positive (odds ratio at or below 1), "
            "the data do not support the claim that modern humans have higher AMTL frequencies than the non-human primate genera "
            "once age, sex, and tooth class are accounted for."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {"response": response_int, "explanation": explanation}

    # Write JSON output to conclusion.txt as required
    with open(base_path / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
