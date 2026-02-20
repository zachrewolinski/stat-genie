import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic derived variables
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Guard against any impossible values due to data issues
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["prop_missing", "age", "sex_estimate", "n_sockets"]
    )

    # Create indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive statistics by genus (unadjusted proportions)
    genus_summary = (
        df.groupby("genus")
        .apply(
            lambda g: pd.Series(
                {
                    "total_missing": g["n_missing"].sum(),
                    "total_sockets": g["n_sockets"].sum(),
                    "mean_prop_missing": (g["n_missing"].sum() / g["n_sockets"].sum())
                    if g["n_sockets"].sum() > 0
                    else np.nan,
                    "n_specimens": g.shape[0],
                }
            )
        )
        .sort_values("mean_prop_missing", ascending=False)
    )

    # Fit binomial regression with is_human indicator, controlling for age, sex, and tooth class
    # We treat prop_missing as the outcome with binomial variance using n_sockets as weights.
    human_model = smf.glm(
        formula="prop_missing ~ is_human + C(tooth_class) + age + sex_estimate",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()

    # Average predicted probabilities under counterfactual "everyone is human" vs "everyone is non-human"
    df_all_human = df.copy()
    df_all_human["is_human"] = 1
    df_all_nonhuman = df.copy()
    df_all_nonhuman["is_human"] = 0

    avg_p_human = float(human_model.predict(df_all_human).mean())
    avg_p_nonhuman = float(human_model.predict(df_all_nonhuman).mean())

    coef_is_human = float(human_model.params["is_human"])
    pval_is_human = float(human_model.pvalues["is_human"])
    or_is_human = float(np.exp(coef_is_human))

    humans_higher = avg_p_human > avg_p_nonhuman

    # Map statistical evidence to a rough confidence score
    if pval_is_human < 0.001:
        base_conf = 98
    elif pval_is_human < 0.01:
        base_conf = 95
    elif pval_is_human < 0.05:
        base_conf = 88
    elif pval_is_human < 0.1:
        base_conf = 70
    else:
        base_conf = 55

    # If the odds ratio is very close to 1, reduce confidence slightly.
    if 0.9 <= or_is_human <= 1.1:
        base_conf = max(0, base_conf - 10)

    confidence = int(max(0, min(100, base_conf)))

    response = "Yes" if humans_higher else "No"

    # Build explanation string summarizing key numerical evidence
    # Use a single-line explanation to keep the JSON compact.
    explanation_parts = []

    explanation_parts.append(
        "I fit a binomial regression model for the proportion of missing teeth per specimen "
        "using the number of missing teeth out of observable sockets as the outcome and "
        "included an indicator for modern humans versus non-human primates, while controlling "
        "for age at death, estimated sex, and tooth class."
    )

    explanation_parts.append(
        f"In this model, the human indicator had an odds ratio of approximately {or_is_human:.2f} "
        f"(log-odds coefficient {coef_is_human:.2f}, p-value {pval_is_human:.3g})."
    )

    explanation_parts.append(
        f"The average model-predicted probability of antemortem tooth loss was about {avg_p_human:.3f} "
        f"if all observations were set to humans and {avg_p_nonhuman:.3f} if all were set to non-human primates, "
        "holding the distributions of age, sex, and tooth class fixed."
    )

    # Add unadjusted descriptive comparison by genus for additional context
    top_genus = genus_summary.index.tolist()
    if top_genus:
        desc_bits = []
        for genus_name in top_genus:
            row = genus_summary.loc[genus_name]
            desc_bits.append(
                f"{genus_name} (mean proportion missing ≈ {row['mean_prop_missing']:.3f}, n = {int(row['n_specimens'])})"
            )
        explanation_parts.append(
            "Unadjusted mean proportions of missing teeth by genus (ignoring covariates) were: "
            + "; ".join(desc_bits)
            + "."
        )

    if response == "Yes":
        explanation_parts.append(
            "Together, these results indicate that, after accounting for age, sex, and tooth class, "
            "modern humans have higher predicted frequencies of antemortem tooth loss than the pooled "
            "set of non-human primate genera in this dataset."
        )
    else:
        explanation_parts.append(
            "Together, these results indicate that, after accounting for age, sex, and tooth class, "
            "modern humans do not have higher predicted frequencies of antemortem tooth loss than "
            "the pooled set of non-human primate genera in this dataset."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

