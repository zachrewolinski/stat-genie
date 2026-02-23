import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).parent
    data_path = base_path / "amtl.csv"

    df = pd.read_csv(data_path)

    # Basic sanity checks on key variables
    df = df.copy()
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Remove any rows with obviously invalid socket/AMTL values
    df = df[df["sockets"] > 0].copy()
    df = df[(df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])].copy()

    # Proportion of missing teeth and weights for binomial GLM
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Center age to improve numerical stability
    df["age_c"] = df["age"] - df["age"].mean()

    formula = "prop_amtl ~ is_human + age_c + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Also fit with cluster-robust SEs by specimen to account for
    # non-independence of multiple tooth classes within individuals.
    result_robust = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["specimen"]},
    )

    # Compute an odds ratio and 95% CI for the human effect
    human_coef = result.params["is_human"]
    human_se = result.bse["is_human"]
    human_or = float(np.exp(human_coef))
    ci_low = float(np.exp(human_coef - 1.96 * human_se))
    ci_high = float(np.exp(human_coef + 1.96 * human_se))
    p_value = float(result.pvalues["is_human"])

    # Cluster-robust version
    human_coef_rb = result_robust.params["is_human"]
    human_se_rb = result_robust.bse["is_human"]
    human_or_rb = float(np.exp(human_coef_rb))
    ci_low_rb = float(np.exp(human_coef_rb - 1.96 * human_se_rb))
    ci_high_rb = float(np.exp(human_coef_rb + 1.96 * human_se_rb))
    p_value_rb = float(result_robust.pvalues["is_human"])

    # Get some representative predicted probabilities at median age, prob_male=0.5
    median_age_c = 0.0  # by construction
    median_prob_male = df["prob_male"].median()
    ref_tooth_class = df["tooth_class"].mode()[0]

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age_c": [median_age_c, median_age_c],
            "prob_male": [median_prob_male, median_prob_male],
            "tooth_class": [ref_tooth_class, ref_tooth_class],
        }
    )

    preds = result.get_prediction(new_data).summary_frame()
    # Predicted mean is on response (probability) scale for Binomial
    prob_nonhuman = float(preds["mean"].iloc[0])
    prob_human = float(preds["mean"].iloc[1])

    summary = {
        "n_rows": int(len(df)),
        "n_specimens": int(df["specimen"].nunique()),
        "genus_counts": df["genus"].value_counts().to_dict(),
        "tooth_class_counts": df["tooth_class"].value_counts().to_dict(),
        "human_or": human_or,
        "human_or_ci": [ci_low, ci_high],
        "human_p_value": p_value,
        "human_or_cluster": human_or_rb,
        "human_or_cluster_ci": [ci_low_rb, ci_high_rb],
        "human_p_value_cluster": p_value_rb,
        "prob_nonhuman_at_median": prob_nonhuman,
        "prob_human_at_median": prob_human,
        "ref_tooth_class": ref_tooth_class,
    }

    # Write a small JSON summary for inspection
    out_path = base_path / "analysis_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
