import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    data = pd.read_csv(data_path)

    # Keep only rows with valid counts.
    data = data.copy()
    data = data.loc[(data["sockets"] > 0) & (data["num_amtl"] >= 0)].reset_index(drop=True)

    # Indicator for modern humans vs non-human primates.
    data["is_human"] = (data["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth in each row.
    data["prop_amtl"] = data["num_amtl"] / data["sockets"]

    # Basic sanity checks.
    genus_counts = data["genus"].value_counts()
    print("Genus counts:")
    print(genus_counts.to_string())
    print()

    print("Mean AMTL proportion by genus:")
    print(
        data.groupby("genus")
        .apply(lambda group: group["num_amtl"].sum() / group["sockets"].sum())
        .to_string()
    )
    print()

    # Center age to improve numerical stability.
    data["age_c"] = data["age"] - data["age"].mean()

    # Fit binomial GLM with logit link, using grouped data with socket counts as weights.
    formula = "prop_amtl ~ is_human + age_c + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data["sockets"],
    )

    # Cluster-robust standard errors at specimen level to account for repeated measures.
    result = model.fit(cov_type="cluster", cov_kwds={"groups": data["specimen"]})

    print("GLM summary (cluster-robust by specimen):")
    print(result.summary())
    print()

    # Extract key statistics for the human indicator.
    coef_human = result.params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    pval_human = result.pvalues.get("is_human", np.nan)

    print(f"Coefficient for is_human: {coef_human:.4f}")
    print(f"Std. error (cluster-robust): {se_human:.4f}")
    print(f"P-value (cluster-robust): {pval_human:.4g}")

    # Predicted AMTL proportions for humans vs non-humans at average covariate values.
    mean_age_c = 0.0  # by construction
    mean_prob_male = float(data["prob_male"].mean())

    # Use the most common tooth class as reference scenario.
    mode_tooth_class = data["tooth_class"].mode().iat[0]

    def predict_prob(is_human: int) -> float:
        row = {
            "is_human": is_human,
            "age_c": mean_age_c,
            "prob_male": mean_prob_male,
            "tooth_class": mode_tooth_class,
        }
        df_row = pd.DataFrame([row])
        # For binomial GLM, predict returns the mean probability.
        return float(result.predict(df_row)[0])

    p_non_human = predict_prob(is_human=0)
    p_human = predict_prob(is_human=1)
    diff = p_human - p_non_human

    print()
    print(f"Predicted AMTL proportion (non-human): {p_non_human:.4f}")
    print(f"Predicted AMTL proportion (human):     {p_human:.4f}")
    print(f"Absolute difference (human - non):     {diff:.4f}")

    # Save a small JSON summary to inspect programmatically if desired.
    summary = {
        "coef_human": coef_human,
        "se_human": se_human,
        "pval_human": pval_human,
        "pred_non_human": p_non_human,
        "pred_human": p_human,
        "pred_diff": diff,
        "formula": formula,
    }
    summary_path = Path("analysis_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote key results to {summary_path}")


if __name__ == "__main__":
    main()

