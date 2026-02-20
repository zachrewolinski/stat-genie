import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory")

    df = pd.read_csv(data_path)

    # Basic data checks
    required_cols = [
        "num_amtl",
        "sockets",
        "genus",
        "age",
        "prob_male",
        "tooth_class",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove rows with obviously invalid entries
    df = df.dropna(subset=required_cols).copy()
    df = df[df["sockets"] > 0].copy()
    df = df[df["num_amtl"].between(0, df["sockets"])].copy()

    # Create human vs non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth as response
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Center/scale continuous covariates for numerical stability
    for col in ["age", "prob_male"]:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            df[f"{col}_z"] = df[col] - mean
        else:
            df[f"{col}_z"] = (df[col] - mean) / std

    # Fit binomial regression: proportion with socket counts as frequency weights
    formula = "prop_amtl ~ is_human + age_z + prob_male_z + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract key information about human vs non-human effect
    coef_human = result.params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    pvalue_human = result.pvalues.get("is_human", np.nan)

    # Compute 95% CI on log-odds scale and transform to odds ratio
    if not np.isnan(coef_human) and not np.isnan(se_human):
        ci_low = coef_human - 1.96 * se_human
        ci_high = coef_human + 1.96 * se_human
        or_human = float(np.exp(coef_human))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))
    else:
        ci_low = ci_high = or_human = or_low = or_high = np.nan

    # Also compute predicted mean AMTL probabilities for human vs non-human
    cov_mean = df[["age_z", "prob_male_z"]].mean()
    reference_tooth = df["tooth_class"].mode().iloc[0]

    def predict_for_group(is_human_flag: int) -> float:
        new = pd.DataFrame(
            {
                "is_human": [is_human_flag],
                "age_z": [cov_mean["age_z"]],
                "prob_male_z": [cov_mean["prob_male_z"]],
                "tooth_class": [reference_tooth],
            }
        )
        return float(result.predict(new)[0])

    pred_human = predict_for_group(1)
    pred_nonhuman = predict_for_group(0)

    summary = {
        "n": int(len(df)),
        "n_humans": int(df["is_human"].sum()),
        "n_nonhumans": int((1 - df["is_human"]).sum()),
        "coef_human_log_odds": float(coef_human),
        "se_human": float(se_human),
        "pvalue_human": float(pvalue_human),
        "or_human": or_human,
        "or_human_ci_low": or_low,
        "or_human_ci_high": or_high,
        "pred_prob_human": pred_human,
        "pred_prob_nonhuman": pred_nonhuman,
        "reference_tooth_class": reference_tooth,
        "formula": formula,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

