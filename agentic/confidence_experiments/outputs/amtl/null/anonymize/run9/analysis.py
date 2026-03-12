import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Basic sanity checks
    for col in ["feature3", "feature4"]:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}.")

    # Compute response as proportion missing with total sockets as weights
    df["missing"] = df["feature3"].astype(float)
    df["sockets"] = df["feature4"].astype(float)
    df = df[df["sockets"] > 0].copy()
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Center age and sex for numerical stability
    df["age_c"] = df["feature5"] - df["feature5"].mean()
    df["sex_c"] = df["feature7"] - df["feature7"].mean()

    # Fit binomial GLM with logit link, using sockets as frequency weights
    formula = "prop_missing ~ is_human + age_c + sex_c + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Extract coefficient and p-value for the human indicator
    coef = model.params["is_human"]
    pval = model.pvalues["is_human"]

    # Average marginal effect on predicted proportion missing
    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0

    pred_human = model.predict(df_human)
    pred_nonhuman = model.predict(df_nonhuman)
    avg_effect = float(np.mean(pred_human - pred_nonhuman))

    # Descriptive statistics by genus
    genus_group = (
        df.assign(prop=df["missing"] / df["sockets"])
        .groupby("feature8")
        .agg(
            mean_prop=("prop", "mean"),
            n_specimens=("feature2", "nunique"),
            total_sockets=("sockets", "sum"),
        )
        .reset_index()
    )

    # Collect results needed for downstream interpretation
    results = {
        "coef_is_human": float(coef),
        "pval_is_human": float(pval),
        "avg_effect": avg_effect,
        "genus_summary": genus_group.to_dict(orient="list"),
    }

    with open("analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

