import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_est",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Basic cleaning
    df = df[df["n_sockets"] > 0].copy()

    # Clamp any rows where missing teeth exceed observable sockets so that
    # binomial modeling assumptions are not violated.
    over_mask = df["n_missing"] > df["n_sockets"]
    n_over = int(over_mask.sum())
    if n_over > 0:
        df.loc[over_mask, "n_missing"] = df.loc[over_mask, "n_sockets"]

    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    df["tooth_class"] = df["tooth_class"].astype("category")
    df["genus"] = df["genus"].astype("category")

    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Descriptive statistics: AMTL rates by genus
    genus_summary = (
        df.groupby("genus")[["n_missing", "n_sockets"]].sum().assign(
            rate=lambda d: d["n_missing"] / d["n_sockets"]
        )
    )

    print("Rows where n_missing clamped to n_sockets:", n_over)
    print("\nAMTL rate (missing / sockets) by genus:")
    print(genus_summary)

    # Binomial regression with socket-level weighting
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_est + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    ).fit()

    print("\nBinomial regression summary (AMTL probability):")
    print(model.summary())

    coef_human = model.params.get("is_human", np.nan)
    se_human = model.bse.get("is_human", np.nan)
    pval_human = model.pvalues.get("is_human", np.nan)
    odds_ratio_human = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan
    ci_low = float(np.exp(coef_human - 1.96 * se_human)) if np.isfinite(se_human) else np.nan
    ci_high = float(np.exp(coef_human + 1.96 * se_human)) if np.isfinite(se_human) else np.nan

    print("\nEffect of being human (Homo sapiens) on AMTL probability:")
    print(f"  Log-odds coefficient: {coef_human:.4f}")
    print(f"  Standard error:       {se_human:.4f}")
    print(f"  Odds ratio:           {odds_ratio_human:.4f}")
    print(f"  95% CI for OR:        [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  p-value:              {pval_human:.4g}")

    # Save a small JSON with key statistics to inspect more easily if needed.
    results = {
        "n_rows": int(df.shape[0]),
        "n_over_clamped": n_over,
        "genus_summary": genus_summary.reset_index().to_dict(orient="records"),
        "human_effect": {
            "coef": float(coef_human),
            "se": float(se_human),
            "odds_ratio": odds_ratio_human,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": float(pval_human),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

