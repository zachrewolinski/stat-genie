import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Keep only variables relevant to this analysis
    cols = ["num_amtl", "sockets", "age", "stdev_age", "prob_male", "genus", "tooth_class"]
    df = df[cols].copy()

    # Basic cleaning and validity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[df["sockets"] > 0]
    df = df[(df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])]

    # Proportion of missing teeth for each specimen and class
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Center age and sex-probability for numerical stability
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Unadjusted rates by genus
    genus_agg = (
        df.groupby("genus")
        .agg(total_amtl=("num_amtl", "sum"), total_sockets=("sockets", "sum"))
        .assign(raw_rate=lambda g: g["total_amtl"] / g["total_sockets"])
        .sort_values("raw_rate", ascending=False)
    )

    print("Unadjusted AMTL proportions by genus (num_amtl / sockets):")
    for genus, row in genus_agg.iterrows():
        print(f"  {genus:15s} rate={row['raw_rate']:.3f} (amtl={row['total_amtl']}, sockets={row['total_sockets']})")
    print()

    # Model 1: Human vs non-human (primary test)
    print("Fitting GLM Binomial: prop_amtl ~ is_human + age_c + prob_male_c + C(tooth_class)")
    model1 = smf.glm(
        "prop_amtl ~ is_human + age_c + prob_male_c + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )
    res1 = model1.fit()
    print(res1.summary())

    human_coef = res1.params["is_human"]
    human_se = res1.bse["is_human"]
    human_p = res1.pvalues["is_human"]
    human_or = float(np.exp(human_coef))
    human_ci_low = float(np.exp(human_coef - 1.96 * human_se))
    human_ci_high = float(np.exp(human_coef + 1.96 * human_se))

    print("\nPrimary effect: Homo sapiens vs non-human genera (Pan, Pongo, Papio)")
    print(f"  log-odds coef (is_human): {human_coef:.3f}")
    print(f"  odds ratio: {human_or:.3f} (95% CI [{human_ci_low:.3f}, {human_ci_high:.3f}])")
    print(f"  p-value: {human_p:.3g}")

    # Average marginal effect of being human vs non-human at observed covariates
    base = df.copy()
    base_nonhuman = base.copy()
    base_nonhuman["is_human"] = 0
    base_human = base.copy()
    base_human["is_human"] = 1

    preds_nonhuman = res1.predict(base_nonhuman)
    preds_human = res1.predict(base_human)

    avg_nonhuman = float(preds_nonhuman.mean())
    avg_human = float(preds_human.mean())
    diff = avg_human - avg_nonhuman

    print(
        f"\nAdjusted mean predicted AMTL proportion (averaged over age, sex, tooth class distribution):\n"
        f"  Non-human genera: {avg_nonhuman:.3f}\n"
        f"  Homo sapiens:     {avg_human:.3f}\n"
        f"  Absolute difference (human - non-human): {diff:.3f}"
    )

    # Model 2: Fully categorical genus effect for comparison
    print("\nFitting GLM Binomial: prop_amtl ~ C(genus) + age_c + prob_male_c + C(tooth_class)")
    model2 = smf.glm(
        "prop_amtl ~ C(genus) + age_c + prob_male_c + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )
    res2 = model2.fit()
    print(res2.summary())

    # Extract genus-level contrasts vs Homo sapiens (reference)
    genus_effects = {}
    for param, coef in res2.params.items():
        if param.startswith("C(genus)[T."):
            genus_name = param[len("C(genus)[T.") : -1]
            se = res2.bse[param]
            pval = res2.pvalues[param]
            or_val = float(np.exp(coef))
            ci_low = float(np.exp(coef - 1.96 * se))
            ci_high = float(np.exp(coef + 1.96 * se))
            genus_effects[genus_name] = {
                "coef": float(coef),
                "se": float(se),
                "p": float(pval),
                "or": or_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

    print("\nGenus-level odds ratios vs Homo sapiens (reference):")
    for genus, stats in sorted(genus_effects.items()):
        print(
            f"  {genus:8s}: OR={stats['or']:.3f} "
            f"(95% CI [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]), p={stats['p']:.3g}"
        )

    # Save a compact JSON summary that can help inform the final narrative (not the required conclusion.txt)
    summary = {
        "unadjusted_rates": genus_agg["raw_rate"].to_dict(),
        "human_vs_nonhuman": {
            "coef": float(human_coef),
            "se": float(human_se),
            "p": float(human_p),
            "or": human_or,
            "ci_low": human_ci_low,
            "ci_high": human_ci_high,
            "avg_pred_nonhuman": avg_nonhuman,
            "avg_pred_human": avg_human,
            "avg_pred_diff": diff,
        },
        "genus_effects_vs_human": genus_effects,
    }

    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote model summary statistics to 'analysis_summary.json'.")


if __name__ == "__main__":
    main()

