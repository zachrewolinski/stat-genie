import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields if any exist.
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]
    )

    # Proportion of missing teeth within each row and corresponding trial counts.
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Descriptive observed proportions by genus (socket-weighted).
    grouped = (
        df.groupby("genus")
        .apply(lambda g: g["num_amtl"].sum() / g["sockets"].sum())
        .sort_values(ascending=False)
    )

    print("Observed AMTL proportion by genus (socket-weighted):")
    for genus, prop in grouped.items():
        print(f"  {genus}: {prop:.4f}")

    # Binomial regression with Homo sapiens as reference genus, controlling for
    # age, sex (prob_male), and tooth class.
    formula = (
        "amtl_rate ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    result = model.fit()

    print("\nGLM Binomial regression summary:")
    print(result.summary())

    print("\nGenus odds ratios vs Homo sapiens (reference):")
    params = result.params
    cov = result.cov_params()

    genus_effects = {}

    for name, coef in params.items():
        if "genus" in name and "Homo sapiens" not in name:
            or_ = float(np.exp(coef))
            se = float(np.sqrt(cov.loc[name, name]))
            lcl = float(np.exp(coef - 1.96 * se))
            ucl = float(np.exp(coef + 1.96 * se))
            pval = float(result.pvalues[name])
            genus_effects[name] = {
                "coef": float(coef),
                "odds_ratio": or_,
                "ci_lower": lcl,
                "ci_upper": ucl,
                "p_value": pval,
            }
            print(
                f"  {name}: OR={or_:.3f} "
                f"(95% CI {lcl:.3f}-{ucl:.3f}), p={pval:.3g}"
            )

    # Also print a compact JSON summary in case programmatic inspection is useful.
    summary = {
        "observed_amtl_by_genus": grouped.to_dict(),
        "genus_effects_vs_homo": genus_effects,
    }
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

