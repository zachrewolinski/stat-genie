import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Drop any rows with missing key variables
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"]
    ).copy()

    # Create helper columns
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    print(f"Rows after dropna: {len(df)}")

    # Descriptive genus-level AMTL proportions
    genus_agg = df.groupby("genus")[["num_amtl", "sockets"]].sum()
    genus_agg["prop_amtl"] = genus_agg["num_amtl"] / genus_agg["sockets"]
    print("\nGenus-level AMTL proportions (num_amtl / sockets):")
    print(genus_agg)

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    try:
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.Binomial(),
            freq_weights=df["sockets"],
        )
    except TypeError:
        # Fallback in case this statsmodels version expects `weights` instead
        model = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.Binomial(),
            weights=df["sockets"],
        )

    result = model.fit()
    print("\nGLM Binomial results with is_human:")
    print(result.summary())

    # Effect of being human (vs non-human primate)
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]
    ci_low, ci_high = result.conf_int().loc["is_human"]

    print("\nEffect of being modern human (is_human):")
    print(f"  Coefficient (log-odds): {coef:.4f}")
    print(f"  Std. error: {se:.4f}")
    print(f"  p-value: {pval:.4g}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Odds ratio: {np.exp(coef):.3f}")
    print(
        "  Odds ratio 95% CI: "
        f"[{np.exp(ci_low):.3f}, {np.exp(ci_high):.3f}]"
    )

    # Predicted probabilities at mean covariates, by tooth class and human status
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    print(
        "\nPredicted AMTL probability at mean age and sex, "
        "by tooth_class and human status:"
    )
    for tooth_class in sorted(df["tooth_class"].unique()):
        design = pd.DataFrame(
            {
                "is_human": [0, 1],
                "age": [mean_age, mean_age],
                "prob_male": [mean_prob_male, mean_prob_male],
                "tooth_class": [tooth_class, tooth_class],
            }
        )
        preds = result.predict(design)
        non_human = preds.iloc[0]
        human = preds.iloc[1]
        print(
            f"  Tooth class {tooth_class}: "
            f"non-human={non_human:.3f}, human={human:.3f}, "
            f"diff={human - non_human:.3f}"
        )


if __name__ == "__main__":
    main()

