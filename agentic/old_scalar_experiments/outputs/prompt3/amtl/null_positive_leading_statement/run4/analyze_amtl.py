import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic cleaning / derived variables
    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of antemortem tooth loss per row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Descriptive genus-level AMTL frequencies
    genus_summary = (
        df.groupby("genus", as_index=False)
        .agg(
            total_amtl=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
            mean_age=("age", "mean"),
        )
    )
    genus_summary["amtl_rate"] = genus_summary["total_amtl"] / genus_summary["total_sockets"]

    print("Genus-level AMTL rates (num_amtl / sockets):")
    print(genus_summary.to_string(index=False))
    print()

    # Binomial regression: AMTL counts given number of sockets
    # Model AMTL as a function of human vs non-human, age at death, estimated sex, and tooth class.
    # We treat prop_amtl as the response with sockets as the number of trials.
    model_formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"

    glm_binom = smf.glm(
        formula=model_formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = glm_binom.fit()

    print("Binomial regression results:")
    print(result.summary())
    print()

    # Extract key statistics for the human effect
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]
    oratio = float(np.exp(coef))

    conf_int = result.conf_int().loc["is_human"].tolist()
    or_ci = list(np.exp(conf_int))

    print("Human (Homo sapiens) effect on AMTL frequency (vs non-human primates):")
    print(f"  Log-odds coefficient: {coef:.3f} (SE = {se:.3f}, p = {pval:.4g})")
    print(f"  Odds ratio: {oratio:.3f}")
    print(f"  95% CI for odds ratio: [{or_ci[0]:.3f}, {or_ci[1]:.3f}]")


if __name__ == "__main__":
    main()
