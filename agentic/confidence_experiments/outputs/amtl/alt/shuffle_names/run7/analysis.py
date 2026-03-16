import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Map columns to their semantic meanings based on info.json
    # sockets       -> tooth class within the mouth (Anterior/Posterior/Premolar)
    # prob_male     -> specimen identifier
    # genus         -> number of missing teeth of that class (AMTL count)
    # age           -> number of observable sockets scored (binomial denominator)
    # pop           -> estimated age at death
    # num_amtl      -> uncertainty of age estimate (not used in model)
    # stdev_age     -> sex estimate (numeric proxy)
    # tooth_class   -> genus (Homo sapiens, Pan, Papio, Pongo)
    # specimen      -> population/region label

    df = df.copy()
    df["tooth_class_cat"] = df["sockets"]
    df["specimen_id"] = df["prob_male"]
    df["amtl_missing"] = df["genus"].astype(float)
    df["n_sockets"] = df["age"].astype(float)
    df["age_years"] = df["pop"].astype(float)
    df["age_uncertainty"] = df["num_amtl"].astype(float)
    df["sex_est"] = df["stdev_age"].astype(float)
    df["genus_name"] = df["tooth_class"]

    # Remove any rows with non-positive socket counts, if present
    df = df[df["n_sockets"] > 0].copy()

    # Response as proportion with binomial weights
    df["prop_amtl"] = df["amtl_missing"] / df["n_sockets"]

    # Indicator for whether specimen is modern human
    df["is_human"] = (df["genus_name"] == "Homo sapiens").astype(int)

    # Fit binomial GLM: AMTL frequency ~ human vs non-human + age + sex + tooth class
    formula = "prop_amtl ~ is_human + age_years + sex_est + C(tooth_class_cat)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )
    result = model.fit()

    # Print summary and key coefficients for inspection
    print(result.summary())
    print("\nKey coefficient for research question:")
    print("is_human coefficient:", result.params.get("is_human"))
    print("is_human std err    :", result.bse.get("is_human"))
    print("is_human z-value    :", result.tvalues.get("is_human"))
    print("is_human p-value    :", result.pvalues.get("is_human"))

    # Compute odds ratio for humans vs non-humans
    coef_human = result.params.get("is_human")
    if coef_human is not None:
        odds_ratio = float(np.exp(coef_human))
        print("is_human odds ratio :", odds_ratio)


if __name__ == "__main__":
    main()
