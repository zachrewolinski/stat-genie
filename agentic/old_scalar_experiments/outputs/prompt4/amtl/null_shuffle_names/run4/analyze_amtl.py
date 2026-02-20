import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename columns to reflect their semantic meaning as described in info.json.
    df = df.rename(
        columns={
            "sockets": "tooth_class",  # Anterior / Posterior / Premolar
            "prob_male": "specimen_id",  # unique specimen identifier
            "genus": "num_missing",  # number of teeth missing in that class
            "age": "num_present",  # number of observable sockets with teeth present
            "pop": "age_years",  # estimated age at death
            "num_amtl": "age_sd",  # uncertainty on age estimate
            "stdev_age": "prob_male",  # probability specimen is male
            "tooth_class": "genus",  # taxonomic genus (Homo sapiens, Pan, Papio, Pongo)
            "specimen": "region",  # geographic region
        }
    )

    # Basic checks.
    print("Unique genera:", df["genus"].unique())
    print("Unique tooth classes:", df["tooth_class"].unique())

    # Construct binomial counts: total potential tooth positions in this class.
    df["total_positions"] = df["num_missing"] + df["num_present"]

    # Drop any clearly invalid rows where the constructed total is non-positive.
    df = df[df["total_positions"] > 0].copy()

    # Missing-tooth frequency for each row.
    df["missing_prop"] = df["num_missing"] / df["total_positions"]

    # Indicator for modern humans vs non-human primates.
    df["is_human"] = df["genus"].str.contains("Homo", case=False, na=False).astype(int)

    # Center and scale continuous covariates for numerical stability.
    df["age_years_c"] = df["age_years"] - df["age_years"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Tooth-class dummies (Anterior / Posterior / Premolar), drop one as reference.
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    # Design matrix.
    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df[["is_human", "age_years_c", "prob_male_c"]],
            tooth_dummies,
        ],
        axis=1,
    )

    # Fit binomial GLM using aggregated proportions with the number of trials as weights.
    model = sm.GLM(
        df["missing_prop"],
        X,
        family=sm.families.Binomial(),
        var_weights=df["total_positions"],
    )
    result = model.fit()

    print("\n==== GLM summary ====")
    print(result.summary())

    # Extract the human effect.
    coef_human = result.params["is_human"]
    se_human = result.bse["is_human"]
    pval_human = result.pvalues["is_human"]

    # Compute predicted AMTL probability for a "typical" specimen:
    # average age, average sex probability, reference tooth class (the dropped category).
    typical = X.mean()
    typical["intercept"] = 1.0

    # Human vs non-human differ only in is_human indicator.
    typical_nonhuman = typical.copy()
    typical_nonhuman["is_human"] = 0.0
    typical_human = typical.copy()
    typical_human["is_human"] = 1.0

    logit_nonhuman = float(np.dot(typical_nonhuman, result.params))
    logit_human = float(np.dot(typical_human, result.params))

    p_nonhuman = 1.0 / (1.0 + np.exp(-logit_nonhuman))
    p_human = 1.0 / (1.0 + np.exp(-logit_human))

    diff = p_human - p_nonhuman
    or_human = float(np.exp(coef_human))

    print("\n==== Effect of being human ====")
    print(f"Coefficient (log-odds): {coef_human:.3f}")
    print(f"Std. error: {se_human:.3f}")
    print(f"P-value: {pval_human:.3g}")
    print(f"Odds ratio (human vs non-human): {or_human:.3f}")
    print(f"Predicted AMTL freq, non-human: {p_nonhuman:.3%}")
    print(f"Predicted AMTL freq, human:     {p_human:.3%}")
    print(f"Absolute difference:            {diff:.3%}")


if __name__ == "__main__":
    main()

