import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Map scrambled column names to their semantic meaning based on info.json
    df["genus_label"] = df["tooth_class"]  # Homo sapiens, Pan, Papio, Pongo
    df["tooth_class_label"] = df["sockets"]  # Anterior, Posterior, Premolar
    df["sex_prob_male"] = df["stdev_age"]  # 0-1 sex estimate
    df["age_at_death"] = df["pop"]  # estimated age at death

    # Counts for binomial model: number of missing teeth vs total observable sockets
    df["num_missing"] = df["genus"]
    df["num_sockets"] = df["age"]

    # Drop rows where counts are inconsistent for a binomial model
    df_valid = df[(df["num_sockets"] > 0) & (df["num_missing"] >= 0) & (df["num_missing"] <= df["num_sockets"])]
    print(f"Original rows: {len(df)}, valid rows for binomial model: {len(df_valid)}")

    # Descriptive summary: overall AMTL frequency by genus
    genus_summary = (
        df_valid.groupby("genus_label")[["num_missing", "num_sockets"]]
        .sum()
        .assign(prop_missing=lambda g: g["num_missing"] / g["num_sockets"])
        .sort_values("prop_missing", ascending=False)
    )
    print("\nAMTL summary by genus (unadjusted):")
    print(genus_summary)

    # Binomial regression: missing vs not missing, controlling for age, sex, and tooth class
    # Encode each observation as [successes, failures]
    y = np.column_stack(
        [df_valid["num_missing"].to_numpy(), (df_valid["num_sockets"] - df_valid["num_missing"]).to_numpy()]
    )

    # Use Homo sapiens as the reference genus; include age, sex, and tooth class
    formula = (
        'C(genus_label, Treatment(reference="Homo sapiens"))'
        " + age_at_death"
        " + sex_prob_male"
        " + C(tooth_class_label)"
    )
    X = patsy.dmatrix(formula, df_valid, return_type="dataframe")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    print("\nBinomial regression summary:")
    print(result.summary())

    coef = result.params
    se = result.bse
    pvals = result.pvalues

    print("\nGenus effects relative to Homo sapiens:")
    for name in X.columns:
        if name.startswith('C(genus_label, Treatment(reference="Homo sapiens"))['):
            est = coef[name]
            se_est = se[name]
            z = est / se_est if se_est != 0 else np.nan
            p = pvals[name]
            ci_low = est - 1.96 * se_est
            ci_high = est + 1.96 * se_est
            odds_ratio = np.exp(est)
            print(
                f"{name}: log-odds diff = {est:.3f}, OR = {odds_ratio:.3f}, "
                f"95% CI [{ci_low:.3f}, {ci_high:.3f}], z = {z:.3f}, p = {p:.3g}"
            )


if __name__ == "__main__":
    main()

