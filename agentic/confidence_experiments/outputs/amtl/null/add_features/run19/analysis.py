import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Ensure numeric types
    df["num_amtl"] = pd.to_numeric(df["num_amtl"], errors="coerce")
    df["sockets"] = pd.to_numeric(df["sockets"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["prob_male"] = pd.to_numeric(df["prob_male"], errors="coerce")

    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male"])

    # Remove rows with non-positive socket counts (should not occur, but be safe)
    df = df[df["sockets"] > 0]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    print("Unique genera and counts:")
    print(df["genus"].value_counts())
    print()

    print("Summary of key variables:")
    print(df[["prop_amtl", "age", "prob_male", "tooth_class", "is_human"]].describe(include="all"))
    print()

    # Binomial regression: AMTL proportion ~ human vs non-human + age + sex + tooth class
    # Use sockets as binomial trial counts via frequency weights.
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    result = model.fit()

    print("Binomial GLM results (AMTL proportion):")
    print(result.summary())
    print()

    # Extract effect of being human
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]

    # 95% Wald CI
    z_crit = 1.96
    ci_low = coef - z_crit * se
    ci_high = coef + z_crit * se

    odds_ratio = float(np.exp(coef))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    print("Effect of being modern human (Homo sapiens) relative to non-human primates:")
    print(f"  Log-odds coefficient: {coef:.3f} (SE = {se:.3f}, p = {pval:.3g})")
    print(f"  95% CI (log-odds): [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  Odds ratio: {odds_ratio:.3f}  (95% CI: [{or_low:.3f}, {or_high:.3f}])")

    # Predicted AMTL proportions for a typical individual by genus
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    common_tooth_class = df["tooth_class"].mode().iat[0]

    print()
    print("Predicted AMTL proportion at mean age/sex for common tooth class:")
    for is_human in [0, 1]:
        design = pd.DataFrame(
            {
                "is_human": [is_human],
                "age": [mean_age],
                "prob_male": [mean_prob_male],
                "tooth_class": [common_tooth_class],
            }
        )
        pred = result.predict(design)[0]
        label = "Non-human primates (Pan/Pongo/Papio)" if is_human == 0 else "Modern humans (Homo sapiens)"
        print(f"  {label}: predicted AMTL proportion = {pred:.3f}")


if __name__ == "__main__":
    main()
