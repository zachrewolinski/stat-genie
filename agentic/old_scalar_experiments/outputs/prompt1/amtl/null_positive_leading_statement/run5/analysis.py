import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Construct AMTL proportion for binomial modeling with frequency weights
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Simple descriptive AMTL rates by genus (unadjusted)
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(amtl_rate=lambda g: g["num_amtl"] / g["sockets"])
    )

    print("Unadjusted AMTL rate by genus (num_amtl / sockets):")
    print(genus_summary.to_string())
    print()

    # Fit binomial regression:
    #   logit(p(AMTL)) = genus + age + prob_male + tooth_class
    # Genus is categorical with Homo sapiens as baseline (alphabetically first),
    # so coefficients for Pan, Papio, and Pongo describe log-odds differences
    # relative to modern humans after adjusting for covariates.
    model = smf.glm(
        formula="amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    result = model.fit()

    print(result.summary())

    # Compute predicted AMTL probabilities by genus at mean covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    # Use the most common tooth class as a reference
    tooth_mode = df["tooth_class"].mode(dropna=True)[0]

    new_data = pd.DataFrame(
        {
            "genus": ["Homo sapiens", "Pan", "Papio", "Pongo"],
            "age": [mean_age] * 4,
            "prob_male": [mean_prob_male] * 4,
            "tooth_class": [tooth_mode] * 4,
        }
    )

    pred = result.get_prediction(new_data)
    pred_df = pred.summary_frame(alpha=0.05)[["mean", "mean_ci_lower", "mean_ci_upper"]]
    pred_df.insert(0, "genus", new_data["genus"])

    print("\nPredicted AMTL probability by genus at mean covariates")
    print(pred_df.to_string(index=False))


if __name__ == "__main__":
    main()
