import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep only target genera
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Basic data filtering
    df = df[df["sockets"] > 0].copy()
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class"])

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial model: AMTL proportion with trial weights = sockets
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Print core results for human indicator and fitted probabilities
    print("GLM coefficients:")
    print(result.params)
    print("\nGLM p-values:")
    print(result.pvalues)

    # Predicted AMTL probabilities for human vs non-human at mean covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    mode_tooth_class = df["tooth_class"].mode().iat[0]

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [mode_tooth_class, mode_tooth_class],
        }
    )
    predictions = result.get_prediction(new_data).predicted_mean

    print("\nPredicted AMTL proportion at mean covariates")
    print(f"Non-human primate: {predictions[0]:.4f}")
    print(f"Modern human:      {predictions[1]:.4f}")

    # Genus-level AMTL rates (total missing / total sockets)
    genus_summary = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(amtl_rate=lambda g: g["num_amtl"] / g["sockets"])
    )
    print("\nGenus-level AMTL rates (num_amtl / sockets):")
    print(genus_summary)


if __name__ == "__main__":
    main()
