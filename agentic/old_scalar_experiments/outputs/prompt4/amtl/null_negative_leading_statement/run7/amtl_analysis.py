import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic cleaning: drop rows with missing key fields and ensure positive socket counts
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "tooth_class",
            "genus",
        ]
    )
    df = df[df["sockets"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # AMTL proportion for descriptive summaries and for the binomial GLM
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    print(f"Number of observations after cleaning: {len(df)}")
    print("\nMean AMTL proportion by genus:")
    print(df.groupby("genus")["amtl_prop"].mean())

    # Binomial regression: AMTL proportion as outcome, with sockets as binomial trials
    # Controls: age at death, sex estimate (prob_male), and tooth class.
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("\nBinomial GLM summary (AMTL proportion):")
    print(model.summary())

    # Extract the key effect: human vs non-human primates
    coef = model.params["is_human"]
    conf_int = model.conf_int().loc["is_human"]
    p_val = model.pvalues["is_human"]

    print("\nEffect of being human (is_human = 1 vs 0):")
    print(f"  Coefficient (log-odds scale): {coef:.4f}")
    print(f"  95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
    print(f"  p-value: {p_val:.4g}")

    # Predicted AMTL probabilities for a typical specimen, human vs non-human
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    mode_tooth_class = df["tooth_class"].mode().iloc[0]

    new_data = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_prob_male, mean_prob_male],
            "tooth_class": [mode_tooth_class, mode_tooth_class],
        }
    )
    preds = model.predict(new_data)

    print(
        "\nPredicted AMTL proportion for a typical specimen "
        f"(age={mean_age:.2f}, prob_male={mean_prob_male:.2f}, "
        f"tooth_class='{mode_tooth_class}'):"
    )
    for label, val in zip(["Non-human primate", "Modern human"], preds):
        print(f"  {label}: {val:.4f}")


if __name__ == "__main__":
    main()

