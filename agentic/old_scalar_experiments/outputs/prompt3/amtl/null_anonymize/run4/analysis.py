import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic checks
    df["missing_rate"] = df["feature3"] / df["feature4"]
    print("Head with missing_rate:")
    print(df.head())
    print("\nGenus counts:")
    print(df["feature8"].value_counts())
    print("\nMissing rate by genus (mean):")
    print(df.groupby("feature8")["missing_rate"].mean())

    # Identify and drop invalid binomial rows (where missing teeth exceed sockets)
    invalid = df[df["feature3"] > df["feature4"]]
    print(f"\nRows with missing > sockets: {len(invalid)}")
    if not invalid.empty:
        print("Example invalid rows:")
        print(invalid.head())

    df_valid = df[df["feature3"] <= df["feature4"]].copy()
    print(f"\nRows retained for modeling: {len(df_valid)} (out of {len(df)})")

    # Binomial regression: missing proportion with trials as frequency weights
    # Adjust for age (feature5), sex estimate (feature7), and tooth class (feature1)
    # Genus (feature8) is the main predictor of interest.
    df_valid["prop_missing"] = df_valid["feature3"] / df_valid["feature4"]

    formula = "prop_missing ~ feature5 + feature7 + C(feature1) + C(feature8)"
    print("\nFitting GLM Binomial with formula:")
    print(formula)

    model = smf.glm(
        formula=formula,
        data=df_valid,
        family=sm.families.Binomial(),
        freq_weights=df_valid["feature4"],
    ).fit()

    print("\nModel summary:")
    print(model.summary())

    # Compute predicted missing probabilities by genus at mean covariates
    cov_means = {
        "feature5": df_valid["feature5"].mean(),
        "feature7": df_valid["feature7"].mean(),
    }

    # Use the most common tooth class as reference for comparison
    common_tooth_class = df_valid["feature1"].mode().iloc[0]
    print(f"\nUsing tooth class for prediction: {common_tooth_class}")

    genera = df_valid["feature8"].unique()
    print("\nPredicted missing probabilities by genus (at mean age/sex, common tooth class):")
    for genus in sorted(genera):
        pred_df = pd.DataFrame(
            {
                "feature5": [cov_means["feature5"]],
                "feature7": [cov_means["feature7"]],
                "feature1": [common_tooth_class],
                "feature8": [genus],
            }
        )
        pred_prob = model.predict(pred_df)[0]
        print(f"{genus}: {pred_prob:.4f}")

    # Additional model with explicit human indicator vs all non-human primates
    df_valid["is_human"] = (df_valid["feature8"] == "Homo sapiens").astype(int)
    formula_human = "prop_missing ~ feature5 + feature7 + C(feature1) + is_human"
    print("\nFitting GLM Binomial with human vs non-human indicator:")
    print(formula_human)

    model_human = smf.glm(
        formula=formula_human,
        data=df_valid,
        family=sm.families.Binomial(),
        freq_weights=df_valid["feature4"],
    ).fit()

    print("\nHuman vs non-human model summary:")
    print(model_human.summary())

    # Predicted probabilities for human vs non-human at mean covariates and common tooth class
    pred_human = model_human.predict(
        pd.DataFrame(
            {
                "feature5": [cov_means["feature5"]],
                "feature7": [cov_means["feature7"]],
                "feature1": [common_tooth_class],
                "is_human": [1],
            }
        )
    )[0]
    pred_nonhuman = model_human.predict(
        pd.DataFrame(
            {
                "feature5": [cov_means["feature5"]],
                "feature7": [cov_means["feature7"]],
                "feature1": [common_tooth_class],
                "is_human": [0],
            }
        )
    )[0]
    print(
        f"\nPredicted missing probability, human: {pred_human:.4f}; "
        f"non-human (average): {pred_nonhuman:.4f}; "
        f"difference (human - non-human): {pred_human - pred_nonhuman:.4f}"
    )


if __name__ == "__main__":
    main()
