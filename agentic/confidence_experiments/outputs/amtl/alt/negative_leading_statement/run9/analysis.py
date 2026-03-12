import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Construct proportion of antemortem tooth loss
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Treat categorical predictors explicitly
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Fit binomial GLM with logit link.
    # Use sockets as frequency weights so that prop_amtl is modeled as a binomial proportion.
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial GLM for AMTL proportion")
    print(result.summary())

    # Compute genus-specific predicted probabilities at average covariates
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    ref_tooth_class = df["tooth_class"].mode().iat[0]

    pred_df = pd.DataFrame(
        {
            "genus": df["genus"].cat.categories,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": ref_tooth_class,
        }
    )
    pred_df["predicted_prop"] = result.predict(pred_df)

    print("\nPredicted AMTL proportion by genus")
    print(pred_df)


if __name__ == "__main__":
    main()

