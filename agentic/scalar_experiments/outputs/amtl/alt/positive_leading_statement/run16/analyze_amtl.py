import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Proportion of missing teeth out of observable sockets
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression with logit link, using sockets as binomial denominators
    formula = (
        "prop_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    )

    result = model.fit()

    print(result.summary())

    # Compute genus-specific predicted AMTL probabilities at mean covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    # Use the most common tooth_class as reference scenario
    mode_tooth_class = df["tooth_class"].mode().iat[0]

    genera = sorted(df["genus"].unique())
    print("\nPredicted AMTL probabilities by genus (at mean covariates):")
    for g in genera:
        row = {
            "genus": g,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": mode_tooth_class,
        }
        pred = result.predict(pd.DataFrame([row]))[0]
        print(f"{g}: {pred:.4f}")


if __name__ == "__main__":
    main()

