import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Remove any rows with zero observable sockets just in case
    df = df[df["feature4"] > 0].copy()

    # Proportion of missing teeth within tooth class for each specimen row
    df["prop_missing"] = df["feature3"] / df["feature4"]

    # Binomial regression: AMTL proportion ~ genus + tooth class + age + sex
    # Using Homo sapiens as the baseline genus via C(feature8)
    model = smf.glm(
        formula="prop_missing ~ C(feature8) + C(feature1) + feature5 + feature7",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()

    print(result.summary())

    # Compute predicted probabilities for each genus at average covariate values
    mean_age = df["feature5"].mean()
    mean_sex = df["feature7"].mean()
    ref_tooth_class = df["feature1"].mode()[0]

    print("\nPredicted probability of tooth loss by genus")
    print(f"(Age={mean_age:.2f}, Sex={mean_sex:.2f}, Tooth class={ref_tooth_class})")

    for genus in sorted(df["feature8"].unique()):
        new_obs = pd.DataFrame(
            {
                "feature5": [mean_age],
                "feature7": [mean_sex],
                "feature1": [ref_tooth_class],
                "feature8": [genus],
            }
        )
        pred = result.predict(new_obs)[0]
        print(f"{genus:12s}: predicted AMTL proportion = {pred:.4f}")


if __name__ == "__main__":
    main()

