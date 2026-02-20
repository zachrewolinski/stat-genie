import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Proportion of missing teeth for each row
    df = df.copy()
    df["missing_prop"] = df["feature3"] / df["feature4"]

    # Binomial regression: missing teeth (as proportion, weighted by number of sockets)
    formula = "missing_prop ~ C(feature8) + feature5 + feature7 + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    )
    result = model.fit()

    print(result.summary())

    # Mean observed and model-predicted AMTL by genus
    df["pred_prob"] = result.predict()
    print("\nMean observed missing proportion by genus:")
    print(df.groupby("feature8")["missing_prop"].mean())

    print("\nMean model-predicted AMTL probability by genus:")
    print(df.groupby("feature8")["pred_prob"].mean())


if __name__ == "__main__":
    main()

