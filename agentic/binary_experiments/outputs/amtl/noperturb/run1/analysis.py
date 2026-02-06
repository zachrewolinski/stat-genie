import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("amtl.csv")

    # Binary indicator: modern human vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of AMTL with binomial trials = sockets
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Fit binomial GLM with logit link
    model = smf.glm(
        "amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print(model.summary())
    print("\nCoefficient for is_human:")
    print(model.params["is_human"], "p=", model.pvalues["is_human"])


if __name__ == "__main__":
    main()
