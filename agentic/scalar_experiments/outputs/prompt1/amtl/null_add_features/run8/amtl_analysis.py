import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Create derived variables
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial regression: AMTL proportion with sockets as binomial denominator
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Print a concise summary focused on the human effect
    print(result.summary())
    print("\nCoefficient for is_human (Homo sapiens vs non-human):")
    print(result.params["is_human"])
    print("Std. error:", result.bse["is_human"])
    print("z-value:", result.tvalues["is_human"])
    print("p-value:", result.pvalues["is_human"])


if __name__ == "__main__":
    main()

