import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth and weights for binomial model
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression using grouped data: response = proportion, weights = number of sockets
    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())

    # Extract and print the human effect coefficient and its confidence interval
    coef = result.params["is_human"]
    conf_int = result.conf_int().loc["is_human"]
    print("\nEffect of being human (log-odds scale):")
    print(f"  Coefficient: {coef:.4f}")
    print(f"  95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

    # Also compute odds ratio and its CI for easier interpretation
    import numpy as np

    or_human = np.exp(coef)
    or_ci_low = np.exp(conf_int[0])
    or_ci_high = np.exp(conf_int[1])
    print("\nEffect of being human (odds ratio):")
    print(f"  OR: {or_human:.3f}")
    print(f"  95% CI: [{or_ci_low:.3f}, {or_ci_high:.3f}]")


if __name__ == "__main__":
    main()

