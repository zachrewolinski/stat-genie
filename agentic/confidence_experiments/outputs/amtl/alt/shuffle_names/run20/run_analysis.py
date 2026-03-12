import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Derived variables based on metadata interpretation
    # genus: number of missing teeth of given class
    # age: number of observable sockets that could be scored
    # pop: estimated age at death
    # stdev_age: estimate of sex (probability specimen is male)
    # sockets: tooth class (Anterior, Posterior, Premolar)
    # tooth_class: specimen genus (Homo sapiens, Pan, Papio, Pongo)

    # Filter to rows where total observable sockets is positive
    df = df[df["age"] > 0].copy()

    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["tooth_class"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth (AMTL frequency)
    df["prop_missing"] = df["genus"] / df["age"]

    # Fit a binomial GLM with aggregated binomial data:
    # response = proportion missing, with weights = number of trials (observable sockets)
    formula = "prop_missing ~ is_human + pop + stdev_age + C(sockets)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["age"],
    )
    result = model.fit()

    print(result.summary())

    # Compute odds ratio and 95% CI for the human indicator
    coef = float(result.params["is_human"])
    se = float(result.bse["is_human"])
    z_crit = 1.96
    or_est = float(np.exp(coef))
    or_low = float(np.exp(coef - z_crit * se))
    or_high = float(np.exp(coef + z_crit * se))

    print("\nEffect of being human (Homo sapiens) vs non-human primates:")
    print(f"  log-odds coefficient: {coef:.4f}")
    print(f"  standard error:       {se:.4f}")
    print(f"  Wald z-value:         {coef / se:.4f}")
    print(f"  odds ratio:           {or_est:.3f}")
    print(f"  95% CI for OR:        [{or_low:.3f}, {or_high:.3f}]")


if __name__ == "__main__":
    main()
