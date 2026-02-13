import pandas as pd
import statsmodels.api as sm
import patsy
import numpy as np


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename based on metadata for clarity.
    df = df.rename(columns={"sockets": "tooth_class", "tooth_class": "genus_name"})

    df["missing"] = df["genus"].astype(float)
    df["sockets_n"] = df["age"].astype(float)

    valid = (df["sockets_n"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets_n"])
    df = df.loc[valid].copy()

    df["present"] = df["sockets_n"] - df["missing"]
    df["age_at_death"] = df["pop"].astype(float)
    df["prob_male"] = df["stdev_age"].astype(float)

    # Binary indicator: modern human vs non-human primate.
    df["is_human"] = (df["genus_name"] == "Homo sapiens").astype(int)

    # Descriptive comparison of raw proportions.
    for label, subset in [("Human", df[df["is_human"] == 1]), ("Non-human", df[df["is_human"] == 0])]:
        missing_total = subset["missing"].sum()
        sockets_total = subset["sockets_n"].sum()
        prop = missing_total / sockets_total
        print(f"{label} AMTL: missing={missing_total}, sockets={sockets_total}, proportion={prop:.4f}")

    # Binomial regression with human indicator.
    formula = "missing + present ~ is_human + age_at_death + prob_male + C(tooth_class)"
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()

    print("\nHuman vs non-human model summary:")
    print(result.summary())

    # Odds ratio for being human vs non-human.
    params = result.params
    conf_int = result.conf_int()

    or_human = float(np.exp(params["is_human"]))
    ci_low, ci_high = np.exp(conf_int.loc["is_human"])
    print(f"\nOdds ratio (human vs non-human): {or_human:.3f} "
          f"(95% CI {ci_low:.3f}–{ci_high:.3f})")


if __name__ == "__main__":
    main()

