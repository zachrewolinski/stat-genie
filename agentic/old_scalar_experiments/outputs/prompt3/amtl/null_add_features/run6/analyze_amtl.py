import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Keep only relevant genera: humans vs non-human primates of interest.
    target_genera = {"Homo sapiens", "Pan", "Pongo", "Papio"}
    df = df[df["genus"].isin(target_genera)].copy()

    # Binary indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Guard against invalid values.
    df = df[df["sockets"] > 0].copy()
    df = df[(df["num_amtl"] >= 0) & (df["num_amtl"] <= df["sockets"])].copy()

    # Binomial regression: AMTL proportion as a function of human status,
    # age, sex (prob_male), and tooth class.
    # Use proportion with frequency weights equal to the number of sockets.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    formula = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    result = model.fit()

    # Print key results for inspection when running the script.
    print("Model formula:", formula)
    print("\nCoefficients:")
    print(result.params)
    print("\nStandard errors:")
    print(result.bse)
    print("\nP-values:")
    print(result.pvalues)
    print("\nIs_human coefficient (effect on log-odds of AMTL):")
    beta_human = result.params["is_human"]
    se_human = result.bse["is_human"]
    p_human = result.pvalues["is_human"]
    print(f"  beta_human = {beta_human:.4f}")
    print(f"  se_human   = {se_human:.4f}")
    print(f"  p_human    = {p_human:.4g}")
    print(f"  odds_ratio = {float(np.exp(beta_human)):.4f}")

    # Descriptive AMTL rates by genus and human vs non-human.
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]
    print("\nMean AMTL proportion by genus:")
    print(df.groupby("genus")["prop_amtl"].mean())
    print("\nMean AMTL proportion by human status (0=non-human, 1=human):")
    print(df.groupby("is_human")["prop_amtl"].mean())


if __name__ == "__main__":
    main()
