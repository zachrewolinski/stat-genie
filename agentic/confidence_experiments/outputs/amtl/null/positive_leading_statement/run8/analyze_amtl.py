import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])

    # Create human indicator (Homo sapiens vs all non-human genera)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Response as proportion with binomial denominator
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Fit GLM with binomial family, weighting by number of sockets
    formula = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract coefficient for human indicator
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    pval = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    # Compute illustrative predicted probabilities at median covariate values
    median_age = df["age"].median()
    median_prob_male = df["prob_male"].median()

    # Use the most common tooth class for a simple comparison
    common_tooth_class = df["tooth_class"].mode().iat[0]

    base_row = {
        "age": median_age,
        "prob_male": median_prob_male,
        "tooth_class": common_tooth_class,
    }

    new_data = pd.DataFrame(
        [
            {**base_row, "is_human": 0},
            {**base_row, "is_human": 1},
        ]
    )
    linpred = result.predict(new_data, linear=True)
    probs = 1.0 / (1.0 + np.exp(-linpred))

    print("=== Binomial GLM: AMTL proportion ~ human + age + sex + tooth_class ===")
    print(result.summary())
    print()
    print("Human indicator (is_human) coefficient:")
    print(f"  coef = {coef:.4f}, se = {se:.4f}, p = {pval:.4g}, OR = {odds_ratio:.3f}")
    print()
    print("Illustrative predicted AMTL probabilities (per socket):")
    print(f"  Non-human (is_human=0): {probs.iloc[0]:.4f}")
    print(f"  Human     (is_human=1): {probs.iloc[1]:.4f}")


if __name__ == "__main__":
    main()
