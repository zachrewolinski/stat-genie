import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename columns to more descriptive names for internal use
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature7": "sex_score",
            "feature8": "genus",
        }
    )

    # Basic derived variables
    df["n_missing"] = df["n_missing"].astype(float)
    df["n_sockets"] = df["n_sockets"].astype(float)
    df = df[df["n_sockets"] > 0].copy()
    df["prop_missing"] = df["n_missing"] / df["n_sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Weighted summary of AMTL by human status
    summary = (
        df.groupby("is_human")[["n_missing", "n_sockets"]]
        .sum()
        .assign(prop_missing=lambda d: d["n_missing"] / d["n_sockets"])
    )

    print("Weighted AMTL proportion by human status (0=non-human, 1=human):")
    print(summary)
    print()

    # Fit a binomial regression model for AMTL, controlling for age, sex, and tooth class.
    # We treat prop_missing as a binomial proportion with n_sockets trials.
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_score + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["n_sockets"],
    )

    result = model.fit()

    print("Binomial regression results:")
    print(result.summary())
    print()

    coef_human = result.params.get("is_human", np.nan)
    p_human = result.pvalues.get("is_human", np.nan)
    odds_ratio_human = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    print(f"Coefficient for is_human: {coef_human:.4f}")
    print(f"Odds ratio for is_human: {odds_ratio_human:.4f}")
    print(f"P-value for is_human: {p_human:.4g}")
    print()

    # Predicted probabilities at typical covariate values for humans vs non-humans
    mean_age = df["age"].mean()
    mean_sex = df["sex_score"].mean()
    common_class = df["tooth_class"].mode().iloc[0]

    design = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "sex_score": [mean_sex, mean_sex],
            "tooth_class": [common_class, common_class],
        }
    )

    pred_probs = result.predict(design)
    print("Predicted AMTL probabilities at typical covariates:")
    print(design.assign(predicted_prop_missing=pred_probs.values))


if __name__ == "__main__":
    main()

