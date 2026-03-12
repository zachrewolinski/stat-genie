import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename key columns for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "n_missing",
            "feature4": "n_sockets",
            "feature5": "age",
            "feature7": "sex_estimate",
            "feature8": "genus",
        }
    )

    # Keep only rows with at least one observable socket
    df = df[df["n_sockets"] > 0].copy()

    # Clamp any impossible counts where missing teeth exceed observable sockets
    n_problem = (df["n_missing"] > df["n_sockets"]).sum()
    if n_problem > 0:
        print(f"Clamping {n_problem} rows where n_missing > n_sockets.")
        df["n_missing"] = np.minimum(df["n_missing"], df["n_sockets"])

    # AMTL rate per row
    df["amtl_rate"] = df["n_missing"] / df["n_sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = df["genus"].str.contains("Homo", case=False).astype(int)

    # Covariates
    df["age"] = df["age"].astype(float)
    df["sex_estimate"] = df["sex_estimate"].astype(float)
    df["tooth_class"] = df["tooth_class"].astype("category")

    print("AMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].agg(["mean", "std", "count"]))
    print()

    # Binomial regression: AMTL rate with binomial family, weighted by number of sockets
    y, X = patsy.dmatrices(
        "amtl_rate ~ is_human + age + sex_estimate + C(tooth_class)",
        df,
        return_type="dataframe",
    )
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=df["n_sockets"])
    result = model.fit()

    print(result.summary())

    coef = result.params["is_human"]
    se = result.bse["is_human"]
    odds_ratio = float(np.exp(coef))
    ci_lower = float(np.exp(coef - 1.96 * se))
    ci_upper = float(np.exp(coef + 1.96 * se))
    p_value = float(result.pvalues["is_human"])

    print()
    print("Effect of being human (vs non-human primates):")
    print(f"  Log-odds coefficient: {coef:.4f}")
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print(f"  95% CI for OR: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  p-value: {p_value:.4g}")


if __name__ == "__main__":
    main()

