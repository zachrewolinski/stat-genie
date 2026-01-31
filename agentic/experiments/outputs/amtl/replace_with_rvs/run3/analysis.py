import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def main():
    df = pd.read_csv("amtl.csv")

    # Basic cleaning
    df = df.copy()
    df = df[df["sockets"].notna() & df["num_amtl"].notna()]
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Binary indicator for modern humans vs non-human primates
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Drop rows with missing covariates
    df = df.dropna(subset=["age", "prob_male", "tooth_class", "human"])

    # Build design matrix
    X = patsy.dmatrix(
        "human + age + prob_male + C(tooth_class)",
        df,
        return_type="dataframe",
    )

    # Binomial endog with successes and failures
    endog = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    coef = result.params.get("human", np.nan)
    pval = result.pvalues.get("human", np.nan)
    odds_ratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan

    print(result.summary())
    print("\nHuman effect (log-odds):", coef)
    print("Human effect p-value:", pval)
    print("Human odds ratio:", odds_ratio)

    # Save key results for conclusion
    with open("analysis_results.txt", "w", encoding="utf-8") as f:
        f.write(f"coef_human\t{coef}\n")
        f.write(f"pval_human\t{pval}\n")
        f.write(f"odds_ratio_human\t{odds_ratio}\n")
        f.write(f"n_obs\t{len(df)}\n")


if __name__ == "__main__":
    main()
