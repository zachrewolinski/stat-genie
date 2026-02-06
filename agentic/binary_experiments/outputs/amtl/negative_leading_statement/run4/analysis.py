import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


def main():
    df = pd.read_csv("amtl.csv")

    # Prepare variables
    df = df.copy()
    df["is_homo"] = (df["genus"] == "Homo sapiens").astype(int)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Drop rows with missing values in required fields
    cols = ["amtl_rate", "sockets", "is_homo", "age", "prob_male", "tooth_class"]
    df = df.dropna(subset=cols)

    # Binomial GLM with logit link, using sockets as binomial trials
    model = smf.glm(
        formula="amtl_rate ~ is_homo + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = model.params["is_homo"]
    pval = model.pvalues["is_homo"]
    odds_ratio = float(np.exp(coef))

    # Adjusted predicted mean AMTL rate for homo vs non-homo
    df_h = df.copy()
    df_h["is_homo"] = 1
    df_nh = df.copy()
    df_nh["is_homo"] = 0
    pred_h = model.predict(df_h).mean()
    pred_nh = model.predict(df_nh).mean()

    print("GLM Binomial (AMTL rate ~ is_homo + age + prob_male + tooth_class)")
    print(f"is_homo coef (log-odds): {coef:.4f}")
    print(f"is_homo odds ratio: {odds_ratio:.4f}")
    print(f"is_homo p-value: {pval:.6f}")
    print(f"Adjusted mean AMTL rate if homo: {pred_h:.4f}")
    print(f"Adjusted mean AMTL rate if non-homo: {pred_nh:.4f}")


if __name__ == "__main__":
    main()
