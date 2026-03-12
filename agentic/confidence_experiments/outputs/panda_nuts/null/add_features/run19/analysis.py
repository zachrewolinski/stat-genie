import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

DATA_PATH = "panda_nuts.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Keep relevant columns and drop rows with missing
    cols = ["chimpanzee", "age", "sex", "help", "nuts_opened", "seconds"]
    df = df[cols].copy()

    # Standardize categories
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["help"] = df["help"].astype(str).str.strip().str.lower()

    # Keep plausible rows
    df = df[df["seconds"].notna() & df["nuts_opened"].notna()]
    df = df[df["seconds"] > 0]

    # Rate of nuts opened per second
    df["rate"] = df["nuts_opened"] / df["seconds"]

    # Encode categories
    df["sex"] = df["sex"].replace({"female": "f", "male": "m"})
    df["help"] = df["help"].replace({"yes": "y", "no": "n"})

    # Filter to expected categories
    df = df[df["sex"].isin(["f", "m"])]
    df = df[df["help"].isin(["y", "n", "n", "N"])]
    df["help"] = df["help"].replace({"n": "n", "N": "n"})

    # Drop missing after cleaning
    df = df.dropna(subset=["age", "sex", "help", "nuts_opened", "seconds"])

    # Summary stats
    summary = {
        "n_rows": int(len(df)),
        "n_chimps": int(df["chimpanzee"].nunique()),
        "rate_mean": float(df["rate"].mean()),
        "rate_median": float(df["rate"].median()),
    }

    # Poisson regression for counts with log(seconds) offset to model rate
    # Using cluster-robust SE by chimpanzee (repeated measures)
    formula = "nuts_opened ~ age + C(sex) + C(help)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"]),
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

    # Likelihood ratio test vs intercept-only model
    null_model = smf.glm(
        formula="nuts_opened ~ 1",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"]),
    ).fit()
    lr_stat = 2 * (result.llf - null_model.llf)
    lr_df = result.df_model
    lr_p = chi2.sf(lr_stat, lr_df)

    # Extract coefficients and p-values
    coef_table = result.summary2().tables[1]
    coef_table = coef_table[["Coef.", "Std.Err.", "z", "P>|z|"]]

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("\nCluster-robust Poisson GLM results (rate via offset):")
    print(coef_table)
    print("\nLR test vs intercept-only: stat=%.3f df=%d p=%.4f" % (lr_stat, lr_df, lr_p))

if __name__ == "__main__":
    main()
