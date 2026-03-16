import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = "panda_nuts.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.copy()
    df = df.dropna(subset=["nuts_opened", "seconds", "age", "sex", "help", "chimpanzee"])
    df = df[df["seconds"] > 0]

    # Efficiency as rate
    df["rate"] = df["nuts_opened"] / df["seconds"]

    # Poisson GLM with offset for exposure (seconds)
    # Model the rate of nuts_opened per second as a function of age, sex, and help
    poisson_model = smf.glm(
        formula="nuts_opened ~ age + C(sex) + C(help)",
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["seconds"])
    ).fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

    # OLS on rate as a robustness check
    ols_model = smf.ols(
        formula="rate ~ age + C(sex) + C(help)",
        data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})

    # Collect results
    results = {
        "n_rows": int(df.shape[0]),
        "n_chimps": int(df["chimpanzee"].nunique()),
        "poisson": {
            "params": poisson_model.params.to_dict(),
            "pvalues": poisson_model.pvalues.to_dict(),
            "rr": {k: float(np.exp(v)) for k, v in poisson_model.params.to_dict().items()},
        },
        "ols": {
            "params": ols_model.params.to_dict(),
            "pvalues": ols_model.pvalues.to_dict(),
        },
        "rate_summary": {
            "mean": float(df["rate"].mean()),
            "std": float(df["rate"].std(ddof=1)),
            "min": float(df["rate"].min()),
            "max": float(df["rate"].max()),
        },
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
