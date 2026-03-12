import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize categorical values
    df["sex"] = df["sex"].astype("category")
    df["help"] = df["help"].astype("category")
    return df


def fit_poisson(df: pd.DataFrame):
    formula = "nuts_opened ~ age + C(sex) + C(help)"
    offset = np.log(df["seconds"])
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=offset,
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})
    return result


def fit_negbin(df: pd.DataFrame):
    formula = "nuts_opened ~ age + C(sex) + C(help)"
    offset = np.log(df["seconds"])
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=offset,
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["chimpanzee"]})
    return result


def main():
    df = load_data("panda_nuts.csv")

    poisson_res = fit_poisson(df)
    overdispersion = poisson_res.pearson_chi2 / poisson_res.df_resid

    negbin_res = None
    if overdispersion > 1.5:
        negbin_res = fit_negbin(df)

    output = {
        "n": int(df.shape[0]),
        "num_chimps": int(df["chimpanzee"].nunique()),
        "overdispersion": float(overdispersion),
        "poisson": {
            "params": poisson_res.params.to_dict(),
            "pvalues": poisson_res.pvalues.to_dict(),
            "aic": float(poisson_res.aic),
        },
        "negbin": None,
    }

    if negbin_res is not None:
        output["negbin"] = {
            "params": negbin_res.params.to_dict(),
            "pvalues": negbin_res.pvalues.to_dict(),
            "aic": float(negbin_res.aic),
        }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
