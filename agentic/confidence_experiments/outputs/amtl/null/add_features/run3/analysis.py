import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Restrict to humans and the three non-human primate genera mentioned
    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    # Basic sanity check: drop rows with missing key fields
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Ensure sockets are positive
    df = df[df["sockets"] > 0].copy()

    # Create failures column for binomial model
    df["failures"] = df["sockets"] - df["num_amtl"]

    # Indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Treat tooth_class as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Fit binomial GLM: AMTL as function of human status, age, sex (prob_male), and tooth class
    formula = "num_amtl + failures ~ is_human + age + prob_male + tooth_class"
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())

    # Cluster-robust standard errors by specimen to account for repeated measures
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    coef = result.params["is_human"]
    se = result.bse["is_human"]
    z_value = coef / se if se != 0 else np.nan
    p_value = result.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    # Compute a simple effect on predicted probabilities at mean covariates, Posterior teeth
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    base_tooth = "Posterior" if "Posterior" in df["tooth_class"].cat.categories else df["tooth_class"].cat.categories[0]

    new_data_nonhuman = pd.DataFrame(
        {
            "is_human": [0],
            "age": [mean_age],
            "prob_male": [mean_prob_male],
            "tooth_class": [base_tooth],
        }
    )
    new_data_human = new_data_nonhuman.copy()
    new_data_human["is_human"] = 1

    pred_nonhuman = float(result.predict(new_data_nonhuman)[0])
    pred_human = float(result.predict(new_data_human)[0])

    summary = {
        "n_rows": int(len(df)),
        "genera_counts": df["genus"].value_counts().to_dict(),
        "coef_is_human": float(coef),
        "se_is_human": float(se),
        "z_is_human": float(z_value),
        "p_is_human": float(p_value),
        "odds_ratio_is_human": odds_ratio,
        "pred_prob_nonhuman": pred_nonhuman,
        "pred_prob_human": pred_human,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

