import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "observed",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    df = df.dropna(
        subset=["missing", "observed", "age", "sex", "tooth_class", "genus"]
    ).copy()
    df = df[df["observed"] > 0].copy()
    df["missing"] = df["missing"].clip(lower=0)
    df["prop"] = df["missing"] / df["observed"]
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    model = smf.glm(
        "prop ~ human + age + sex + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observed"],
    ).fit()

    print(model.summary())

    coef = model.params["human"]
    se = model.bse["human"]
    pval = model.pvalues["human"]
    ci_low, ci_high = model.conf_int().loc["human"]

    or_est = float(np.exp(coef))
    or_low, or_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    print(
        "HUMAN_COEF={:.6f} SE={:.6f} p={:.6g}".format(
            float(coef), float(se), float(pval)
        )
    )
    print("HUMAN_OR={:.6f} OR_CI=({:.6f}, {:.6f})".format(or_est, or_low, or_high))


if __name__ == "__main__":
    main()
