import json
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
            "feature4": "total",
            "feature5": "age",
            "feature6": "age_uncert",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Keep only relevant columns and valid rows
    df = df[["tooth_class", "missing", "total", "age", "sex", "genus"]].copy()
    df = df.dropna()
    df = df[df["total"] > 0]

    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_missing"] = df["missing"] / df["total"]

    # Binomial GLM with total teeth as trial weights
    model = smf.glm(
        "prop_missing ~ is_human + age + sex + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total"],
    ).fit()

    coef = model.params["is_human"]
    pval = model.pvalues["is_human"]
    odds_ratio = float(np.exp(coef))

    # Marginal predictions at mean age/sex, averaged across tooth classes
    mean_age = df["age"].mean()
    mean_sex = df["sex"].mean()
    tc_weights = df["tooth_class"].value_counts(normalize=True)

    def marginal_pred(is_human):
        rows = []
        weights = []
        for tc, w in tc_weights.items():
            rows.append({"is_human": is_human, "age": mean_age, "sex": mean_sex, "tooth_class": tc})
            weights.append(w)
        pred = model.predict(pd.DataFrame(rows))
        return float(np.sum(pred * np.array(weights)))

    pred_human = marginal_pred(1)
    pred_nonhuman = marginal_pred(0)
    diff = pred_human - pred_nonhuman

    results = {
        "n_rows": int(len(df)),
        "coef_is_human": float(coef),
        "p_value_is_human": float(pval),
        "odds_ratio_is_human": odds_ratio,
        "pred_human": pred_human,
        "pred_nonhuman": pred_nonhuman,
        "pred_diff_human_minus_nonhuman": diff,
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
