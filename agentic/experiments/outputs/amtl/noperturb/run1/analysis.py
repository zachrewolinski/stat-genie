import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["non_amtl"] = df["sockets"] - df["num_amtl"]

    # Binomial GLM with logit link; cluster-robust SE by specimen
    y = df[["num_amtl", "non_amtl"]]
    formula = "is_human + age + prob_male + C(tooth_class)"
    X = patsy.dmatrix(formula, df, return_type="dataframe")

    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    coef = float(res.params["is_human"])
    pval = float(res.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Average marginal difference in predicted AMTL rate
    X_h = X.copy()
    X_h["is_human"] = 1
    X_n = X.copy()
    X_n["is_human"] = 0
    pred_h = res.predict(X_h)
    pred_n = res.predict(X_n)
    avg_diff = float((pred_h - pred_n).mean())
    avg_h = float(pred_h.mean())
    avg_n = float(pred_n.mean())

    # Decision rule: higher AMTL if coefficient > 0 and p < 0.05
    higher = coef > 0 and pval < 0.05
    first_line = "Yes" if higher else "No"

    explanation = (
        f"Binomial GLM controlling for age, sex (prob_male), and tooth class shows "
        f"the human indicator has a log-odds coefficient of {coef:.3f} "
        f"(odds ratio {odds_ratio:.2f}, p={pval:.3g}). "
        f"Predicted AMTL rate averages {avg_h:.3f} for humans vs {avg_n:.3f} for non-humans "
        f"(difference {avg_diff:.3f})."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(first_line + "\n")
        f.write(explanation + "\n")


if __name__ == "__main__":
    main()
