import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy


def main():
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "total",
            "feature5": "age",
            "feature6": "age_unc",
            "feature7": "sex",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    df = df.dropna(subset=["missing", "total", "age", "sex", "tooth_class", "genus"])
    df = df[(df["total"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["total"])]

    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    exog = patsy.dmatrix(
        "human + age + sex + C(tooth_class)",
        data=df,
        return_type="dataframe",
    )

    endog = np.column_stack([df["missing"].to_numpy(), (df["total"] - df["missing"]).to_numpy()])

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = model.fit()

    coef = res.params.get("human", np.nan)
    se = res.bse.get("human", np.nan)
    z = coef / se if se and not np.isnan(se) else np.nan
    p = res.pvalues.get("human", np.nan)
    odds_ratio = float(np.exp(coef)) if not np.isnan(coef) else np.nan

    exog_h0 = exog.copy()
    exog_h0["human"] = 0
    exog_h1 = exog.copy()
    exog_h1["human"] = 1

    pred_h0 = res.predict(exog_h0)
    pred_h1 = res.predict(exog_h1)
    diff = float(np.mean(pred_h1 - pred_h0))

    # Map effect + confidence to Likert-style scalar
    if np.isnan(diff) or np.isnan(p):
        score = 0
    else:
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        confidence = max(0.0, min(1.0, 1.0 - float(p)))
        effect_strength = max(0.0, min(1.0, abs(diff) / 0.20))
        score = int(round(100 * (0.5 * confidence + 0.5 * effect_strength)))
        score *= sign

    print("rows", len(df))
    print("coef_human", coef)
    print("se_human", se)
    print("z", z)
    print("p", p)
    print("odds_ratio", odds_ratio)
    print("pred_diff", diff)
    print("score", score)

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()
