import pandas as pd
import patsy
import statsmodels.api as sm


def main():
    df = pd.read_csv("amtl.csv")
    df = df[df["sockets"] > 0].copy()
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["non_amtl"] = df["sockets"] - df["num_amtl"]

    formula = "is_human + age + prob_male + C(tooth_class)"
    exog = patsy.dmatrix(formula, data=df, return_type="dataframe")
    endog = df[["num_amtl", "non_amtl"]]

    model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    coef = model.params["is_human"]
    pval = model.pvalues["is_human"]

    exog_human = patsy.build_design_matrices([exog.design_info], df.assign(is_human=1))[0]
    exog_nonhuman = patsy.build_design_matrices([exog.design_info], df.assign(is_human=0))[0]
    pred_human = model.predict(exog_human).mean()
    pred_nonhuman = model.predict(exog_nonhuman).mean()

    print(model.summary())
    print("\nAdjusted mean AMTL rate (human):", pred_human)
    print("Adjusted mean AMTL rate (non-human):", pred_nonhuman)
    print("is_human coef:", coef, "p-value:", pval)

    return coef, pval, pred_human, pred_nonhuman


if __name__ == "__main__":
    main()
