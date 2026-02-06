import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("crofoot.csv")

    # Relative group size and relative location advantage
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    X = df[["size_diff", "loc_adv"]]
    X = sm.add_constant(X)
    y = df["win"]

    model = sm.Logit(y, X).fit(disp=False)
    print(model.summary())

    # Extract key results for reporting
    coef = model.params
    pvals = model.pvalues
    out = pd.DataFrame({"coef": coef, "pvalue": pvals})
    print("\nKey coefficients and p-values:\n", out)


if __name__ == "__main__":
    main()
