import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("amtl.csv")

    # Basic cleaning / validation
    df = df.copy()
    df = df[df["feature4"].notna() & df["feature3"].notna()]
    df = df[(df["feature4"] > 0) & (df["feature3"] >= 0)]
    df = df[df["feature3"] <= df["feature4"]]

    # Derived fields
    df["missing_prop"] = df["feature3"] / df["feature4"]

    # Categorical fields
    df["feature1"] = df["feature1"].astype("category")
    df["feature8"] = df["feature8"].astype("category")

    # GLM binomial: proportion with frequency weights
    model = smf.glm(
        "missing_prop ~ C(feature8) + feature5 + feature7 + C(feature1)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["feature4"],
    ).fit()

    print(model.summary())

    # Extract Homo sapiens coefficient (vs baseline genus)
    categories = list(df["feature8"].cat.categories)
    print("Genus categories:", categories)

    homo_is_baseline = categories[0] == "Homo sapiens"
    if homo_is_baseline:
        print("Homo sapiens is baseline; coefficients for other genera are relative to Homo sapiens.")
    else:
        term = "C(feature8)[T.Homo sapiens]"
        if term in model.params:
            coef = model.params[term]
            pval = model.pvalues[term]
            print("Homo sapiens coefficient:", coef)
            print("Homo sapiens p-value:", pval)
        else:
            print("Homo sapiens term not found in model params.")


if __name__ == "__main__":
    main()
