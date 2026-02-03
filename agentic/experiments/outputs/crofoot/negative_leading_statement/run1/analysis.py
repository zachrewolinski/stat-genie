import pandas as pd
import statsmodels.api as sm


def main():
    df = pd.read_csv("crofoot.csv")

    # Relative predictors
    df["rel_size"] = df["n_focal"] - df["n_other"]
    # Positive rel_dist means focal is closer to its home-range center than the other group is to its own
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    # Primary model: win ~ relative size + relative location
    X = sm.add_constant(df[["rel_size", "rel_dist"]])
    y = df["win"]
    model = sm.Logit(y, X).fit(disp=False)

    # Alternative specification using size ratio
    df["size_ratio"] = df["n_focal"] / df["n_other"]
    X2 = sm.add_constant(df[["size_ratio", "rel_dist"]])
    model2 = sm.Logit(y, X2).fit(disp=False)

    # Expanded model with separate size and distance terms
    X3 = sm.add_constant(df[["n_focal", "n_other", "dist_focal", "dist_other"]])
    model3 = sm.Logit(y, X3).fit(disp=False)

    print("=== Primary model: win ~ rel_size + rel_dist ===")
    print(model.summary())
    print("\nCoefficients and p-values:")
    print(model.params)
    print(model.pvalues)

    print("\n=== Alternative model: win ~ size_ratio + rel_dist ===")
    print(model2.summary())
    print("\nCoefficients and p-values:")
    print(model2.params)
    print(model2.pvalues)

    print("\n=== Expanded model: win ~ n_focal + n_other + dist_focal + dist_other ===")
    print(model3.summary())
    print("\nCoefficients and p-values:")
    print(model3.params)
    print(model3.pvalues)


if __name__ == "__main__":
    main()
