import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("Sample size:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts())

    print("\nAffairs by children (mean, median, std):")
    print(df.groupby("children")["affairs"].agg(["mean", "median", "std"]))

    print("\nProportion with any affair by children:")
    print(df.groupby("children")["any_affair"].mean())

    # Logistic regression: any affair ~ children (unadjusted)
    model1 = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("\nLogistic regression (unadjusted): any_affair ~ C(children)")
    print(model1.summary())

    params1 = model1.params
    conf1 = model1.conf_int()
    child_terms1 = [name for name in params1.index if "children" in name]
    if child_terms1:
        term1 = child_terms1[0]
        or1 = float(np.exp(params1[term1]))
        ci1 = np.exp(conf1.loc[term1].to_numpy())
        p1 = float(model1.pvalues[term1])
        print(
            f"\nUnadjusted OR for {term1}: {or1:.3f} "
            f"(95% CI {ci1[0]:.3f}, {ci1[1]:.3f}), p = {p1:.4f}"
        )

    # Logistic regression with controls for key covariates
    formula2 = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    model2 = smf.logit(formula2, data=df).fit(disp=False)
    print(
        "\nLogistic regression (adjusted): any_affair ~ C(children) + "
        "age + yearsmarried + religiousness + education + occupation + "
        "rating + C(gender)"
    )
    print(model2.summary())

    params2 = model2.params
    conf2 = model2.conf_int()
    child_terms2 = [name for name in params2.index if "children" in name]
    if child_terms2:
        term2 = child_terms2[0]
        or2 = float(np.exp(params2[term2]))
        ci2 = np.exp(conf2.loc[term2].to_numpy())
        p2 = float(model2.pvalues[term2])
        print(
            f"\nAdjusted OR for {term2}: {or2:.3f} "
            f"(95% CI {ci2[0]:.3f}, {ci2[1]:.3f}), p = {p2:.4f}"
        )


if __name__ == "__main__":
    main()

