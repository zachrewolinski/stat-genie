import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename for readability
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Derived variables
    df["social"] = df["choice"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["choice"] == 2).astype(int)

    # Age groups to approximate developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 8, 10, 12, 15],
        labels=["4-6", "7-8", "9-10", "11-12", "13-14"],
        right=True,
    )

    print("Basic counts:")
    print(df[["choice", "gender", "age", "majority_first", "site"]].describe(include="all"))
    print()

    print("Site-level proportions:")
    site_stats = (
        df.groupby("site")
        .agg(
            n=("choice", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )
    print(site_stats)
    print()

    print("Age-group-level proportions:")
    age_stats = (
        df.groupby("age_group")
        .agg(
            n=("choice", "size"),
            social_rate=("social", "mean"),
            majority_rate=("majority_choice", "mean"),
        )
        .reset_index()
    )
    print(age_stats)
    print()

    # Logistic regression: reliance on social information
    print("Logistic regression: social (1 = majority/minority, 0 = undemonstrated)")
    model_social = smf.logit("social ~ age + C(site)", data=df).fit(disp=False)
    print("Age coefficient (social):", model_social.params["age"])
    print("Age p-value (social):", model_social.pvalues["age"])
    print("Site dummy coefficients and p-values (social):")
    for name in model_social.params.index:
        if name.startswith("C(site)"):
            print(name, "coef", model_social.params[name], "p", model_social.pvalues[name])
    print()

    # Logistic regression: preference for majority vs minority among social choosers
    social_df = df[df["social"] == 1].copy()
    print("Logistic regression: majority_choice (1 = majority, 0 = minority) among social choosers")
    model_majority = smf.logit("majority_choice ~ age + C(site)", data=social_df).fit(disp=False)
    print("Age coefficient (majority):", model_majority.params["age"])
    print("Age p-value (majority):", model_majority.pvalues["age"])
    print("Site dummy coefficients and p-values (majority):")
    for name in model_majority.params.index:
        if name.startswith("C(site)"):
            print(name, "coef", model_majority.params[name], "p", model_majority.pvalues[name])


if __name__ == "__main__":
    main()

