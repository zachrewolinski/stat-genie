import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Social-information use: chose majority or minority option
    df["social"] = df["feature1"].isin([2, 3]).astype(int)

    # Among social choices, majority preference vs minority
    df_social = df[df["social"] == 1].copy()
    df_social["majority"] = (df_social["feature1"] == 2).astype(int)

    # Treat site as categorical
    df["site"] = df["feature5"].astype("category")
    df_social["site"] = df_social["feature5"].astype("category")

    # Age variable (feature3) is already in years

    print("=== Descriptive statistics ===")
    print("N total:", len(df))
    print("Proportion using social information (any):", df["social"].mean())
    print(
        "Proportion choosing majority option overall:",
        (df["feature1"] == 2).mean(),
    )
    print(
        "Proportion choosing minority option overall:",
        (df["feature1"] == 3).mean(),
    )
    print(
        "Proportion choosing undemonstrated option overall:",
        (df["feature1"] == 1).mean(),
    )
    print()

    print("Proportion using social information by site:")
    print(df.groupby("site")["social"].mean())
    print()

    if not df_social.empty:
        print("Proportion choosing majority (vs minority) among social users by site:")
        print(df_social.groupby("site")["majority"].mean())
        print()

    # Age groups for descriptive patterns
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["feature3"], bins=bins, labels=labels)
    df_social["age_group"] = pd.cut(df_social["feature3"], bins=bins, labels=labels)

    print("Proportion using social information by age group:")
    print(df.groupby("age_group")["social"].mean())
    print()

    if not df_social.empty:
        print("Proportion choosing majority (vs minority) by age group (social users):")
        print(df_social.groupby("age_group")["majority"].mean())
        print()

    # Logistic regression: social-information use ~ age + site
    print("=== Logistic regression: social-information use (social) ~ age + site ===")
    model_social = smf.logit("social ~ feature3 + C(site)", data=df).fit(disp=False)
    print(model_social.summary())
    print()

    if len(df_social["majority"].unique()) > 1:
        print(
            "=== Logistic regression: majority choice (vs minority) "
            "~ age + site, among social users ==="
        )
        model_majority = smf.logit(
            "majority ~ feature3 + C(site)", data=df_social
        ).fit(disp=False)
        print(model_majority.summary())
        print()
    else:
        print(
            "All social users made the same choice (all majority or all minority); "
            "logistic model for majority vs minority not estimable."
        )


if __name__ == "__main__":
    main()

