import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["chose_social"] = df["feature1"].isin([2, 3]).astype(int)
    df["chose_majority_any"] = (df["feature1"] == 2).astype(int)

    social_df = df[df["feature1"].isin([2, 3])].copy()
    social_df["chose_majority"] = (social_df["feature1"] == 2).astype(int)

    print("N total:", len(df))
    print("N chose majority:", int(df["chose_majority_any"].sum()))
    print("N chose minority:", int((df["feature1"] == 3).sum()))
    print("N chose undemonstrated:", int((df["feature1"] == 1).sum()))
    print()

    # Age groups for descriptive summaries
    bins = [4, 6, 8, 10, 12, 14, 16]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-13", "14-15"]
    df["age_group"] = pd.cut(df["feature3"], bins=bins, labels=labels, right=False, include_lowest=True)

    print("=== Proportion choosing majority option by site (feature5) ===")
    site_majority = df.groupby("feature5")["chose_majority_any"].mean()
    print(site_majority.to_string())
    print()

    print("=== Proportion choosing majority option by age group ===")
    age_majority = df.groupby("age_group")["chose_majority_any"].mean()
    print(age_majority.to_string())
    print()

    print("=== Proportion choosing any demonstrated option by site (feature5) ===")
    site_social = df.groupby("feature5")["chose_social"].mean()
    print(site_social.to_string())
    print()

    print("=== Proportion choosing any demonstrated option by age group ===")
    age_social = df.groupby("age_group")["chose_social"].mean()
    print(age_social.to_string())
    print()

    print("=== Logistic regression: chose_social ~ age + site + gender + majority_first ===")
    try:
        model_social = smf.logit(
            "chose_social ~ feature3 + C(feature5) + C(feature2) + feature4",
            data=df,
        ).fit(disp=0)
        print(model_social.summary())
    except Exception as exc:  # pragma: no cover - defensive
        print("Logistic model for chose_social failed to converge:", exc)
    print()

    print("=== Logistic regression: chose_majority (among social choosers) ~ age + site + gender + majority_first ===")
    try:
        model_majority = smf.logit(
            "chose_majority ~ feature3 + C(feature5) + C(feature2) + feature4",
            data=social_df,
        ).fit(disp=0)
        print(model_majority.summary())
    except Exception as exc:  # pragma: no cover - defensive
        print("Logistic model for chose_majority failed to converge:", exc)
    print()


if __name__ == "__main__":
    main()

