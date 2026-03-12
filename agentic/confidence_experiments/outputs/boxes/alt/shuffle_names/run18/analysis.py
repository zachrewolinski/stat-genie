import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcome recoding
    # 1 = undemonstrated option (no social information)
    # 2 = majority option
    # 3 = minority option
    df["used_social"] = (df["majority_first"] != 1).astype(int)
    df["site"] = df["y"].astype("category")

    # Subset where children followed either majority or minority demonstrators
    df_social = df[df["used_social"] == 1].copy()
    df_social["chose_majority"] = (df_social["majority_first"] == 2).astype(int)
    df_social["site"] = df_social["y"].astype("category")

    # Age groups for descriptive summaries
    df["age_group"] = pd.cut(
        df["age"],
        bins=[4, 7, 10, 13, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=False,
    )
    df_social["age_group"] = pd.cut(
        df_social["age"],
        bins=[4, 7, 10, 13, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
        right=False,
    )

    print("=== Basic counts ===")
    print(f"N total: {len(df)}")
    print("Outcome distribution (1=undemonstrated, 2=majority, 3=minority):")
    print(df["majority_first"].value_counts().sort_index())
    print()

    print("=== Reliance on social information (used_social) ===")
    print("Overall proportion using social information:")
    print(df["used_social"].mean())
    print("\nBy age group:")
    print(df.groupby("age_group")["used_social"].mean())
    print("\nBy site:")
    print(df.groupby("site")["used_social"].mean())
    print()

    # Logistic regression: reliance on social information ~ age + site
    print("=== Logistic regression: used_social ~ age + C(site) ===")
    model_used = smf.logit("used_social ~ age + C(site)", data=df).fit(disp=False)
    print(model_used.summary())
    print("\nParameters:")
    print(model_used.params)
    print("\nP-values:")
    print(model_used.pvalues)
    print()

    # Reduced models to separately test age and site effects on reliance
    print("=== Reduced models for used_social ===")
    model_used_age = smf.logit("used_social ~ age", data=df).fit(disp=False)
    print("Model used_social ~ age: LLR p-value:", model_used_age.llr_pvalue)
    print(model_used_age.summary())
    print()

    model_used_site = smf.logit("used_social ~ C(site)", data=df).fit(disp=False)
    print("Model used_social ~ C(site): LLR p-value:", model_used_site.llr_pvalue)
    print(model_used_site.summary())
    print()

    if len(df_social) > 0:
        print("=== Preference for majority among social users (chose_majority) ===")
        print("Overall proportion choosing majority (conditional on using social info):")
        print(df_social["chose_majority"].mean())
        print("\nBy age group:")
        print(df_social.groupby("age_group")["chose_majority"].mean())
        print("\nBy site:")
        print(df_social.groupby("site")["chose_majority"].mean())
        print()

        print("=== Logistic regression: chose_majority ~ age + C(site) ===")
        try:
            model_majority = smf.logit(
                "chose_majority ~ age + C(site)", data=df_social
            ).fit(disp=False)
            print(model_majority.summary())
            print("\nParameters:")
            print(model_majority.params)
            print("\nP-values:")
            print(model_majority.pvalues)
            print()

            # Reduced models to separately test age and site effects on majority preference
            print("=== Reduced models for chose_majority ===")
            model_majority_age = smf.logit(
                "chose_majority ~ age", data=df_social
            ).fit(disp=False)
            print(
                "Model chose_majority ~ age: LLR p-value:",
                model_majority_age.llr_pvalue,
            )
            print(model_majority_age.summary())
            print()

            model_majority_site = smf.logit(
                "chose_majority ~ C(site)", data=df_social
            ).fit(disp=False)
            print(
                "Model chose_majority ~ C(site): LLR p-value:",
                model_majority_site.llr_pvalue,
            )
            print(model_majority_site.summary())
        except Exception as exc:  # pragma: no cover - diagnostic only
            print("Logistic regression for majority preference failed:", repr(exc))
    else:
        print("No observations with social information used; skipping majority preference model.")


if __name__ == "__main__":
    main()
