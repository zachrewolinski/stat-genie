import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social_reliance"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Subset for children who followed a demonstrated option
    df_demonstrated = df[df["y"] != 1].copy()
    df_demonstrated["majority_preference"] = (df_demonstrated["y"] == 2).astype(
        int
    )

    # Age groups for descriptive summaries (developmental stages)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        right=True,
        include_lowest=True,
    )
    df_demonstrated["age_group"] = pd.cut(
        df_demonstrated["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        right=True,
        include_lowest=True,
    )

    print("N total:", len(df))
    print("N demonstrated (social_reliance=1):", df["social_reliance"].sum())
    print()

    # Descriptive summaries
    print("=== Descriptive: social reliance by culture ===")
    print(
        df.groupby("culture")["social_reliance"]
        .mean()
        .rename("prop_social_reliance")
    )
    print()

    print("=== Descriptive: social reliance by age_group ===")
    print(
        df.groupby("age_group")["social_reliance"]
        .mean()
        .rename("prop_social_reliance")
    )
    print()

    print("=== Descriptive: majority preference by culture (among demonstrators) ===")
    print(
        df_demonstrated.groupby("culture")["majority_preference"]
        .mean()
        .rename("prop_majority_preference")
    )
    print()

    print("=== Descriptive: majority preference by age_group (among demonstrators) ===")
    print(
        df_demonstrated.groupby("age_group")["majority_preference"]
        .mean()
        .rename("prop_majority_preference")
    )
    print()

    # Logistic regression: reliance on social information ~ age + culture
    print("=== Logistic regression: social_reliance ~ age + C(culture) ===")
    model_rel = smf.logit("social_reliance ~ age + C(culture)", data=df)
    result_rel = model_rel.fit(disp=False, maxiter=200)
    print(result_rel.summary())
    print()

    # Wald tests for age and culture (overall)
    wald_age_rel = result_rel.wald_test("age = 0")
    # culture has levels 1-8; 1 is reference
    culture_terms = ", ".join(
        [f"C(culture)[T.{k}] = 0" for k in range(2, 9)]
    )
    wald_culture_rel = result_rel.wald_test(culture_terms)
    print("Social reliance - age effect p-value:", float(wald_age_rel.pvalue))
    print("Social reliance - culture effect p-value:", float(wald_culture_rel.pvalue))
    print()

    # Logistic regression: majority preference ~ age + culture (among demonstrators)
    print(
        "=== Logistic regression: majority_preference ~ age + C(culture) "
        "(among demonstrators) ==="
    )
    model_maj = smf.logit("majority_preference ~ age + C(culture)", data=df_demonstrated)
    result_maj = model_maj.fit(disp=False, maxiter=200)
    print(result_maj.summary())
    print()

    wald_age_maj = result_maj.wald_test("age = 0")
    culture_terms_dem = ", ".join(
        [f"C(culture)[T.{k}] = 0" for k in range(2, 9)]
    )
    wald_culture_maj = result_maj.wald_test(culture_terms_dem)
    print("Majority preference - age effect p-value:", float(wald_age_maj.pvalue))
    print("Majority preference - culture effect p-value:", float(wald_culture_maj.pvalue))


if __name__ == "__main__":
    main()

