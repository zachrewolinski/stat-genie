import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("boxes.csv")
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Outcome recoding
    # choice: 1 = undemonstrated option, 2 = majority option, 3 = minority option
    df["social_info"] = df["choice"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["choice"] == 2,
        1,
        np.where(df["choice"] == 3, 0, np.nan),
    )

    # Basic type casting
    df["gender"] = df["gender"].astype("category")
    df["site"] = df["site"].astype("category")
    df["majority_first"] = df["majority_first"].astype(int)

    # Age groups for descriptive summaries
    bins = [4, 7, 10, 13, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    print("N observations:", len(df))
    print()

    # Descriptive statistics: reliance on social information
    print("=== Reliance on social information (any demonstrated option) ===")
    print("Overall proportion using social information:",
          df["social_info"].mean())
    print()
    print("By age group:")
    print(df.groupby("age_group")["social_info"].mean())
    print()
    print("By site:")
    print(df.groupby("site")["social_info"].mean())
    print()

    # Logistic regression: social_info ~ age + site (core developmental and cultural factors)
    print("=== Logistic regression: social_info ~ age + C(site) ===")
    model_social_null = smf.logit("social_info ~ 1", data=df).fit(disp=False)
    model_social = smf.logit("social_info ~ age + C(site)", data=df).fit(disp=False)
    print(model_social.summary())
    lr_stat = 2 * (model_social.llf - model_social_null.llf)
    df_diff = model_social.df_model - model_social_null.df_model
    p_lr = chi2.sf(lr_stat, df_diff)
    print(f"LR test for age + site vs null: chi2({df_diff}) = {lr_stat:.3f}, p = {p_lr:.3f}")
    print()

    # Descriptive statistics: majority preference among social users
    df_social = df[df["social_info"] == 1].copy()
    print("N using social information:", len(df_social))
    print()
    print("=== Majority preference among children using social information ===")
    print("Overall proportion choosing majority (vs minority):",
          df_social["majority_choice"].mean())
    print()
    print("By age group:")
    print(df_social.groupby("age_group")["majority_choice"].mean())
    print()
    print("By site:")
    print(df_social.groupby("site")["majority_choice"].mean())
    print()

    # Logistic regression: majority_choice ~ age + C(site) (core developmental and cultural factors)
    print("=== Logistic regression: majority_choice ~ age + C(site) (social users only) ===")
    model_maj_null = smf.logit("majority_choice ~ 1", data=df_social).fit(disp=False)
    model_maj = smf.logit(
        "majority_choice ~ age + C(site)",
        data=df_social,
    ).fit(disp=False)
    print(model_maj.summary())
    lr_stat_maj = 2 * (model_maj.llf - model_maj_null.llf)
    df_diff_maj = model_maj.df_model - model_maj_null.df_model
    p_lr_maj = chi2.sf(lr_stat_maj, df_diff_maj)
    print(
        f"LR test for age + site vs null (majority choice): "
        f"chi2({df_diff_maj}) = {lr_stat_maj:.3f}, p = {p_lr_maj:.3f}"
    )


if __name__ == "__main__":
    main()
