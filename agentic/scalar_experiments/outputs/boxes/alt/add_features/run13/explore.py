import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    df["social_reliance"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = np.where(
        df["y"].isin([2, 3]), (df["y"] == 2).astype(int), np.nan
    )

    print("Dataset shape:", df.shape)
    print("\nOutcome distribution (y):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nSocial reliance (uses any demonstrated option):")
    print(df["social_reliance"].mean())
    print("\nSocial reliance by culture:")
    print(df.groupby("culture")["social_reliance"].agg(["mean", "count"]))

    print("\nMajority choice among social learners:")
    print(
        df.loc[df["social_reliance"] == 1, "majority_choice"]
        .value_counts(normalize=True)
        .sort_index()
    )
    print("\nMajority choice by culture (among social learners):")
    print(
        df[df["social_reliance"] == 1]
        .groupby("culture")["majority_choice"]
        .mean()
    )

    df1 = df.dropna(subset=["social_reliance", "age", "culture"]).copy()
    df1["age_c"] = df1["age"] - df1["age"].mean()

    model_full = smf.logit("social_reliance ~ age_c + C(culture)", data=df1).fit(
        disp=False
    )
    model_age = smf.logit("social_reliance ~ age_c", data=df1).fit(disp=False)
    model_null = smf.logit("social_reliance ~ 1", data=df1).fit(disp=False)

    print("\nLogit: social_reliance ~ age + culture")
    print(model_full.summary())
    lr_age = 2 * (model_age.llf - model_null.llf)
    p_age = stats.chi2.sf(lr_age, df=1)
    print(f"Age effect on social reliance (LR= {lr_age:.3f}, p= {p_age:.5g})")

    lr_culture = 2 * (model_full.llf - model_age.llf)
    df_diff = int(model_full.df_model - model_age.df_model)
    p_culture = stats.chi2.sf(lr_culture, df=df_diff)
    print(
        f"Culture effect on social reliance (LR= {lr_culture:.3f}, "
        f"df= {df_diff}, p= {p_culture:.5g})"
    )

    df2 = df.dropna(subset=["majority_choice", "age", "culture"]).copy()
    df2["age_c"] = df2["age"] - df2["age"].mean()

    model2_full = smf.logit("majority_choice ~ age_c + C(culture)", data=df2).fit(
        disp=False
    )
    model2_age = smf.logit("majority_choice ~ age_c", data=df2).fit(disp=False)
    model2_null = smf.logit("majority_choice ~ 1", data=df2).fit(disp=False)

    print("\nLogit: majority_choice ~ age + culture (social learners only)")
    print(model2_full.summary())
    lr2_age = 2 * (model2_age.llf - model2_null.llf)
    p2_age = stats.chi2.sf(lr2_age, df=1)
    print(
        f"Age effect on majority vs minority (LR= {lr2_age:.3f}, "
        f"p= {p2_age:.5g})"
    )

    lr2_culture = 2 * (model2_full.llf - model2_age.llf)
    df2_diff = int(model2_full.df_model - model2_age.df_model)
    p2_culture = stats.chi2.sf(lr2_culture, df=df2_diff)
    print(
        f"Culture effect on majority vs minority (LR= {lr2_culture:.3f}, "
        f"df= {df2_diff}, p= {p2_culture:.5g})"
    )


if __name__ == "__main__":
    main()

