import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Reliance on social info: chose majority or minority vs undemonstrated option
    df["social_choice"] = (df["y"] != 1).astype(int)

    # Preference for majority among social choices
    df_social = df[df["y"].isin([2, 3])].copy()
    df_social["majority_choice"] = (df_social["y"] == 2).astype(int)

    # Models for social reliance
    m_social_full = smf.logit("social_choice ~ age + C(culture)", data=df).fit(disp=0)
    m_social_age = smf.logit("social_choice ~ age", data=df).fit(disp=0)
    m_social_culture = smf.logit("social_choice ~ C(culture)", data=df).fit(disp=0)

    # LRTs for social reliance
    lr_social_culture = lr_test(m_social_full, m_social_age)
    lr_social_age = lr_test(m_social_full, m_social_culture)

    # Models for majority preference (among social choices)
    m_majority_full = smf.logit("majority_choice ~ age + C(culture)", data=df_social).fit(disp=0)
    m_majority_age = smf.logit("majority_choice ~ age", data=df_social).fit(disp=0)
    m_majority_culture = smf.logit("majority_choice ~ C(culture)", data=df_social).fit(disp=0)

    # LRTs for majority preference
    lr_majority_culture = lr_test(m_majority_full, m_majority_age)
    lr_majority_age = lr_test(m_majority_full, m_majority_culture)

    # Summaries
    summary = {
        "social_reliance": {
            "culture_LR": lr_social_culture,
            "age_LR": lr_social_age,
        },
        "majority_preference": {
            "culture_LR": lr_majority_culture,
            "age_LR": lr_majority_age,
        },
    }

    print("Sample size:", len(df))
    print("Social-choice sample:", len(df_social))
    print("LRT social reliance (culture | age): stat, df, p =", summary["social_reliance"]["culture_LR"])
    print("LRT social reliance (age | culture): stat, df, p =", summary["social_reliance"]["age_LR"])
    print("LRT majority preference (culture | age): stat, df, p =", summary["majority_preference"]["culture_LR"])
    print("LRT majority preference (age | culture): stat, df, p =", summary["majority_preference"]["age_LR"])


if __name__ == "__main__":
    main()
