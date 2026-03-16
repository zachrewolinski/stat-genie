import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data)
    result = model.fit(disp=False)
    return result


def lr_test(full, reduced, df_diff: int):
    lr_stat = 2 * (full.llf - reduced.llf)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["relies_on_social"] = (df["y"] != 1).astype(int)
    df_demo = df[df["y"].isin([2, 3])].copy()
    df_demo["majority_choice"] = (df_demo["y"] == 2).astype(int)

    # Reliance on social information: chooses any demonstrated option vs undemonstrated
    rel_full = fit_logit("relies_on_social ~ age + C(culture)", df)
    rel_reduced_age = fit_logit("relies_on_social ~ age", df)
    rel_reduced_culture = fit_logit("relies_on_social ~ C(culture)", df)

    # Majority preference among demonstrated choices
    maj_full = fit_logit("majority_choice ~ age + C(culture)", df_demo)
    maj_reduced_age = fit_logit("majority_choice ~ age", df_demo)
    maj_reduced_culture = fit_logit("majority_choice ~ C(culture)", df_demo)

    # Likelihood-ratio tests
    rel_df_diff_culture = rel_full.df_model - rel_reduced_age.df_model
    rel_lr_culture, rel_p_culture = lr_test(rel_full, rel_reduced_age, int(rel_df_diff_culture))

    rel_df_diff_age = rel_full.df_model - rel_reduced_culture.df_model
    rel_lr_age, rel_p_age = lr_test(rel_full, rel_reduced_culture, int(rel_df_diff_age))

    maj_df_diff_culture = maj_full.df_model - maj_reduced_age.df_model
    maj_lr_culture, maj_p_culture = lr_test(maj_full, maj_reduced_age, int(maj_df_diff_culture))

    maj_df_diff_age = maj_full.df_model - maj_reduced_culture.df_model
    maj_lr_age, maj_p_age = lr_test(maj_full, maj_reduced_culture, int(maj_df_diff_age))

    # Simple descriptive summaries
    rel_by_age = df.groupby("age")["relies_on_social"].mean()
    rel_by_culture = df.groupby("culture")["relies_on_social"].mean()

    maj_by_age = df_demo.groupby("age")["majority_choice"].mean()
    maj_by_culture = df_demo.groupby("culture")["majority_choice"].mean()

    print("=== Reliance on social information (any demonstrated option) ===")
    print(rel_full.summary())
    print("\nLR test for culture (adding C(culture) beyond age):")
    print(f"  LR stat = {rel_lr_culture:.3f}, df = {int(rel_df_diff_culture)}, p = {rel_p_culture:.3g}")
    print("LR test for age (adding age beyond C(culture)):")
    print(f"  LR stat = {rel_lr_age:.3f}, df = {int(rel_df_diff_age)}, p = {rel_p_age:.3g}")
    print("\nMean reliance on social information by age:")
    print(rel_by_age)
    print("\nMean reliance on social information by culture:")
    print(rel_by_culture)

    print("\n=== Majority preference among demonstrated choices ===")
    print(maj_full.summary())
    print("\nLR test for culture (adding C(culture) beyond age):")
    print(f"  LR stat = {maj_lr_culture:.3f}, df = {int(maj_df_diff_culture)}, p = {maj_p_culture:.3g}")
    print("LR test for age (adding age beyond C(culture)):")
    print(f"  LR stat = {maj_lr_age:.3f}, df = {int(maj_df_diff_age)}, p = {maj_p_age:.3g}")
    print("\nMean majority choice among demonstrated options by age:")
    print(maj_by_age)
    print("\nMean majority choice among demonstrated options by culture:")
    print(maj_by_culture)


if __name__ == "__main__":
    main()

