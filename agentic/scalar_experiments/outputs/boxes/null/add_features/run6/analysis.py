import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df_mm = df[df["y"].isin([2, 3])].copy()
    df_mm["majority_choice"] = (df_mm["y"] == 2).astype(int)

    print("N rows:", len(df))
    print("Outcome distribution y (proportions):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    # Descriptive stats by culture
    print("Social-choice rate by culture:")
    print(df.groupby("culture")["social_choice"].mean())
    print()

    print("Majority-choice rate (among social choices) by culture:")
    print(df_mm.groupby("culture")["majority_choice"].mean())
    print()

    # Descriptive stats by age quartiles
    df["age_q"] = pd.qcut(df["age"], 4, duplicates="drop")
    df_mm["age_q"] = pd.qcut(df_mm["age"], 4, duplicates="drop")

    print("Social-choice rate by age quartile:")
    print(df.groupby("age_q")["social_choice"].mean())
    print()

    print("Majority-choice rate by age quartile (social choices only):")
    print(df_mm.groupby("age_q")["majority_choice"].mean())
    print()

    # Chi-square tests for culture associations
    cont_social = pd.crosstab(df["culture"], df["social_choice"])
    chi2_sc, p_sc, dof_sc, _ = chi2_contingency(cont_social)
    print("Chi-square: culture vs social_choice")
    print("  chi2={:.3f}, dof={}, p-value={:.5f}".format(chi2_sc, dof_sc, p_sc))
    print()

    cont_maj = pd.crosstab(df_mm["culture"], df_mm["majority_choice"])
    chi2_m, p_m, dof_m, _ = chi2_contingency(cont_maj)
    print("Chi-square: culture vs majority_choice (among social choices)")
    print("  chi2={:.3f}, dof={}, p-value={:.5f}".format(chi2_m, dof_m, p_m))
    print()

    # Logistic regressions for age and culture effects
    print("Logit: social_choice ~ age + C(culture)")
    model_sc = smf.logit("social_choice ~ age + C(culture)", data=df).fit(disp=False)
    print(model_sc.summary())
    print()

    print("Logit: majority_choice ~ age + C(culture) (social choices only)")
    model_m = smf.logit("majority_choice ~ age + C(culture)", data=df_mm).fit(disp=False)
    print(model_m.summary())


if __name__ == "__main__":
    main()

