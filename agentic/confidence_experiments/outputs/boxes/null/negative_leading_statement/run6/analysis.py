import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf


def chi_square(table: pd.DataFrame):
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof


def main():
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["uses_social"] = (df["y"] != 1).astype(int)

    # Age groups for descriptive and chi-square tests
    bins = [4, 6, 8, 10, 12, 15]
    labels = ["4-5", "6-7", "8-9", "10-11", "12-14"]
    df["age_group"] = pd.cut(
        df["age"], bins=bins, labels=labels, include_lowest=True, right=False
    )

    df_social = df[df["uses_social"] == 1].copy()
    df_social["major_choice"] = (df_social["y"] == 2).astype(int)
    df_social["age_group"] = pd.cut(
        df_social["age"], bins=bins, labels=labels, include_lowest=True, right=False
    )

    print("N total:", len(df))
    print("N using social information:", df["uses_social"].sum())
    print(
        "Proportion using social information overall:",
        df["uses_social"].mean(),
    )

    # Descriptive variation by culture and age
    print("\nProportion using social information by culture:")
    print(
        df.groupby("culture")["uses_social"]
        .mean()
        .to_frame("prop_uses_social")
        .assign(n=lambda x: df.groupby("culture").size())
    )

    print("\nProportion using social information by age_group:")
    print(
        df.groupby("age_group")["uses_social"]
        .mean()
        .to_frame("prop_uses_social")
        .assign(n=lambda x: df.groupby("age_group").size())
    )

    print("\nAmong social learners, proportion choosing majority option by culture:")
    print(
        df_social.groupby("culture")["major_choice"]
        .mean()
        .to_frame("prop_major_choice")
        .assign(n=lambda x: df_social.groupby("culture").size())
    )

    print("\nAmong social learners, proportion choosing majority option by age_group:")
    print(
        df_social.groupby("age_group")["major_choice"]
        .mean()
        .to_frame("prop_major_choice")
        .assign(n=lambda x: df_social.groupby("age_group").size())
    )

    # Chi-square tests for association
    print("\nChi-square tests:")
    tab1 = pd.crosstab(df["culture"], df["uses_social"])
    chi1, p1, dof1 = chi_square(tab1)
    print("uses_social ~ culture: chi2=%.3f, dof=%d, p=%.4g" % (chi1, dof1, p1))

    tab2 = pd.crosstab(df["age_group"], df["uses_social"])
    chi2, p2, dof2 = chi_square(tab2)
    print("uses_social ~ age_group: chi2=%.3f, dof=%d, p=%.4g" % (chi2, dof2, p2))

    tab3 = pd.crosstab(df_social["culture"], df_social["major_choice"])
    chi3, p3, dof3 = chi_square(tab3)
    print("major_choice ~ culture: chi2=%.3f, dof=%d, p=%.4g" % (chi3, dof3, p3))

    tab4 = pd.crosstab(df_social["age_group"], df_social["major_choice"])
    chi4, p4, dof4 = chi_square(tab4)
    print("major_choice ~ age_group: chi2=%.3f, dof=%d, p=%.4g" % (chi4, dof4, p4))

    # Logistic regression models
    print("\nLogistic regression: uses_social ~ age + culture")
    model_uses = smf.logit("uses_social ~ age + C(culture)", data=df).fit(disp=False)
    print(model_uses.summary())

    print("\nLogistic regression: majority vs minority among social learners")
    model_major = smf.logit("major_choice ~ age + C(culture)", data=df_social).fit(
        disp=False
    )
    print(model_major.summary())


if __name__ == "__main__":
    main()

