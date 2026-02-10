import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["social_choice"] = (df["y"] != 1).astype(int)

    # Bin age into coarse developmental stages (younger / middle / older)
    bins = [3, 6, 9, 12, 15]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)

    print("Basic outcome distribution (y: 1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())
    print()

    # --- Culture variation: chi-square tests ---
    print("Chi-square test: outcome (y) vs culture")
    ct_culture = pd.crosstab(df["culture"], df["y"])
    chi2_cult, p_cult, dof_cult, _ = stats.chi2_contingency(ct_culture)
    print("Chi2:", chi2_cult, "df:", dof_cult, "p-value:", p_cult)
    print()

    print("Chi-square test: majority_choice vs culture")
    ct_maj_cult = pd.crosstab(df["culture"], df["majority_choice"])
    chi2_mc, p_mc, dof_mc, _ = stats.chi2_contingency(ct_maj_cult)
    print("Chi2:", chi2_mc, "df:", dof_mc, "p-value:", p_mc)
    print()

    # --- Developmental (age) variation: chi-square on age_group ---
    print("Chi-square test: outcome (y) vs age_group")
    ct_age = pd.crosstab(df["age_group"], df["y"])
    chi2_age, p_age, dof_age, _ = stats.chi2_contingency(ct_age)
    print("Chi2:", chi2_age, "df:", dof_age, "p-value:", p_age)
    print()

    print("Chi-square test: majority_choice vs age_group")
    ct_maj_age = pd.crosstab(df["age_group"], df["majority_choice"])
    chi2_ma, p_ma, dof_ma, _ = stats.chi2_contingency(ct_maj_age)
    print("Chi2:", chi2_ma, "df:", dof_ma, "p-value:", p_ma)
    print()

    # --- Regression models as a complementary check ---
    print("Logistic regression: majority_choice ~ age + C(culture)")
    logit_maj = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)
    print(logit_maj.summary())
    print()

    print("Logistic regression: social_choice ~ age + C(culture)")
    logit_soc = smf.logit("social_choice ~ age + C(culture)", data=df).fit(disp=False)
    print(logit_soc.summary())


if __name__ == "__main__":
    main()

