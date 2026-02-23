import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcome coding:
    # feature1: 1 = undemonstrated option, 2 = majority option, 3 = minority option
    # feature3: age in years
    # feature5: site ID (proxy for cultural context)

    # Reliance on social information: choosing either majority or minority option
    df["reliant"] = (df["feature1"] != 1).astype(int)

    # Preference for majority cues: among those who follow social information,
    # do they choose the majority option (2) over the minority option (3)?
    df_soc = df[df["feature1"].isin([2, 3])].copy()
    df_soc["majority_choice"] = (df_soc["feature1"] == 2).astype(int)

    print("N total:", len(df))
    print("N using social information (majority or minority):", df["reliant"].sum())
    print(
        "Overall reliance on social information (proportion):",
        df["reliant"].mean(),
    )

    # --- Cultural variation (site effects) ---
    # Chi-square tests of independence between site and reliance/preference
    ct_rel_site = pd.crosstab(df["feature5"], df["reliant"])
    chi2_rel_site, p_rel_site, dof_rel_site, exp_rel_site = stats.chi2_contingency(
        ct_rel_site
    )

    ct_maj_site = pd.crosstab(df_soc["feature5"], df_soc["majority_choice"])
    chi2_maj_site, p_maj_site, dof_maj_site, exp_maj_site = stats.chi2_contingency(
        ct_maj_site
    )

    print("\n=== Cultural (site) variation ===")
    print("Reliance vs site chi2:", chi2_rel_site, "df:", dof_rel_site, "p:", p_rel_site)
    print(
        "Majority preference vs site chi2:",
        chi2_maj_site,
        "df:",
        dof_maj_site,
        "p:",
        p_maj_site,
    )
    print("\nReliance proportion by site:")
    print(df.groupby("feature5")["reliant"].mean())
    print("\nMajority-choice proportion by site (among social learners):")
    print(df_soc.groupby("feature5")["majority_choice"].mean())

    # --- Developmental (age) variation ---
    # Logistic regression of reliance on age
    model_rel_age = smf.logit("reliant ~ feature3", data=df).fit(disp=False)
    print("\n=== Developmental (age) variation ===")
    print("Logit(reliant) ~ age coefficient and p-value:")
    print(model_rel_age.params)
    print(model_rel_age.pvalues)

    # Logistic regression of majority preference on age (within social learners)
    model_maj_age = smf.logit("majority_choice ~ feature3", data=df_soc).fit(disp=False)
    print("\nLogit(majority_choice) ~ age coefficient and p-value:")
    print(model_maj_age.params)
    print(model_maj_age.pvalues)


if __name__ == "__main__":
    main()

