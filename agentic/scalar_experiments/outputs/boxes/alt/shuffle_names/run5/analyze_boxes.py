import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcome coding:
    # 1 = undemonstrated option (no social information use)
    # 2 = majority option
    # 3 = minority option
    outcome = df["majority_first"]

    # Binary indicators for analyses
    df["uses_social"] = outcome.isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(outcome == 2, 1, np.where(outcome == 3, 0, np.nan))

    # Treat site as a categorical "culture" proxy
    df["site"] = df["y"].astype(int).astype(str)

    # Simple developmental stages based on age in years
    bins = [3, 6, 9, 12, 15]  # 4–6, 7–9, 10–12, 13–14
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)

    print("N total:", len(df))
    print("\nOverall reliance on social information (choose majority or minority):")
    print(df["uses_social"].value_counts(normalize=True).sort_index())

    print("\nOverall majority preference among social learners:")
    social = df[df["uses_social"] == 1].copy()
    print(social["majority_choice"].value_counts(normalize=True).sort_index())

    print("\nReliance on social information by site:")
    print(pd.crosstab(df["site"], df["uses_social"], normalize="index"))

    print("\nReliance on social information by age group:")
    print(pd.crosstab(df["age_group"], df["uses_social"], normalize="index"))

    print("\nMajority preference by site (among social learners):")
    print(pd.crosstab(social["site"], social["majority_choice"], normalize="index"))

    print("\nMajority preference by age group (among social learners):")
    print(pd.crosstab(social["age_group"], social["majority_choice"], normalize="index"))

    # Logistic regression: reliance on social info ~ age + site
    print("\nLogistic regression: uses_social ~ age + C(site)")
    model1 = smf.logit("uses_social ~ age + C(site)", data=df).fit(disp=False)
    print(model1.summary())

    # Logistic regression: majority choice ~ age + site (social learners only)
    print("\nLogistic regression: majority_choice ~ age + C(site) (social learners only)")
    model2 = smf.logit("majority_choice ~ age + C(site)", data=social).fit(disp=False)
    print(model2.summary())

    # Chi-square tests of independence
    print("\nChi-square tests of independence:")

    ct_site_social = pd.crosstab(df["site"], df["uses_social"])
    chi2, p, dof, _ = stats.chi2_contingency(ct_site_social)
    print("uses_social vs site: chi2={:.2f}, dof={}, p={:.4f}".format(chi2, dof, p))

    ct_age_social = pd.crosstab(df["age_group"], df["uses_social"])
    chi2, p, dof, _ = stats.chi2_contingency(ct_age_social)
    print("uses_social vs age_group: chi2={:.2f}, dof={}, p={:.4f}".format(chi2, dof, p))

    ct_site_majority = pd.crosstab(social["site"], social["majority_choice"])
    chi2, p, dof, _ = stats.chi2_contingency(ct_site_majority)
    print("majority_choice vs site: chi2={:.2f}, dof={}, p={:.4f}".format(chi2, dof, p))

    ct_age_majority = pd.crosstab(social["age_group"], social["majority_choice"])
    chi2, p, dof, _ = stats.chi2_contingency(ct_age_majority)
    print("majority_choice vs age_group: chi2={:.2f}, dof={}, p={:.4f}".format(chi2, dof, p))


if __name__ == "__main__":
    main()

