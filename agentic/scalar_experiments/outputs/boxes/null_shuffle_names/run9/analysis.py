import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcomes
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)

    print("Basic counts")
    print(df[["majority_choice", "social_choice"]].mean())
    print()

    # Age effects (developmental stage)
    age_maj_corr = df["age"].corr(df["majority_choice"])
    age_soc_corr = df["age"].corr(df["social_choice"])

    print("Age correlations")
    print(f"corr(age, majority_choice) = {age_maj_corr:.3f}")
    print(f"corr(age, social_choice)   = {age_soc_corr:.3f}")
    print()

    # Site as proxy for cultural context (column y)
    print("Mean outcomes by site (y)")
    site_means = df.groupby("y")[["majority_choice", "social_choice"]].mean()
    print(site_means)
    print()

    # Chi-square tests for variation across sites
    ct_majority = pd.crosstab(df["y"], df["majority_choice"])
    chi2_maj, p_maj, dof_maj, _ = stats.chi2_contingency(ct_majority)

    ct_social = pd.crosstab(df["y"], df["social_choice"])
    chi2_soc, p_soc, dof_soc, _ = stats.chi2_contingency(ct_social)

    print("Chi-square tests for site differences")
    print("Majority choice across sites:")
    print(f"  chi2 = {chi2_maj:.2f}, dof = {dof_maj}, p = {p_maj:.3g}")
    print("Social vs undemonstrated choice across sites:")
    print(f"  chi2 = {chi2_soc:.2f}, dof = {dof_soc}, p = {p_soc:.3g}")
    print()

    # Logistic regression models with age and site as predictors
    print("Logistic regression: majority_choice ~ age + C(y)")
    model_maj = smf.logit("majority_choice ~ age + C(y)", data=df).fit(disp=False)
    print(model_maj.summary())
    print()

    print("Logistic regression: social_choice ~ age + C(y)")
    model_soc = smf.logit("social_choice ~ age + C(y)", data=df).fit(disp=False)
    print(model_soc.summary())


if __name__ == "__main__":
    main()

