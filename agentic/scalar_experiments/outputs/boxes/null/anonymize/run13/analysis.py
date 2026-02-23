import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site_id",
        }
    )

    # Derived outcomes
    df["majority_choice"] = (df["choice"] == 2).astype(int)
    df["social_choice"] = df["choice"].isin([2, 3]).astype(int)

    # Categorical predictors
    df["site"] = df["site_id"].astype("category")
    df["gender_cat"] = df["gender"].astype("category")

    def fit_models(outcome: str):
        formula_base = f"{outcome} ~ age + C(gender_cat) + majority_first"
        formula_site = formula_base + " + C(site)"

        m0 = smf.logit(formula_base, data=df).fit(disp=False)
        m1 = smf.logit(formula_site, data=df).fit(disp=False)

        lr_stat = 2 * (m1.llf - m0.llf)
        df_diff = m1.df_model - m0.df_model
        p_site = chi2.sf(lr_stat, df_diff)
        age_p = m1.pvalues["age"]

        return m1, age_p, p_site

    maj_model, maj_age_p, maj_site_p = fit_models("majority_choice")
    soc_model, soc_age_p, soc_site_p = fit_models("social_choice")

    # Descriptive statistics by site and age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3.5, 6.5, 9.5, 12.5, 14.5],
        labels=["4-6", "7-9", "10-12", "13-14"],
    )

    majority_by_site = df.groupby("site")["majority_choice"].mean()
    majority_by_age = df.groupby("age_group")["majority_choice"].mean()
    social_by_site = df.groupby("site")["social_choice"].mean()
    social_by_age = df.groupby("age_group")["social_choice"].mean()

    print("=== Logistic regression results ===")
    print(f"Majority choice: age p-value = {maj_age_p:.4g}, site p-value = {maj_site_p:.4g}")
    print(f"Social choice:   age p-value = {soc_age_p:.4g}, site p-value = {soc_site_p:.4g}")
    print()

    print("=== Descriptive statistics ===")
    print("Mean majority-choice rate by site:")
    print(majority_by_site.to_string())
    print()

    print("Mean majority-choice rate by age group:")
    print(majority_by_age.to_string())
    print()

    print("Mean social-choice (any demonstrated option) rate by site:")
    print(social_by_site.to_string())
    print()

    print("Mean social-choice (any demonstrated option) rate by age group:")
    print(social_by_age.to_string())


if __name__ == "__main__":
    main()

