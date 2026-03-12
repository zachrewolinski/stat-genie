import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "choice",
            "feature2": "gender",
            "feature3": "age",
            "feature4": "majority_first",
            "feature5": "site",
        }
    )

    # Basic derived variables
    df["social_choice"] = (df["choice"] != 1).astype(int)
    df["majority_choice"] = (df["choice"] == 2).astype(int)

    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    social_formula = "social_choice ~ age + C(site) + C(gender) + majority_first"
    social_model = smf.logit(social_formula, data=df).fit(disp=False)

    # Restrict to trials where a social option was chosen to study majority preference
    df_social = df[df["social_choice"] == 1].copy()
    majority_formula = (
        "majority_choice ~ age + C(site) + C(gender) + majority_first"
    )
    majority_model = smf.logit(majority_formula, data=df_social).fit(disp=False)

    # Chi-square tests for culture effects as a robustness check
    site_social_table = pd.crosstab(df["site"], df["social_choice"])
    chi2_social, p_social, _, _ = stats.chi2_contingency(site_social_table)

    site_majority_table = pd.crosstab(df_social["site"], df_social["majority_choice"])
    chi2_majority, p_majority, _, _ = stats.chi2_contingency(site_majority_table)

    print("=== Descriptive counts ===")
    print("N total:", len(df))
    print(df["choice"].value_counts().sort_index())
    print("\n=== Social-choice logistic regression ===")
    print(social_model.summary())
    print("\nLikelihood-ratio test for social model vs null:")
    print(social_model.llr, social_model.llr_pvalue)
    print("\nChi-square test of site x social_choice:")
    print("chi2 =", chi2_social, "p =", p_social)

    print("\n=== Majority-choice logistic regression (social choices only) ===")
    print("N social choices:", len(df_social))
    print(df_social["choice"].value_counts().sort_index())
    print(majority_model.summary())
    print("\nLikelihood-ratio test for majority model vs null:")
    print(majority_model.llr, majority_model.llr_pvalue)
    print("\nChi-square test of site x majority_choice:")
    print("chi2 =", chi2_majority, "p =", p_majority)


if __name__ == "__main__":
    main()

