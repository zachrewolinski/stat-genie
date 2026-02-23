import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Encode key derived variables
    df["social"] = (df["y"] != 1).astype(int)  # 1 = followed any demonstrator
    df_social = df[df["social"] == 1].copy()
    df_social["majority"] = (df_social["y"] == 2).astype(int)  # 1 = majority, 0 = minority

    print("=== Basic counts ===")
    print("Total N:", len(df))
    print("Social choices (y != 1):", df["social"].sum())
    print("Majority choices among social:", df_social["majority"].sum())

    # Chi-square tests for association between choice and culture / age group
    print("\n=== Chi-square: full outcome (y) by culture ===")
    ct_culture_y = pd.crosstab(df["culture"], df["y"])
    chi2_c, p_c, dof_c, _ = stats.chi2_contingency(ct_culture_y)
    print("Chi2 =", chi2_c, "df =", dof_c, "p =", p_c)

    print("\n=== Chi-square: full outcome (y) by age group ===")
    df["age_group"] = pd.cut(df["age"], bins=[4, 6, 8, 10, 12, 14], right=True, include_lowest=True)
    ct_age_y = pd.crosstab(df["age_group"], df["y"])
    chi2_a, p_a, dof_a, _ = stats.chi2_contingency(ct_age_y)
    print("Chi2 =", chi2_a, "df =", dof_a, "p =", p_a)

    # Logistic regression: reliance on social information (any demonstrator vs undemonstrated)
    print("\n=== Logistic regression: social (any demonstrator) ~ age + culture ===")
    model_social = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    print(model_social.summary())

    # Likelihood-ratio test for contribution of culture in the social model
    model_social_nocult = smf.logit("social ~ age", data=df).fit(disp=False)
    lr_stat = 2 * (model_social.llf - model_social_nocult.llf)
    lr_p = stats.chi2.sf(lr_stat, df["culture"].nunique() - 1)
    print("\nLR test for adding culture to social model: LR =", lr_stat, "p =", lr_p)

    # Logistic regression: majority vs minority among social choices
    print("\n=== Logistic regression: majority (vs minority) ~ age + culture ===")
    model_maj = smf.logit("majority ~ age + C(culture)", data=df_social).fit(disp=False)
    print(model_maj.summary())

    # LR test for culture in majority-preference model
    model_maj_nocult = smf.logit("majority ~ age", data=df_social).fit(disp=False)
    lr_stat_m = 2 * (model_maj.llf - model_maj_nocult.llf)
    lr_p_m = stats.chi2.sf(lr_stat_m, df_social["culture"].nunique() - 1)
    print("\nLR test for adding culture to majority model: LR =", lr_stat_m, "p =", lr_p_m)


if __name__ == "__main__":
    main()

