import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived variables
    df["is_social"] = (df["y"] != 1).astype(int)
    df["is_majority"] = (df["y"] == 2).astype(int)

    # Define coarse age groups to capture developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 12, 15],
        labels=["4-6", "7-9", "10-12", "13-14"],
        include_lowest=True,
    )

    print("N observations:", len(df))
    print("\nOverall outcome distribution (1=unchosen, 2=majority, 3=minority):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nOutcome by culture (row = culture):")
    by_culture = pd.crosstab(df["culture"], df["y"])
    print(by_culture)
    chi2_culture_y, p_culture_y, dof_culture_y, _ = stats.chi2_contingency(by_culture)
    print("\nChi-square test: culture x outcome")
    print("chi2 =", chi2_culture_y, "df =", dof_culture_y, "p =", p_culture_y)

    print("\nOutcome by age group:")
    by_age = pd.crosstab(df["age_group"], df["y"])
    print(by_age)
    chi2_age_y, p_age_y, dof_age_y, _ = stats.chi2_contingency(by_age)
    print("\nChi-square test: age_group x outcome")
    print("chi2 =", chi2_age_y, "df =", dof_age_y, "p =", p_age_y)

    # Logistic regression: reliance on any social information
    print("\nLogistic regression: any social info (y != 1)")
    model_social = smf.logit("is_social ~ age + C(culture)", data=df).fit(disp=False)
    print(model_social.summary())

    # Logistic regression: majority preference among trials where social info is used
    print("\nLogistic regression: majority vs minority among social choices")
    social_df = df[df["is_social"] == 1].copy()
    model_majority = smf.logit("is_majority ~ age + C(culture)", data=social_df).fit(
        disp=False
    )
    print(model_majority.summary())


if __name__ == "__main__":
    main()

