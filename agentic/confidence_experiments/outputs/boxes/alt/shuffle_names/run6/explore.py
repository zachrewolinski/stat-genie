import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcomes
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)
    df["site"] = df["y"].astype("category")

    print("N =", len(df))
    print("Overall majority choice rate:", df["majority_choice"].mean())
    print("Overall social choice rate:", df["social_choice"].mean())

    # Logistic regression: majority choice ~ age
    model_age = smf.logit("majority_choice ~ age", data=df).fit(disp=False)
    print("\nLogit majority ~ age")
    print(model_age.summary())

    # Logistic regression: majority choice ~ age + site (culture proxy)
    model_age_site = smf.logit("majority_choice ~ age + C(site)", data=df).fit(
        disp=False
    )
    print("\nLogit majority ~ age + site")
    print(model_age_site.summary())

    # Compare models with and without site
    lr_stat = 2 * (model_age_site.llf - model_age.llf)
    df_diff = model_age_site.df_model - model_age.df_model
    from scipy.stats import chi2

    p_lr = chi2.sf(lr_stat, df_diff)
    print("\nLR test for adding site (majority):")
    print("LR stat:", lr_stat, "df:", df_diff, "p:", p_lr)

    # Social (any demonstrated) vs asocial choice
    model_social_age = smf.logit("social_choice ~ age", data=df).fit(disp=False)
    print("\nLogit social ~ age")
    print(model_social_age.summary())

    model_social_age_site = smf.logit(
        "social_choice ~ age + C(site)", data=df
    ).fit(disp=False)
    print("\nLogit social ~ age + site")
    print(model_social_age_site.summary())

    lr_stat_social = 2 * (model_social_age_site.llf - model_social_age.llf)
    df_diff_social = model_social_age_site.df_model - model_social_age.df_model
    p_lr_social = chi2.sf(lr_stat_social, df_diff_social)
    print("\nLR test for adding site (social):")
    print("LR stat:", lr_stat_social, "df:", df_diff_social, "p:", p_lr_social)


if __name__ == "__main__":
    main()
