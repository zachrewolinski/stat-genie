import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


def lr_test(model_restricted, model_full):
    """Likelihood-ratio test comparing two nested models."""
    lr = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = stats.chi2.sf(lr, df_diff)
    return lr, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Outcome 1: reliance on social information
    # 1 = unchosen option (individual), 2 = majority, 3 = minority (social)
    df["social_choice"] = df["majority_first"].isin([2, 3]).astype(int)

    # Outcome 2: preference for majority vs minority among social choices
    df_mm = df[df["majority_first"].isin([2, 3])].copy()
    df_mm["majority_choice"] = (df_mm["majority_first"] == 2).astype(int)

    # Treat site ID as a categorical proxy for culture
    df["site"] = df["y"].astype("category")
    df_mm["site"] = df_mm["y"].astype("category")

    print("N total:", len(df))
    print("N social choices:", df["social_choice"].sum())
    print("N majority-or-minority choices used in majority model:", len(df_mm))
    print()

    # Model A: reliance on social information ~ age + site
    model_social_base = smf.glm(
        "social_choice ~ 1", data=df, family=sm.families.Binomial()
    ).fit()
    model_social_age = smf.glm(
        "social_choice ~ age", data=df, family=sm.families.Binomial()
    ).fit()
    model_social_age_site = smf.glm(
        "social_choice ~ age + C(site)", data=df, family=sm.families.Binomial()
    ).fit()

    lr_age_social, df_age_social, p_age_social = lr_test(
        model_social_base, model_social_age
    )
    lr_site_social, df_site_social, p_site_social = lr_test(
        model_social_age, model_social_age_site
    )

    print("=== Reliance on social information (social_choice) ===")
    print("Age effect (vs intercept-only):")
    print(f"  LR = {lr_age_social:.3f}, df = {df_age_social}, p = {p_age_social:.6f}")
    print("Site (culture proxy) effect given age:")
    print(f"  LR = {lr_site_social:.3f}, df = {df_site_social}, p = {p_site_social:.6f}")
    print("Coefficients (age + site model):")
    print(model_social_age_site.params)
    print()

    # Simple age effect sizes: predicted probabilities at youngest vs oldest ages
    ages = [df["age"].min(), df["age"].max()]
    pred_df = pd.DataFrame({"age": ages})
    # Use reference site (first category) for illustration
    pred_df["site"] = df["site"].cat.categories[0]
    preds = model_social_age_site.predict(pred_df)
    print("Predicted probability of any social choice by age (holding site constant):")
    for age_val, p_val in zip(ages, preds):
        print(f"  Age {age_val:.0f}: p(social) ≈ {p_val:.3f}")
    print()

    # Model B: preference for majority vs minority among social choices ~ age + site
    model_maj_base = smf.glm(
        "majority_choice ~ 1", data=df_mm, family=sm.families.Binomial()
    ).fit()
    model_maj_age = smf.glm(
        "majority_choice ~ age", data=df_mm, family=sm.families.Binomial()
    ).fit()
    model_maj_age_site = smf.glm(
        "majority_choice ~ age + C(site)", data=df_mm, family=sm.families.Binomial()
    ).fit()

    lr_age_maj, df_age_maj, p_age_maj = lr_test(model_maj_base, model_maj_age)
    lr_site_maj, df_site_maj, p_site_maj = lr_test(model_maj_age, model_maj_age_site)

    print("=== Preference for majority vs minority (majority_choice) ===")
    print("Age effect (vs intercept-only):")
    print(f"  LR = {lr_age_maj:.3f}, df = {df_age_maj}, p = {p_age_maj:.6f}")
    print("Site (culture proxy) effect given age:")
    print(f"  LR = {lr_site_maj:.3f}, df = {df_site_maj}, p = {p_site_maj:.6f}")
    print("Coefficients (age + site model):")
    print(model_maj_age_site.params)
    print()

    ages_mm = [df_mm["age"].min(), df_mm["age"].max()]
    pred_df_mm = pd.DataFrame({"age": ages_mm})
    pred_df_mm["site"] = df_mm["site"].cat.categories[0]
    preds_mm = model_maj_age_site.predict(pred_df_mm)
    print(
        "Predicted probability of choosing the majority option (vs minority) by age "
        "(holding site constant):"
    )
    for age_val, p_val in zip(ages_mm, preds_mm):
        print(f"  Age {age_val:.0f}: p(majority | social) ≈ {p_val:.3f}")


if __name__ == "__main__":
    main()

