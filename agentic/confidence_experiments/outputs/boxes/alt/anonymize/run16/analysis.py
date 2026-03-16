import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


def lr_test(model_reduced, model_full):
    """Likelihood-ratio test comparing nested models."""
    lr_stat = 2 * (model_full.llf - model_reduced.llf)
    df_diff = int(model_full.df_model - model_reduced.df_model)
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def analyze_social_reliance(df: pd.DataFrame):
    """
    Social information reliance:
    1/0 variable: 1 if child chose any demonstrated option (majority or minority),
    0 if they chose the undemonstrated option.
    """
    data = df.copy()
    data["social_choice"] = (data["feature1"] != 1).astype(int)

    # Descriptive statistics
    overall_mean = data["social_choice"].mean()
    by_site = data.groupby("feature5")["social_choice"].mean()

    # Logistic regression: social_choice ~ age (+ site)
    model_age = smf.logit("social_choice ~ feature3", data=data).fit(disp=False)
    model_age_site = smf.logit("social_choice ~ feature3 + C(feature5)", data=data).fit(
        disp=False
    )

    lr_site, df_site, p_site = lr_test(model_age, model_age_site)
    age_coef = model_age.params["feature3"]
    age_p = model_age.pvalues["feature3"]

    # Effect of age: predicted probabilities at younger vs older ages
    age_young = 5
    age_old = 12
    pred_young = model_age.predict(exog=dict(feature3=age_young)).mean()
    pred_old = model_age.predict(exog=dict(feature3=age_old)).mean()

    return {
        "overall_mean": overall_mean,
        "by_site": by_site.to_dict(),
        "age_coef": float(age_coef),
        "age_p": float(age_p),
        "site_lr_stat": float(lr_site),
        "site_lr_df": df_site,
        "site_lr_p": float(p_site),
        "age_pred_young": float(pred_young),
        "age_pred_old": float(pred_old),
    }


def analyze_majority_preference(df: pd.DataFrame):
    """
    Majority preference among children who relied on social information.
    Binary variable: 1 if majority option chosen, 0 if minority option chosen.
    """
    data = df.copy()
    data = data[data["feature1"].isin([2, 3])].copy()
    data["majority_choice"] = (data["feature1"] == 2).astype(int)

    # Descriptive statistics
    overall_mean = data["majority_choice"].mean()
    by_site = data.groupby("feature5")["majority_choice"].mean()

    # Logistic regression: majority_choice ~ age (+ site)
    model_age = smf.logit("majority_choice ~ feature3", data=data).fit(disp=False)
    model_age_site = smf.logit(
        "majority_choice ~ feature3 + C(feature5)", data=data
    ).fit(disp=False)

    lr_site, df_site, p_site = lr_test(model_age, model_age_site)
    age_coef = model_age.params["feature3"]
    age_p = model_age.pvalues["feature3"]

    # Effect of age: predicted probabilities at younger vs older ages
    age_young = 5
    age_old = 12
    pred_young = model_age.predict(exog=dict(feature3=age_young)).mean()
    pred_old = model_age.predict(exog=dict(feature3=age_old)).mean()

    return {
        "overall_mean": overall_mean,
        "by_site": by_site.to_dict(),
        "age_coef": float(age_coef),
        "age_p": float(age_p),
        "site_lr_stat": float(lr_site),
        "site_lr_df": df_site,
        "site_lr_p": float(p_site),
        "age_pred_young": float(pred_young),
        "age_pred_old": float(pred_old),
    }


def main():
    df = pd.read_csv("boxes.csv")

    social_results = analyze_social_reliance(df)
    majority_results = analyze_majority_preference(df)

    print("=== Social Information Reliance (any social vs undemonstrated) ===")
    print(f"Overall proportion using social information: {social_results['overall_mean']:.3f}")
    print("Proportion by site (feature5):")
    for site, val in sorted(social_results["by_site"].items()):
        print(f"  Site {site}: {val:.3f}")
    print(
        f"Logit age coefficient (social_choice ~ age): "
        f"{social_results['age_coef']:.3f}, p = {social_results['age_p']:.3g}"
    )
    print(
        "Likelihood-ratio test for adding site (C(feature5)):\n"
        f"  LR chi2({social_results['site_lr_df']}) = {social_results['site_lr_stat']:.3f}, "
        f"p = {social_results['site_lr_p']:.3g}"
    )
    print(
        "Predicted probability of using social information:\n"
        f"  At age 5:  {social_results['age_pred_young']:.3f}\n"
        f"  At age 12: {social_results['age_pred_old']:.3f}"
    )

    print("\n=== Majority Preference (majority vs minority among social choosers) ===")
    print(
        "Overall proportion choosing majority option (among those "
        "who followed any demonstrator): "
        f"{majority_results['overall_mean']:.3f}"
    )
    print("Proportion by site (feature5):")
    for site, val in sorted(majority_results["by_site"].items()):
        print(f"  Site {site}: {val:.3f}")
    print(
        f"Logit age coefficient (majority_choice ~ age): "
        f"{majority_results['age_coef']:.3f}, p = {majority_results['age_p']:.3g}"
    )
    print(
        "Likelihood-ratio test for adding site (C(feature5)):\n"
        f"  LR chi2({majority_results['site_lr_df']}) = {majority_results['site_lr_stat']:.3f}, "
        f"p = {majority_results['site_lr_p']:.3g}"
    )
    print(
        "Predicted probability of choosing majority option:\n"
        f"  At age 5:  {majority_results['age_pred_young']:.3f}\n"
        f"  At age 12: {majority_results['age_pred_old']:.3f}"
    )


if __name__ == "__main__":
    main()

