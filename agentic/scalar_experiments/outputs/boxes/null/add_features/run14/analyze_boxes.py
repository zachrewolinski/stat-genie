import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def lr_test(model_restricted, model_full):
    """Likelihood ratio test comparing two nested models."""
    lr_stat = 2.0 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def main():
    df = pd.read_csv("boxes.csv")

    # Define key derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Basic description
    print("N total:", len(df))
    print("Columns:", list(df.columns))
    print()

    # --- Model 1: Reliance on social information (any demonstrated option vs undemonstrated) ---
    print("=== Model 1: Social reliance (social = 1 if y != 1) ===")
    model_social_full = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
    model_social_no_age = smf.logit("social ~ C(culture)", data=df).fit(disp=False)
    model_social_no_culture = smf.logit("social ~ age", data=df).fit(disp=False)

    lr_age_social = lr_test(model_social_no_age, model_social_full)
    lr_cult_social = lr_test(model_social_no_culture, model_social_full)

    print("Social model - age coef:", float(model_social_full.params["age"]))
    print(
        "Social model - LR test age (vs. model without age): "
        f"LR={lr_age_social[0]:.3f}, df={lr_age_social[1]:.0f}, p={lr_age_social[2]:.4g}"
    )
    print(
        "Social model - LR test culture (vs. model without culture): "
        f"LR={lr_cult_social[0]:.3f}, df={lr_cult_social[1]:.0f}, p={lr_cult_social[2]:.4g}"
    )

    # Predicted probabilities of social reliance at min/max age (holding culture at mode)
    age_min, age_max = df["age"].min(), df["age"].max()
    culture_mode = df["culture"].mode().iloc[0]
    new_social = pd.DataFrame(
        {"age": [age_min, age_max], "culture": [culture_mode, culture_mode]}
    )
    pred_social = model_social_full.predict(new_social)
    print(
        f"Predicted Pr(social=1) at age {age_min:.1f}: {pred_social.iloc[0]:.3f}; "
        f"at age {age_max:.1f}: {pred_social.iloc[1]:.3f}"
    )
    print()

    # --- Model 2: Preference for majority vs minority among social choosers ---
    print("=== Model 2: Majority preference among social choosers ===")
    social_df = df[df["social"] == 1].copy()
    print("N social choosers:", len(social_df))

    model_maj_full = smf.logit(
        "majority_choice ~ age + C(culture)", data=social_df
    ).fit(disp=False)
    model_maj_no_age = smf.logit(
        "majority_choice ~ C(culture)", data=social_df
    ).fit(disp=False)
    model_maj_no_culture = smf.logit(
        "majority_choice ~ age", data=social_df
    ).fit(disp=False)

    lr_age_maj = lr_test(model_maj_no_age, model_maj_full)
    lr_cult_maj = lr_test(model_maj_no_culture, model_maj_full)

    print("Majority model - age coef:", float(model_maj_full.params["age"]))
    print(
        "Majority model - LR test age (vs. model without age): "
        f"LR={lr_age_maj[0]:.3f}, df={lr_age_maj[1]:.0f}, p={lr_age_maj[2]:.4g}"
    )
    print(
        "Majority model - LR test culture (vs. model without culture): "
        f"LR={lr_cult_maj[0]:.3f}, df={lr_cult_maj[1]:.0f}, p={lr_cult_maj[2]:.4g}"
    )

    # Predicted probabilities of choosing the majority by age and culture
    age_min_s, age_max_s = social_df["age"].min(), social_df["age"].max()
    new_maj = pd.DataFrame(
        {"age": [age_min_s, age_max_s], "culture": [culture_mode, culture_mode]}
    )
    pred_maj = model_maj_full.predict(new_maj)
    print(
        f"Predicted Pr(majority_choice=1 | social) at age {age_min_s:.1f}: "
        f"{pred_maj.iloc[0]:.3f}; at age {age_max_s:.1f}: {pred_maj.iloc[1]:.3f}"
    )

    # Variation across cultures at mean age
    mean_age = social_df["age"].mean()
    cultures = np.sort(social_df["culture"].unique())
    culture_grid = pd.DataFrame(
        {"age": np.repeat(mean_age, len(cultures)), "culture": cultures}
    )
    pred_maj_cult = model_maj_full.predict(culture_grid)
    print("Predicted Pr(majority_choice=1 | social) by culture at mean age:")
    for c_val, p_val in zip(cultures, pred_maj_cult):
        print(f"  culture {int(c_val)}: {p_val:.3f}")


if __name__ == "__main__":
    main()

