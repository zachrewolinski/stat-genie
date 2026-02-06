import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full, reduced):
    lr_stat = 2 * (full.llf - reduced.llf)
    df_diff = int(full.df_model - reduced.df_model)
    p_value = chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value


def fit_models(df, outcome):
    model_age = smf.glm(f"{outcome} ~ age", data=df, family=sm.families.Binomial()).fit()
    model_culture = smf.glm(f"{outcome} ~ C(culture)", data=df, family=sm.families.Binomial()).fit()
    model_age_culture = smf.glm(f"{outcome} ~ age + C(culture)", data=df, family=sm.families.Binomial()).fit()
    model_interaction = smf.glm(f"{outcome} ~ age * C(culture)", data=df, family=sm.families.Binomial()).fit()

    age_test = lr_test(model_age_culture, model_culture)
    culture_test = lr_test(model_age_culture, model_age)
    interaction_test = lr_test(model_interaction, model_age_culture)

    return {
        "model_age": model_age,
        "model_culture": model_culture,
        "model_age_culture": model_age_culture,
        "model_interaction": model_interaction,
        "age_test": age_test,
        "culture_test": culture_test,
        "interaction_test": interaction_test,
    }


def main():
    df = pd.read_csv("boxes.csv")

    df["social_reliance"] = (df["y"].isin([2, 3])).astype(int)

    df_demo = df[df["y"].isin([2, 3])].copy()
    df_demo["majority_choice"] = (df_demo["y"] == 2).astype(int)

    print("Rows:", len(df))
    print("Social reliance rate:", df["social_reliance"].mean())
    print("Majority preference rate (among demonstrated choices):", df_demo["majority_choice"].mean())

    print("\nSocial reliance by culture:")
    print(df.groupby("culture")["social_reliance"].mean().sort_index())

    print("\nMajority preference by culture (among demonstrated choices):")
    print(df_demo.groupby("culture")["majority_choice"].mean().sort_index())

    print("\nLogistic regression LR tests for social reliance:")
    social_models = fit_models(df, "social_reliance")
    print("Age effect (add age to culture): LR=%.3f, df=%d, p=%.4g" % social_models["age_test"])
    print("Culture effect (add culture to age): LR=%.3f, df=%d, p=%.4g" % social_models["culture_test"])
    print("Age*Culture interaction: LR=%.3f, df=%d, p=%.4g" % social_models["interaction_test"])

    print("\nLogistic regression LR tests for majority preference (among demonstrated choices):")
    majority_models = fit_models(df_demo, "majority_choice")
    print("Age effect (add age to culture): LR=%.3f, df=%d, p=%.4g" % majority_models["age_test"])
    print("Culture effect (add culture to age): LR=%.3f, df=%d, p=%.4g" % majority_models["culture_test"])
    print("Age*Culture interaction: LR=%.3f, df=%d, p=%.4g" % majority_models["interaction_test"])


if __name__ == "__main__":
    main()
