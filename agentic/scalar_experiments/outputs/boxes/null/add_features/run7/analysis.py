import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def likelihood_ratio(full_model, reduced_model):
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df = full_model.df_model - reduced_model.df_model
    p = stats.chi2.sf(lr_stat, df)
    return lr_stat, df, p


def main() -> None:
    base = Path(__file__).parent

    with open(base / "info.json", "r") as f:
        info = json.load(f)

    research_qs = info.get("research_questions", [])
    print("RESEARCH_QUESTIONS", research_qs)

    df = pd.read_csv(base / "boxes.csv")

    print("N_ROWS", len(df))
    print("COLUMNS", ",".join(df.columns))

    # Basic summaries for age and outcome
    if "age" in df.columns:
        age = df["age"]
        print("AGE_MIN", age.min())
        print("AGE_MAX", age.max())
        print("AGE_MEAN", age.mean())
        print("AGE_STD", age.std())

    counts_y = df["y"].value_counts().sort_index()
    for val, cnt in counts_y.items():
        print(f"Y_COUNT_{val}", cnt)

    # Define derived variables
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = np.where(
        df["y"] == 2, 1, np.where(df["y"] == 3, 0, np.nan)
    )

    social_rate_overall = df["social_choice"].mean()
    majority_rate_overall = df["majority_choice"].mean()
    print("SOCIAL_RATE_OVERALL", social_rate_overall)
    print("MAJORITY_RATE_OVERALL", majority_rate_overall)

    # Per-culture descriptive statistics
    if "culture" in df.columns:
        culture_groups = df.groupby("culture")
        for culture_id, g in culture_groups:
            social_rate = g["social_choice"].mean()
            maj_rate = g["majority_choice"].mean()
            n = len(g)
            mean_age = g["age"].mean() if "age" in g.columns else np.nan
            print(f"CULTURE_{culture_id}_N", n)
            print(f"CULTURE_{culture_id}_AGE_MEAN", mean_age)
            print(f"CULTURE_{culture_id}_SOCIAL_RATE", social_rate)
            print(f"CULTURE_{culture_id}_MAJORITY_RATE", maj_rate)

    # Logistic models for reliance on social information
    models_social: dict[str, object] = {}
    if "age" in df.columns and "culture" in df.columns:
        for name, formula in [
            ("main", "social_choice ~ age + C(culture)"),
            ("no_age", "social_choice ~ C(culture)"),
            ("no_culture", "social_choice ~ age"),
            ("inter", "social_choice ~ age * C(culture)"),
        ]:
            try:
                models_social[name] = smf.logit(formula, data=df).fit(disp=False)
            except Exception as e:
                print(f"SOCIAL_MODEL_FAIL_{name}", repr(e))
                models_social[name] = None

        main_model = models_social.get("main")
        if main_model is not None:
            age_coef = main_model.params.get("age", np.nan)
            age_or = float(np.exp(age_coef)) if np.isfinite(age_coef) else np.nan
            age_p = main_model.pvalues.get("age", np.nan)
            print("SOCIAL_AGE_COEF", age_coef)
            print("SOCIAL_AGE_OR", age_or)
            print("SOCIAL_AGE_P", float(age_p))

        # LRTs for age and culture, and interaction
        if (
            models_social.get("main") is not None
            and models_social.get("no_age") is not None
        ):
            lr, df_lr, p_lr = likelihood_ratio(
                models_social["main"], models_social["no_age"]
            )
            print("SOCIAL_LR_AGE", float(lr))
            print("SOCIAL_LR_AGE_DF", int(df_lr))
            print("SOCIAL_LR_AGE_P", float(p_lr))

        if (
            models_social.get("main") is not None
            and models_social.get("no_culture") is not None
        ):
            lr, df_lr, p_lr = likelihood_ratio(
                models_social["main"], models_social["no_culture"]
            )
            print("SOCIAL_LR_CULTURE", float(lr))
            print("SOCIAL_LR_CULTURE_DF", int(df_lr))
            print("SOCIAL_LR_CULTURE_P", float(p_lr))

        if (
            models_social.get("main") is not None
            and models_social.get("inter") is not None
        ):
            lr, df_lr, p_lr = likelihood_ratio(
                models_social["inter"], models_social["main"]
            )
            print("SOCIAL_LR_INTERACTION", float(lr))
            print("SOCIAL_LR_INTERACTION_DF", int(df_lr))
            print("SOCIAL_LR_INTERACTION_P", float(p_lr))

    # Logistic models for preference for majority cues (within social choices)
    social_df = df[df["social_choice"] == 1].copy()
    if len(social_df) > 0 and "age" in social_df.columns and "culture" in social_df.columns:
        models_maj: dict[str, object] = {}
        for name, formula in [
            ("main", "majority_choice ~ age + C(culture)"),
            ("no_age", "majority_choice ~ C(culture)"),
            ("no_culture", "majority_choice ~ age"),
            ("inter", "majority_choice ~ age * C(culture)"),
        ]:
            try:
                models_maj[name] = smf.logit(formula, data=social_df).fit(disp=False)
            except Exception as e:
                print(f"MAJ_MODEL_FAIL_{name}", repr(e))
                models_maj[name] = None

        main_model_m = models_maj.get("main")
        if main_model_m is not None:
            age_coef = main_model_m.params.get("age", np.nan)
            age_or = float(np.exp(age_coef)) if np.isfinite(age_coef) else np.nan
            age_p = main_model_m.pvalues.get("age", np.nan)
            print("MAJ_AGE_COEF", age_coef)
            print("MAJ_AGE_OR", age_or)
            print("MAJ_AGE_P", float(age_p))

        if (
            models_maj.get("main") is not None
            and models_maj.get("no_age") is not None
        ):
            lr, df_lr, p_lr = likelihood_ratio(
                models_maj["main"], models_maj["no_age"]
            )
            print("MAJ_LR_AGE", float(lr))
            print("MAJ_LR_AGE_DF", int(df_lr))
            print("MAJ_LR_AGE_P", float(p_lr))

        if (
            models_maj.get("main") is not None
            and models_maj.get("no_culture") is not None
        ):
            lr, df_lr, p_lr = likelihood_ratio(
                models_maj["main"], models_maj["no_culture"]
            )
            print("MAJ_LR_CULTURE", float(lr))
            print("MAJ_LR_CULTURE_DF", int(df_lr))
            print("MAJ_LR_CULTURE_P", float(p_lr))

        if (
            models_maj.get("main") is not None
            and models_maj.get("inter") is not None
        ):
            lr, df_lr, p_lr = likelihood_ratio(
                models_maj["inter"], models_maj["main"]
            )
            print("MAJ_LR_INTERACTION", float(lr))
            print("MAJ_LR_INTERACTION_DF", int(df_lr))
            print("MAJ_LR_INTERACTION_P", float(p_lr))


if __name__ == "__main__":
    main()

