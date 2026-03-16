import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def load_data(path: str = "boxes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Reliance on social information: 1 if child follows any demonstrator (majority or minority),
    # 0 if child chooses undemonstrated third option.
    df = df.copy()
    df["social_reliance"] = np.where(df["y"].isin([2, 3]), 1, 0)

    # Majority preference among those who relied on social information.
    mask_social = df["y"].isin([2, 3])
    df_social = df.loc[mask_social].copy()
    df_social["majority_choice"] = np.where(df_social["y"] == 2, 1, 0)

    return df, df_social


def fit_logit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data).fit(disp=False)
    return model


def summarize_effects(model, variable_prefix: str):
    """
    Collect p-values and odds ratios for terms whose names start with variable_prefix.
    This is useful for culture dummies (C(culture)) and interactions (age:C(culture)).
    """
    summary = []
    params = model.params
    conf = model.conf_int()
    pvalues = model.pvalues

    for name in params.index:
        if name.startswith(variable_prefix):
            odds_ratio = np.exp(params[name])
            ci_low = np.exp(conf.loc[name, 0])
            ci_high = np.exp(conf.loc[name, 1])
            pval = pvalues[name]
            summary.append(
                {
                    "term": name,
                    "odds_ratio": odds_ratio,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": pval,
                }
            )
    return summary


def main():
    df = load_data()
    df, df_social = prepare_variables(df)

    # Center age to make interaction terms easier to interpret.
    df["age_c"] = df["age"] - df["age"].mean()
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()

    # Treat culture as categorical.
    df["culture"] = df["culture"].astype("category")
    df_social["culture"] = df_social["culture"].astype("category")

    # Model 1: social reliance ~ age + culture
    model_reliance = fit_logit("social_reliance ~ age_c + C(culture)", df)

    # Model 2: majority preference ~ age + culture
    model_majority = fit_logit("majority_choice ~ age_c + C(culture)", df_social)

    print("=== Model 1: Social reliance (any demonstrator) ===")
    print(model_reliance.summary())
    print("\nAge effect (social reliance):")
    age_or = float(np.exp(model_reliance.params["age_c"]))
    age_ci = np.exp(model_reliance.conf_int().loc["age_c"].values)
    age_p = float(model_reliance.pvalues["age_c"])
    print(
        f"  OR per 1-year increase: {age_or:.3f} "
        f"(95% CI {age_ci[0]:.3f}–{age_ci[1]:.3f}), p = {age_p:.4f}"
    )

    culture_effects_reliance = summarize_effects(model_reliance, "C(culture)")
    print("\nCulture effects (social reliance):")
    for eff in culture_effects_reliance:
        print(
            f"  {eff['term']}: OR = {eff['odds_ratio']:.3f} "
            f"(95% CI {eff['ci_low']:.3f}–{eff['ci_high']:.3f}), "
            f"p = {eff['p_value']:.4f}"
        )

    print("\n=== Model 2: Majority preference (vs minority, among social learners) ===")
    print(model_majority.summary())
    print("\nAge effect (majority preference):")
    age_or_m = float(np.exp(model_majority.params["age_c"]))
    age_ci_m = np.exp(model_majority.conf_int().loc["age_c"].values)
    age_p_m = float(model_majority.pvalues["age_c"])
    print(
        f"  OR per 1-year increase: {age_or_m:.3f} "
        f"(95% CI {age_ci_m[0]:.3f}–{age_ci_m[1]:.3f}), p = {age_p_m:.4f}"
    )

    culture_effects_majority = summarize_effects(model_majority, "C(culture)")
    print("\nCulture effects (majority preference):")
    for eff in culture_effects_majority:
        print(
            f"  {eff['term']}: OR = {eff['odds_ratio']:.3f} "
            f"(95% CI {eff['ci_low']:.3f}–{eff['ci_high']:.3f}), "
            f"p = {eff['p_value']:.4f}"
        )


if __name__ == "__main__":
    main()

