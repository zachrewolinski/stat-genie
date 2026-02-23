import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def fit_models(outcome: str, df: pd.DataFrame, desc: str) -> None:
    print(f"\n=== Outcome: {desc} ===")
    formula_full = f"{outcome} ~ age + C(culture)"
    formula_age_only = f"{outcome} ~ age"
    formula_culture_only = f"{outcome} ~ C(culture)"

    model_full = smf.logit(formula_full, data=df).fit(disp=False)
    model_age_only = smf.logit(formula_age_only, data=df).fit(disp=False)
    model_culture_only = smf.logit(formula_culture_only, data=df).fit(disp=False)

    lr_age_stat = 2 * (model_full.llf - model_culture_only.llf)
    lr_age_df = model_full.df_model - model_culture_only.df_model
    lr_age_p = chi2.sf(lr_age_stat, lr_age_df)

    lr_culture_stat = 2 * (model_full.llf - model_age_only.llf)
    lr_culture_df = model_full.df_model - model_age_only.df_model
    lr_culture_p = chi2.sf(lr_culture_stat, lr_culture_df)

    print("N =", len(df))
    print("Age coef (full model):", float(model_full.params.get("age", float("nan"))))
    print("Age p-value (full model):", float(model_full.pvalues.get("age", float("nan"))))
    print(
        "LR test for adding age given culture: "
        "stat=%.3f, p=%.5f, df=%d" % (lr_age_stat, lr_age_p, lr_age_df)
    )
    print(
        "LR test for adding culture given age: "
        "stat=%.3f, p=%.5f, df=%d" % (lr_culture_stat, lr_culture_p, lr_culture_df)
    )

    age_min, age_max = df["age"].min(), df["age"].max()
    mode_culture = df["culture"].mode().iloc[0]

    pred_low = model_full.predict(
        pd.DataFrame({"age": [age_min], "culture": [mode_culture]})
    )[0]
    pred_high = model_full.predict(
        pd.DataFrame({"age": [age_max], "culture": [mode_culture]})
    )[0]

    print(
        "Predicted prob at age min=%.1f: %.3f; at age max=%.1f: %.3f"
        % (age_min, pred_low, age_max, pred_high)
    )


def main() -> None:
    df = pd.read_csv("boxes.csv")

    df["rely_social"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    df_rely = df[df["rely_social"] == 1].copy()
    df_rely["majority_given_rely"] = (df_rely["y"] == 2).astype(int)

    print("Head of data:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe(include="all"))

    fit_models(
        "rely_social",
        df,
        "Reliance on any social info (majority or minority vs undemonstrated)",
    )
    fit_models(
        "majority_choice",
        df,
        "Choosing majority option vs others",
    )
    fit_models(
        "majority_given_rely",
        df_rely,
        "Choosing majority vs minority conditional on using social info",
    )


if __name__ == "__main__":
    main()
