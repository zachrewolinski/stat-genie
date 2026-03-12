import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_logit(formula: str, data: pd.DataFrame):
    """Fit a logistic regression and print a compact summary."""
    print(f"\n=== Logistic regression: {formula} ===")
    try:
        model = smf.logit(formula, data=data).fit(disp=False)
    except Exception as exc:  # catch convergence or separation issues
        print(f"Model failed to converge: {exc}")
        return None

    print(model.summary())
    print(f"McFadden pseudo R2: {model.prsquared:.3f}")
    print(f"LLR p-value vs intercept-only: {model.llr_pvalue:.4g}")
    return model


def main():
    df = pd.read_csv("boxes.csv")

    # Outcome encodings
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_use"] = (df["majority_first"] != 1).astype(int)

    # Among children who copied any demonstrator, preference for majority vs minority
    df_mm = df[df["majority_first"] != 1].copy()
    df_mm["majority_pref"] = (df_mm["majority_first"] == 2).astype(int)

    # Age groups for descriptive summaries
    bins = [3, 6, 9, 14]
    labels = ["4-6", "7-9", "10-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=False)
    df_mm["age_group"] = pd.cut(df_mm["age"], bins=bins, labels=labels, include_lowest=False)

    print("=== Basic counts ===")
    print(df["majority_first"].value_counts().sort_index())
    print("\n=== Mean tendencies by age group (all children) ===")
    print(
        df.groupby("age_group")[["majority_choice", "social_use"]]
        .mean()
        .rename(columns={"majority_choice": "P(majority)", "social_use": "P(any social)"})
    )

    print("\n=== Mean majority preference among social users by age group ===")
    print(
        df_mm.groupby("age_group")[["majority_pref"]]
        .mean()
        .rename(columns={"majority_pref": "P(majority | social use)"})
    )

    print("\n=== Mean tendencies by site (proxy for culture) ===")
    print(
        df.groupby("y")[["majority_choice", "social_use"]]
        .mean()
        .rename(columns={"majority_choice": "P(majority)", "social_use": "P(any social)"})
    )

    print("\n=== Mean majority preference among social users by site ===")
    print(
        df_mm.groupby("y")[["majority_pref"]]
        .mean()
        .rename(columns={"majority_pref": "P(majority | social use)"})
    )

    # Logistic regressions: developmental (age) trends
    fit_logit("majority_choice ~ age", df)
    fit_logit("social_use ~ age", df)
    fit_logit("majority_pref ~ age", df_mm)

    # Logistic regressions: cross-cultural/site differences (treat y as site/culture ID)
    fit_logit("majority_choice ~ C(y)", df)
    fit_logit("social_use ~ C(y)", df)
    fit_logit("majority_pref ~ C(y)", df_mm)


if __name__ == "__main__":
    main()

