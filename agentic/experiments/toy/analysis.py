import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "compas.csv"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only Black (African-American) and White (Caucasian) defendants
    df = df.copy()
    df = df[(df["race:African-American"] == 1) | (df["race:Caucasian"] == 1)]
    # Binary indicator for Black
    df["black"] = (df["race:African-American"] == 1).astype(int)
    return df


def fit_logit(df: pd.DataFrame, features: list[str]):
    X = df[features]
    X = sm.add_constant(X, has_constant="add")
    y = df["two_year_recid"].astype(float)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def main():
    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    # Control variables representing criminal history and background
    controls = [
        "age",
        "priors_count",
        "days_b_screening_arrest",
        "c_jail_time",
        "juv_fel_count",
        "juv_other_count",
        "juv_misd_count",
        "c_charge_degree:F",
        "c_charge_degree:M",
        "age_cat:25_-_45",
        "age_cat:Greater_than_45",
        "age_cat:Less_than_25",
        "sex:Female",
        "sex:Male",
    ]

    features = controls + ["black"]
    result = fit_logit(df, features)

    coef_black = result.params["black"]
    pval_black = result.pvalues["black"]

    # Counterfactual predicted risks holding covariates fixed
    X_base = df[controls]
    X_black = X_base.copy()
    X_white = X_base.copy()
    X_black["black"] = 1
    X_white["black"] = 0

    X_black = sm.add_constant(X_black, has_constant="add")
    X_white = sm.add_constant(X_white, has_constant="add")

    pred_black = result.predict(X_black)
    pred_white = result.predict(X_white)

    avg_pred_black = pred_black.mean()
    avg_pred_white = pred_white.mean()
    avg_diff = avg_pred_black - avg_pred_white

    # High-risk classification at 0.5 threshold for comparison
    high_risk_black = (pred_black >= 0.5).mean()
    high_risk_white = (pred_white >= 0.5).mean()
    high_risk_diff = high_risk_black - high_risk_white

    summary = {
        "n": int(len(df)),
        "coef_black": float(coef_black),
        "pval_black": float(pval_black),
        "avg_pred_black": float(avg_pred_black),
        "avg_pred_white": float(avg_pred_white),
        "avg_pred_diff": float(avg_diff),
        "high_risk_black": float(high_risk_black),
        "high_risk_white": float(high_risk_white),
        "high_risk_diff": float(high_risk_diff),
    }

    # Print a concise summary for inspection
    print("Logit coef (black):", round(coef_black, 4), "p=", round(pval_black, 6))
    print("Avg predicted risk (black):", round(avg_pred_black, 4))
    print("Avg predicted risk (white):", round(avg_pred_white, 4))
    print("Avg predicted diff:", round(avg_diff, 4))
    print("High-risk rate (black):", round(high_risk_black, 4))
    print("High-risk rate (white):", round(high_risk_white, 4))
    print("High-risk diff:", round(high_risk_diff, 4))

    # Save summary for downstream use if needed
    pd.Series(summary).to_csv("analysis_summary.csv")


if __name__ == "__main__":
    main()
