import json
import pandas as pd
import numpy as np
import statsmodels.api as sm


def load_data(path="mortgage.csv"):
    df = pd.read_csv(path)
    return df


def prep_data(df):
    # Ensure binary columns are numeric 0/1
    binary_cols = ["female", "black", "self_employed", "married", "bad_history", "deny", "denied_PMI", "accept"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def logistic_regression(df, outcome, predictors):
    data = df[[outcome] + predictors].dropna()
    y = data[outcome]
    X = data[predictors]
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def main():
    df = prep_data(load_data())

    # Unadjusted: deny ~ female
    unadj = logistic_regression(df, "deny", ["female"])

    # Adjusted: deny ~ female + controls
    controls = [
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
        "denied_PMI",
    ]
    adj = logistic_regression(df, "deny", ["female"] + controls)

    # Also consider accept outcome for sanity (accept = 1 - deny)
    # But accept is redundant; we use deny only and interpret opposite.

    def odds_ratio(res, var):
        coef = res.params[var]
        se = res.bse[var]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)
        return float(or_val), float(ci_low), float(ci_high)

    unadj_or, unadj_ci_low, unadj_ci_high = odds_ratio(unadj, "female")
    adj_or, adj_ci_low, adj_ci_high = odds_ratio(adj, "female")

    results = {
        "n": int(df.shape[0]),
        "deny_rate": float(df["deny"].mean()),
        "unadjusted": {
            "coef": float(unadj.params["female"]),
            "p_value": float(unadj.pvalues["female"]),
            "odds_ratio": unadj_or,
            "ci_low": unadj_ci_low,
            "ci_high": unadj_ci_high,
        },
        "adjusted": {
            "coef": float(adj.params["female"]),
            "p_value": float(adj.pvalues["female"]),
            "odds_ratio": adj_or,
            "ci_low": adj_ci_low,
            "ci_high": adj_ci_high,
        },
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
