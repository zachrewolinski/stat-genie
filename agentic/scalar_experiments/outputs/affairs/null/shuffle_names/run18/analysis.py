import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json, `age` encodes frequency of extramarital intercourse in the last year
    # and `religiousness` is a yes/no factor: "Are there children in the marriage?".
    # Define binary outcomes and predictors based on that metadata.
    df["has_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop any rows with missing key fields (should be none, but safe).
    df = df.dropna(subset=["has_affair", "has_children"])

    n_total = len(df)
    n_children = int(df["has_children"].sum())
    n_no_children = n_total - n_children

    # Group-level rates
    rates = df.groupby("has_children")["has_affair"].agg(["mean", "sum", "count"])
    rate_children = float(rates.loc[1, "mean"])
    rate_no_children = float(rates.loc[0, "mean"])

    # Difference in proportions test: does the probability of at least one affair differ?
    counts = np.array([rates.loc[1, "sum"], rates.loc[0, "sum"]])
    nobs = np.array([rates.loc[1, "count"], rates.loc[0, "count"]])
    z_stat, p_prop = proportions_ztest(count=counts, nobs=nobs, alternative="two-sided")

    # Unadjusted logistic regression: has_affair ~ has_children
    X = sm.add_constant(df[["has_children"]])
    y = df["has_affair"]
    logit_model = sm.Logit(y, X).fit(disp=False)

    coef_children = float(logit_model.params["has_children"])
    se_children = float(logit_model.bse["has_children"])
    p_logit = float(logit_model.pvalues["has_children"])

    # Odds ratio and 95% CI
    or_children = float(np.exp(coef_children))
    ci_low = float(np.exp(coef_children - 1.96 * se_children))
    ci_high = float(np.exp(coef_children + 1.96 * se_children))

    # Also fit an adjusted model with available covariates to check robustness.
    # We include numeric columns that are plausibly pre-treatment characteristics.
    covariates = []
    for col in ["education", "occupation", "children", "rating", "yearsmarried", "rownames"]:
        if col in df.columns:
            covariates.append(col)

    # Encode gender as a dummy variable if present.
    if "gender" in df.columns:
        gender_dummies = pd.get_dummies(df["gender"], drop_first=True, prefix="gender")
        df = pd.concat([df, gender_dummies], axis=1)
        covariates.extend(gender_dummies.columns.tolist())

    if covariates:
        X_adj = sm.add_constant(df[["has_children"] + covariates])
        try:
            logit_adj = sm.Logit(y, X_adj).fit(disp=False)
            coef_children_adj = float(logit_adj.params["has_children"])
            se_children_adj = float(logit_adj.bse["has_children"])
            p_logit_adj = float(logit_adj.pvalues["has_children"])
            or_children_adj = float(np.exp(coef_children_adj))
            ci_low_adj = float(np.exp(coef_children_adj - 1.96 * se_children_adj))
            ci_high_adj = float(np.exp(coef_children_adj + 1.96 * se_children_adj))
        except Exception:
            logit_adj = None
            p_logit_adj = float("nan")
            or_children_adj = float("nan")
            ci_low_adj = float("nan")
            ci_high_adj = float("nan")
    else:
        logit_adj = None
        p_logit_adj = float("nan")
        or_children_adj = float("nan")
        ci_low_adj = float("nan")
        ci_high_adj = float("nan")

    # Print a concise, machine-readable summary for the agent to interpret.
    print("N_total:", n_total)
    print("N_with_children:", n_children)
    print("N_without_children:", n_no_children)
    print("Rate_affair_with_children:", rate_children)
    print("Rate_affair_without_children:", rate_no_children)
    print("Prop_test_z:", z_stat)
    print("Prop_test_p:", p_prop)
    print("Logit_coef_children:", coef_children)
    print("Logit_p_children:", p_logit)
    print("Logit_OR_children:", or_children)
    print("Logit_OR_95CI_low:", ci_low)
    print("Logit_OR_95CI_high:", ci_high)
    print("Adj_Logit_p_children:", p_logit_adj)
    print("Adj_Logit_OR_children:", or_children_adj)
    print("Adj_Logit_OR_95CI_low:", ci_low_adj)
    print("Adj_Logit_OR_95CI_high:", ci_high_adj)


if __name__ == "__main__":
    main()

