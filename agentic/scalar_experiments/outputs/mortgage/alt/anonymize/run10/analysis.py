import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import Table2x2

DATA_PATH = "mortgage.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Variables
    gender = df["feature2"]  # 1 female, 0 male
    accepted = df["feature14"]  # 1 accepted, 0 denied

    # Basic counts
    n_female = int((gender == 1).sum())
    n_male = int((gender == 0).sum())
    acc_female = int(((gender == 1) & (accepted == 1)).sum())
    acc_male = int(((gender == 0) & (accepted == 1)).sum())

    rate_female = acc_female / n_female if n_female else np.nan
    rate_male = acc_male / n_male if n_male else np.nan
    rate_diff = rate_female - rate_male

    # Two-proportion z-test (female vs male)
    count = np.array([acc_female, acc_male])
    nobs = np.array([n_female, n_male])
    z_stat, p_z = proportions_ztest(count, nobs)

    # 2x2 table for odds ratio
    # rows: female/male, cols: accepted/denied
    denied_female = int(((gender == 1) & (accepted == 0)).sum())
    denied_male = int(((gender == 0) & (accepted == 0)).sum())
    table = np.array([[acc_female, denied_female], [acc_male, denied_male]])
    t2x2 = Table2x2(table)
    or_unadj = float(t2x2.oddsratio)
    or_ci = t2x2.oddsratio_confint()

    # Adjusted logistic regression
    # Exclude feature11 (denied) to avoid perfect collinearity with acceptance
    covariate_cols = [
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
        "feature12",
        "feature13",
    ]
    model_df = df[covariate_cols + ["feature14"]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
    X = model_df[covariate_cols]
    y = model_df["feature14"]

    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    coef = result.params["feature2"]
    p_val = result.pvalues["feature2"]
    or_adj = float(np.exp(coef))
    ci_low, ci_high = result.conf_int().loc["feature2"].tolist()
    or_ci_adj = (float(np.exp(ci_low)), float(np.exp(ci_high)))

    output = {
        "n_total": int(df.shape[0]),
        "n_female": n_female,
        "n_male": n_male,
        "accept_rate_female": rate_female,
        "accept_rate_male": rate_male,
        "accept_rate_diff_female_minus_male": rate_diff,
        "z_test_p_value": float(p_z),
        "z_test_z": float(z_stat),
        "odds_ratio_unadjusted": or_unadj,
        "odds_ratio_unadjusted_ci": [float(or_ci[0]), float(or_ci[1])],
        "logit_gender_coef": float(coef),
        "logit_gender_p_value": float(p_val),
        "odds_ratio_adjusted": or_adj,
        "odds_ratio_adjusted_ci": [or_ci_adj[0], or_ci_adj[1]],
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
