import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = "mortgage.csv"


def fmt_pct(x):
    return f"{x*100:.2f}%"


def fit_adjusted_model(accepted, X):
    """
    Try Logit first; if it fails, fall back to GLM Binomial.
    Returns (coef, p_value, odds_ratio, method_name) or (None, None, None, None) on failure.
    """
    try:
        model = sm.Logit(accepted, X)
        res = model.fit(disp=False, method="lbfgs", maxiter=200)
        coef = float(res.params["feature2"])
        p_val = float(res.pvalues["feature2"])
        or_adj = float(math.exp(coef))
        return coef, p_val, or_adj, "logit"
    except Exception:
        pass

    try:
        model = sm.GLM(accepted, X, family=sm.families.Binomial())
        res = model.fit(maxiter=200, disp=False)
        coef = float(res.params["feature2"])
        p_val = float(res.pvalues["feature2"])
        or_adj = float(math.exp(coef))
        return coef, p_val, or_adj, "glm_binomial"
    except Exception:
        return None, None, None, None


def main():
    df = pd.read_csv(DATA_PATH)

    gender = df["feature2"]  # 1=female, 0=male
    accepted = df["feature14"]  # 1=accepted, 0=denied

    # Basic counts
    n_f = int((gender == 1).sum())
    n_m = int((gender == 0).sum())
    acc_f = int(accepted[gender == 1].sum())
    acc_m = int(accepted[gender == 0].sum())

    rate_f = acc_f / n_f if n_f else float("nan")
    rate_m = acc_m / n_m if n_m else float("nan")
    diff = rate_f - rate_m

    # 95% CI for difference in proportions (normal approx)
    se = math.sqrt(rate_f * (1 - rate_f) / n_f + rate_m * (1 - rate_m) / n_m)
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    # Two-proportion z-test
    zstat, p_z = proportions_ztest([acc_f, acc_m], [n_f, n_m])

    # Chi-square test of independence
    ctab = pd.crosstab(gender, accepted)
    chi2, p_chi, dof, _ = chi2_contingency(ctab)

    # Logistic regression with controls (exclude feature1 id-like, feature11 denied, feature14 accepted)
    control_cols = [
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
    model_cols = ["feature2"] + control_cols
    reg_df = df[model_cols + ["feature14"]].replace([np.inf, -np.inf], np.nan).dropna()

    p_adj = None
    coef = None
    or_adj = None
    model_name = None
    n_reg = len(reg_df)

    if n_reg > 0:
        X = reg_df[model_cols].copy()
        X = sm.add_constant(X, has_constant="add")
        y = reg_df["feature14"]
        coef, p_adj, or_adj, model_name = fit_adjusted_model(y, X)

    # Scoring logic
    p_used = p_adj if p_adj is not None else p_z

    if p_used < 0.05:
        response = 60
        if abs(diff) >= 0.05:
            response = 80
        elif abs(diff) >= 0.03:
            response = 72
        elif abs(diff) >= 0.02:
            response = 66
        else:
            response = 58
    else:
        response = 40
        if p_used >= 0.2:
            response = 30
        if p_used >= 0.5:
            response = 20
        if p_used >= 0.8:
            response = 10
        if abs(diff) < 0.01:
            response = max(0, response - 5)
        elif abs(diff) >= 0.05:
            response = min(49, response + 5)

    # Ensure bounds and integer
    response = int(max(0, min(100, round(response))))

    # Build explanation
    explanation_parts = []
    explanation_parts.append(
        f"Sample size n={len(df)}. Gender is feature2 (1=female, 0=male) and approval is feature14 (1=accepted)."
    )
    explanation_parts.append(
        f"Approval rates: female {fmt_pct(rate_f)} (n={n_f}), male {fmt_pct(rate_m)} (n={n_m}); difference {diff*100:.2f} pp with 95% CI [{ci_low*100:.2f}, {ci_high*100:.2f}] pp."
    )
    explanation_parts.append(
        f"Unadjusted tests: two-proportion z-test p={p_z:.4g}; chi-square p={p_chi:.4g}."
    )

    if p_adj is not None:
        explanation_parts.append(
            "Adjusted regression (" + model_name + ") controlling for credit and application factors (features 3-10, 12-13) on "
            f"n={n_reg} complete cases finds gender odds ratio {or_adj:.3f} with p={p_adj:.4g}."
        )
        if p_adj < 0.05:
            explanation_parts.append(
                "The adjusted gender effect is statistically significant, indicating evidence that gender is associated with approval likelihood."
            )
        else:
            explanation_parts.append(
                "The adjusted gender effect is not statistically significant, so there is insufficient evidence that gender affects approval after accounting for observed factors."
            )
    else:
        explanation_parts.append(
            "Adjusted regression could not be estimated on complete cases; conclusion relies on unadjusted tests."
        )
        if p_z < 0.05:
            explanation_parts.append(
                "Unadjusted tests are significant, suggesting a gender association, but without adjustment this could reflect confounding."
            )
        else:
            explanation_parts.append(
                "Unadjusted tests are not significant, suggesting no clear gender association."
            )

    explanation = " ".join(explanation_parts)

    result = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
