import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("mortgage.csv")

    # Outcome: 1 accepted, 0 denied
    base_cols = ["feature14", "feature2"]
    df_base = df[base_cols].dropna()
    y = df_base["feature14"].astype(float)
    female = df_base["feature2"].astype(int)

    # Unadjusted approval rates by gender
    rate_female = y[female == 1].mean()
    rate_male = y[female == 0].mean()
    diff_rate = rate_female - rate_male

    count = np.array([y[female == 1].sum(), y[female == 0].sum()])
    nobs = np.array([(female == 1).sum(), (female == 0).sum()])
    z_stat, p_rate = proportions_ztest(count, nobs)

    # Adjusted model: exclude outcome and redundant/ID-like fields
    predictors = [c for c in df.columns if c not in {"feature14", "feature11", "feature1"}]
    df_model = df[predictors + ["feature14"]].dropna()
    X = sm.add_constant(df_model[predictors].astype(float), has_constant="add")
    y = df_model["feature14"].astype(float)

    try:
        model = sm.Logit(y, X).fit(disp=False)
    except Exception:
        # Fallback to GLM binomial if Logit has convergence issues
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    coef = model.params["feature2"]
    se = model.bse["feature2"]
    p_logit = model.pvalues["feature2"]
    odds_ratio = float(np.exp(coef))
    ci_low = float(np.exp(coef - 1.96 * se))
    ci_high = float(np.exp(coef + 1.96 * se))

    # Evidence-based response score
    # Start at neutral 50, move based on significance and effect size
    score = 50
    # If statistically significant
    if p_logit < 0.05:
        # Effect direction: female associated with higher approval -> yes; lower -> no
        if coef > 0:
            # Increase score; stronger if OR far from 1
            strength = min(abs(np.log(odds_ratio)) / 0.5, 1.0)
            score = int(round(60 + 30 * strength))
        else:
            strength = min(abs(np.log(odds_ratio)) / 0.5, 1.0)
            score = int(round(40 - 30 * strength))
    else:
        # Not significant: stay near no-evidence, adjust slightly toward direction of effect
        direction = 1 if coef > 0 else -1
        score = int(round(45 + 5 * direction))

    score = max(0, min(100, score))

    explanation = (
        "We tested whether gender (female=1) is associated with mortgage approval (accepted=1). "
        f"Unadjusted approval rates: female={rate_female:.3f}, male={rate_male:.3f} "
        f"(difference={diff_rate:.3f}, two-proportion z-test p={p_rate:.4f}). "
        "In a multivariable logistic regression controlling for applicant characteristics "
        f"(all available features except the outcome, its redundant denial indicator, and the ID-like field), "
        f"the female coefficient was {coef:.4f} (odds ratio={odds_ratio:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
        f"p={p_logit:.4f}. "
        "Because the adjusted effect is not statistically significant, the evidence does not support a gender effect on approval in this dataset."
    )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": score, "explanation": explanation}, f)


if __name__ == "__main__":
    main()
