def extract_final_answer(model_output):
    """
    Extracts age-related effects (slopes) on choosing the majority option for each cultural site
    from a fitted logistic model with Age_c x Culture interactions.

    Returns:
      dict with keys:
        - "object": pandas.DataFrame indexed by culture label (including the reference culture)
                    with columns:
                      slope_logit  : slope of Age_c (change in log-odds of choosing majority per year)
                      se            : standard error of the slope
                      z             : z-statistic
                      p             : two-sided p-value
                      ci_lower      : lower bound of 95% CI (logit scale)
                      ci_upper      : upper bound of 95% CI (logit scale)
                      OR            : odds ratio per year = exp(slope)
                      OR_ci_lower   : lower bound of 95% CI for OR
                      OR_ci_upper   : upper bound of 95% CI for OR
        - "description": textual explanation of what the numbers mean.

    Notes:
      - Assumes the model params include 'Age_c' and interaction terms named like
        'Age_c:C(culture)[T.x]'. The reference culture (no C(culture)[T.*] term) is reported
        as 'culture=reference'.
      - Uses the provided covariance matrix (model_output.cov_params) to compute SEs
        for linear combinations.
    """
    import re
    import numpy as np
    import pandas as pd
    from math import exp
    try:
        from scipy import stats
        _has_scipy = True
    except Exception:
        _has_scipy = False

    params = model_output.params
    cov = model_output.cov_params

    # Ensure index are strings
    param_names = [str(n) for n in params.index]

    # Base age slope
    if 'Age_c' not in params.index:
        raise ValueError("Model output does not contain 'Age_c' parameter.")
    beta_age = float(params['Age_c'])

    # Find interaction terms of the form Age_c:C(culture)[T.x]
    pattern = re.compile(r'^Age_c:C\(culture\)\[T\.(.+)\]$')
    interactions = {}
    for name in params.index:
        m = pattern.match(str(name))
        if m:
            culture_label = m.group(1)
            interactions[culture_label] = float(params[name])

    # Infer reference culture label as 'reference' (since no param exists for it)
    # Build list of cultures = reference plus those found in interactions
    culture_labels = ['reference'] + sorted(interactions.keys(), key=lambda x: (int(x) if x.isdigit() else x))

    rows = []
    for lab in culture_labels:
        if lab == 'reference':
            slope = beta_age
            # variance = Var(Age_c)
            var = float(cov.loc['Age_c', 'Age_c'])
        else:
            int_name = f'Age_c:C(culture)[T.{lab}]'
            if int_name not in params.index:
                # Should not happen, continue as NA
                slope = np.nan
                var = np.nan
            else:
                beta_int = float(params[int_name])
                slope = beta_age + beta_int
                # var(β_age + β_int) = var(β_age) + var(β_int) + 2 cov(β_age, β_int)
                var_age = float(cov.loc['Age_c', 'Age_c'])
                var_int = float(cov.loc[int_name, int_name])
                cov_age_int = float(cov.loc['Age_c', int_name])
                var = var_age + var_int + 2.0 * cov_age_int

        se = np.sqrt(var) if (not np.isnan(var)) else np.nan
        z = slope / se if (se and not np.isnan(se)) else np.nan

        if _has_scipy and (not np.isnan(z)):
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        elif not np.isnan(z):
            # fallback using math.erfc: p = erfc(|z|/sqrt(2))
            import math
            p = math.erfc(abs(z) / math.sqrt(2.0))
        else:
            p = np.nan

        ci_low = slope - 1.96 * se if (not np.isnan(se)) else np.nan
        ci_high = slope + 1.96 * se if (not np.isnan(se)) else np.nan
        OR = float(np.exp(slope)) if (not np.isnan(slope)) else np.nan
        OR_ci_low = float(np.exp(ci_low)) if (not np.isnan(ci_low)) else np.nan
        OR_ci_high = float(np.exp(ci_high)) if (not np.isnan(ci_high)) else np.nan

        rows.append({
            'culture': ('culture=' + lab) if lab != 'reference' else 'culture=reference',
            'slope_logit': slope,
            'se': se,
            'z': z,
            'p': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'OR': OR,
            'OR_ci_lower': OR_ci_low,
            'OR_ci_upper': OR_ci_high
        })

    df = pd.DataFrame(rows).set_index('culture')

    # Build a concise description interpreting the numbers
    desc_lines = []
    desc_lines.append(
        "For each cultural site, the reported 'slope_logit' is the estimated change in log-odds of a child "
        "choosing the majority-demonstrated option per additional year of age. "
        "OR is exp(slope) (multiplicative change in odds per year)."
    )
    desc_lines.append(
        "Results are based on the model coefficients and the model covariance matrix to compute SEs "
        "for the linear combinations (Age_c + Age_c:C(culture)[T.x])."
    )
    desc_lines.append(
        "Interpretation guidance: a positive slope_logit (and OR>1) means reliance on the majority increases with age "
        "in that cultural site; negative slope_logit (OR<1) means reliance decreases with age. "
        "The p-value tests whether the slope differs from zero."
    )
    description = " ".join(desc_lines)

    return {"object": df, "description": description}