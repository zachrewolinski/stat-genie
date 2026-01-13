def extract_final_answer(model_output):
    """
    Extract per-culture age effects from a fitted statsmodels Logit (BinaryResultsWrapper).

    Returns a dictionary with:
      - "object": pandas.DataFrame with one row per culture containing:
            culture: category label
            slope_logodds: estimated effect of a 1-year increase in age on log-odds of choosing the majority
            se: standard error of that slope
            z: z-statistic for slope != 0
            p: two-sided p-value
            OR: odds ratio for a 1-year increase in age (exp(slope))
            OR_CI_lower, OR_CI_upper: 95% CI for the odds ratio
            ref: boolean, True for the reference culture (the baseline in the dummy coding)
      - "description": short explanation of what the table shows and how to interpret it.

    Interpretation guide:
      - Positive slope_logodds => with increasing age children are more likely to choose the majority option in that culture.
      - Negative slope_logodds => with increasing age children are less likely to choose the majority option.
      - p gives whether the slope differs from zero (two-sided).
      - OR > 1 corresponds to increased odds per year; OR < 1 corresponds to decreased odds per year.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Extract coefficients and covariance matrix
    params = model_output.params
    cov = model_output.cov_params()

    # Get culture category ordering from the model's data (if available)
    try:
        # model_output.model.data.frame should exist for statsmodels results
        df_model = model_output.model.data.frame
        if 'culture' in df_model.columns and pd.api.types.is_categorical_dtype(df_model['culture']):
            categories = list(df_model['culture'].cat.categories)
        else:
            # fallback: infer categories from parameter names containing 'C(culture)'
            categories = []
            for name in params.index:
                if name.startswith('C(culture)[T.'):
                    # extract after 'C(culture)[T.' and before ']'
                    inside = name.split('C(culture)[T.')[1].split(']')[0]
                    categories.append(type(inside)(inside) if False else inside)
            # try to ensure unique ordering (may be str)
            categories = sorted(set(categories), key=lambda x: str(x))
    except Exception:
        # If anything goes wrong, try to infer categories from parameter names
        categories = []
        for name in params.index:
            if name.startswith('C(culture)[T.'):
                inside = name.split('C(culture)[T.')[1].split(']')[0]
                categories.append(inside)
        categories = sorted(set(categories), key=lambda x: str(x))

    # If categories found are strings of numbers, keep as strings (they must match param naming)
    # The reference category is the first category in the categorical ordering used by the model.
    if len(categories) == 0:
        # As a last resort, try numeric labels 1..8 (common in this dataset)
        categories = [str(i) for i in range(1, 9)]

    ref_category = categories[0]

    results = []
    # Base age coefficient name
    base_name = 'age_c'
    base_coef = params.get(base_name, 0.0)
    # For each culture, compute slope = base_coef + interaction (if present)
    for cat in categories:
        # interaction parameter name as statsmodels names it
        interaction_name = f'age_c:C(culture)[T.{cat}]'
        inter_coef = params.get(interaction_name, 0.0)

        slope = base_coef + inter_coef

        # Compute variance of linear combination:
        # var(slope) = var(age_c) + var(interaction) + 2*cov(age_c, interaction)
        var_age = cov.loc[base_name, base_name] if base_name in cov.index else 0.0
        if interaction_name in cov.index:
            var_inter = cov.loc[interaction_name, interaction_name]
            cov_ai = cov.loc[base_name, interaction_name] if base_name in cov.index else 0.0
        else:
            var_inter = 0.0
            cov_ai = 0.0

        var_slope = var_age + var_inter + 2.0 * cov_ai
        se_slope = np.sqrt(var_slope) if var_slope > 0 else 0.0

        # z and p-value
        z = slope / se_slope if se_slope > 0 else np.nan
        p = 2.0 * (1.0 - norm.cdf(abs(z))) if se_slope > 0 else np.nan

        # Odds ratio and 95% CI (on OR scale)
        OR = np.exp(slope)
        ci_low_logodds = slope - 1.96 * se_slope
        ci_high_logodds = slope + 1.96 * se_slope
        OR_CI_lower = np.exp(ci_low_logodds)
        OR_CI_upper = np.exp(ci_high_logodds)

        results.append({
            'culture': cat,
            'ref': (cat == ref_category),
            'slope_logodds': slope,
            'se': se_slope,
            'z': z,
            'p': p,
            'OR': OR,
            'OR_CI_lower': OR_CI_lower,
            'OR_CI_upper': OR_CI_upper
        })

    df_results = pd.DataFrame(results, columns=[
        'culture', 'ref', 'slope_logodds', 'se', 'z', 'p', 'OR', 'OR_CI_lower', 'OR_CI_upper'
    ])

    description = (
        "Per-culture estimates of the effect of age (age_c, in years) on the log-odds of choosing the majority option.\n"
        "- slope_logodds: estimated change in log-odds per 1-year increase in age for that culture (reference culture uses the main 'age_c' coefficient).\n"
        "- se, z, p: standard error, z-statistic, and two-sided p-value for slope != 0.\n"
        "- OR and OR_CI_{lower,upper}: odds ratio per year and its 95% CI (exp of log-odds slope and its CI).\n\n"
        "Interpretation: Positive slope_logodds (OR>1) indicates increasing reliance on the majority with age in that culture; "
        "negative slope_logodds (OR<1) indicates decreasing reliance with age. Use p to judge statistical evidence that the slope differs from zero."
    )

    return {"object": df_results, "description": description}