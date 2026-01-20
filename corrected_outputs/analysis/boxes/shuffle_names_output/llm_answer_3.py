def extract_final_answer(model_output):
    """
    Extracts age-related effects (slopes), their significance, and odds-ratios
    from a fitted statsmodels GLMResultsWrapper produced by the provided model().
    Returns a dict with:
      - "object": dict mapping each culture level to its age-slope info:
          { 'estimate_logodds', 'se', 'z', 'p', '95%CI_logodds', 'odds_ratio', '95%CI_or' }
      - "description": short verbal interpretation of whether reliance on the majority
                       increases/decreases with age in each cultural context
                       (significance at alpha=0.05).
    """
    import re
    import numpy as np
    from math import sqrt
    try:
        from scipy import stats
    except Exception:
        # fallback: use normal cdf from statsmodels if scipy not available
        from statsmodels.stats import ztools as _ztools
        stats = None

    res = model_output  # GLMResultsWrapper

    # Extract parameters and covariance matrix
    params = res.params  # pandas Series
    cov = res.cov_params()  # DataFrame

    # Check Age_c present
    if 'Age_c' not in params.index:
        raise ValueError("Model does not contain 'Age_c' parameter in params. Found params: "
                         f"{list(params.index)}")

    age_coef_name = 'Age_c'

    # Try to get the culture levels from the original dataframe if available
    data_levels = None
    try:
        df = res.model.data.frame
        if 'CultureSite' in df.columns:
            # preserve order as observed in data
            data_levels = list(pd.unique(df['CultureSite']))
    except Exception:
        df = None

    # Identify interaction parameter names and extract their culture level labels
    interaction_pattern = re.compile(r'Age_c.*C\(CultureSite\)\[T\.?([^\]\:]+)\]?|C\(CultureSite\)\[T\.?([^\]\:]+)\]?.*Age_c')
    interaction_params = {}
    for param_name in params.index:
        if 'Age_c' in param_name and 'C(CultureSite)' in param_name:
            m = interaction_pattern.search(param_name)
            if m:
                level = m.group(1) if m.group(1) else m.group(2)
            else:
                # fallback: try to pull text between 'T.' and ']'
                if 'T.' in param_name and ']' in param_name:
                    start = param_name.find('T.') + 2
                    end = param_name.find(']', start)
                    level = param_name[start:end]
                else:
                    # as last resort, use full param name as key
                    level = param_name
            interaction_params[level] = param_name

    # If we have data_levels from the original dataframe, use it; otherwise build levels
    if data_levels is None:
        # Try to reconstruct levels: all levels that appear in interaction names plus one reference
        # We can infer the reference by looking for C(CultureSite)[T.<level>] params:
        inferred_levels = list(interaction_params.keys())
        # we don't know reference name; name it "reference" if unknown
        # But try to read categories from model.exog_names if possible to get all levels:
        try:
            # sometimes model.data.orig_exog contains category dummies; we can't reliably get labels
            all_levels = inferred_levels.copy()
        except Exception:
            all_levels = inferred_levels.copy()
        # If no level available, raise
        if len(all_levels) == 0:
            raise ValueError("Could not infer any CultureSite levels from model. "
                             "Ensure CultureSite was present and coded as a factor in the model.")
        data_levels = all_levels.copy()
        # Add a placeholder for the omitted (reference) level if necessary
        # We will mark that one level is the reference when we detect it's not in interaction list.
        # If interactions present for some levels, then the reference is any level not listed;
        # but since we only have listed ones, create a generic 'reference' if needed.
        # To be safe, append a 'reference' label if interactions cover all observed levels.
        data_levels.append('REFERENCE') if 'REFERENCE' not in data_levels else None

    import pandas as pd  # used for organizing results

    results_by_culture = {}
    # For establishing list of cultures to report: if we have actual df levels, use them.
    # If the model's df was available and had all unique CultureSite values, use that order.
    cultures = data_levels

    # Ensure that if a real dataframe existed, and it included levels we didn't pick up from interactions,
    # those are included here. Otherwise we will proceed with cultures as above.
    if df is not None:
        cultures = list(pd.unique(df['CultureSite']))

    # Helper to safely get covariance entries (return 0 if missing)
    def cov_get(a, b):
        try:
            return cov.loc[a, b]
        except Exception:
            try:
                return cov.loc[b, a]
            except Exception:
                return 0.0

    # For p-value calculation using normal distribution
    def two_sided_p(z):
        if stats is not None:
            return 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        else:
            # use statsmodels' normal cdf
            from statsmodels.distributions.empirical_distribution import ECDF
            # fallback: approximate using math.erfc
            import math
            return 2.0 * (0.5 * math.erfc(-abs(z) / math.sqrt(2.0)))

    # Compute Age slope for each culture
    for cult in cultures:
        # Determine which interaction param corresponds to this culture (if any)
        interaction_param = None
        # Try to find exact match from interaction_params keys (they are culture-level strings)
        if cult in interaction_params:
            interaction_param = interaction_params[cult]
        else:
            # sometimes the culture labels in the original df might be of different types (e.g., ints)
            # Try string form matching
            for key, pname in interaction_params.items():
                if str(key) == str(cult):
                    interaction_param = pname
                    break
        # Build estimate and variance
        base_est = params.get(age_coef_name, 0.0)
        inter_est = params.get(interaction_param, 0.0) if interaction_param is not None else 0.0
        est = base_est + inter_est

        # Variance calculation
        var_age = cov_get(age_coef_name, age_coef_name)
        var_inter = cov_get(interaction_param, interaction_param) if interaction_param else 0.0
        cov_ai = cov_get(age_coef_name, interaction_param) if interaction_param else 0.0
        var_sum = var_age + var_inter + 2.0 * cov_ai
        se = sqrt(var_sum) if var_sum >= 0 else float('nan')

        z = est / se if se and not np.isnan(se) else float('nan')
        p = two_sided_p(z) if not np.isnan(z) else float('nan')
        ci_low = est - 1.96 * se if not np.isnan(se) else float('nan')
        ci_high = est + 1.96 * se if not np.isnan(se) else float('nan')

        # convert to odds ratio scale
        or_est = np.exp(est) if not np.isnan(est) else float('nan')
        or_ci = (np.exp(ci_low), np.exp(ci_high)) if not np.isnan(ci_low) and not np.isnan(ci_high) else (float('nan'), float('nan'))

        results_by_culture[str(cult)] = {
            'estimate_logodds': est,
            'se': se,
            'z': z,
            'p': p,
            '95%CI_logodds': (ci_low, ci_high),
            'odds_ratio': or_est,
            '95%CI_or': or_ci,
            'interaction_param_name': interaction_param if interaction_param is not None else None
        }

    # Build a human-readable description summarizing significance and direction
    desc_lines = []
    alpha = 0.05
    for cult, info in results_by_culture.items():
        p = info['p']
        est = info['estimate_logodds']
        direction = 'increase' if est > 0 else ('decrease' if est < 0 else 'no change')
        sig = 'significant' if (not np.isnan(p) and p < alpha) else 'not significant'
        or_est = info['odds_ratio']
        desc_lines.append(
            f"For culture '{cult}': age slope (log-odds) = {est:.3f}, SE = {info['se']:.3f}, "
            f"p = {p:.3f} ({sig}). This indicates a {direction} in probability of choosing the majority with age."
            f" Corresponding OR = {or_est:.3f}, 95% CI = ({info['95%CI_or'][0]:.3f}, {info['95%CI_or'][1]:.3f})."
        )

    description = ("Age-by-culture results (each line reports whether reliance on the majority "
                   "changes with age in that cultural context):\n" + "\n".join(desc_lines))

    return {
        "object": results_by_culture,
        "description": description
    }