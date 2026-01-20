def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels GLMResultsWrapper (Poisson or NegBin)
    that examined how age, sex, and receiving help influence nut-cracking rate.

    Returns a dictionary with:
      - "object": a dict containing a coefficient table and focused combined effects
                  (with estimates on log-rate scale, SE, z, p, 95% CI, and rate ratios)
      - "description": brief interpretation of the extracted quantities and how to read them

    The function attempts to compute:
      - Coefficient table for all model parameters (coef, se, z, p, CI, exp(coef), exp(CI))
      - Combined effects:
          * Age effect when help = 0
          * Age effect when help = 1 (age main + age:help interaction)
          * Sex (male vs female) effect when help = 0
          * Sex effect when help = 1 (sex main + sex:help interaction)
          * Help effect at the sample mean age and mean sex (if model data frame available)
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Basic parameter table
    try:
        params = res.params.copy()
        bse = res.bse.copy()
        pvalues = res.pvalues.copy()
        ci = res.conf_int().copy()
    except Exception as e:
        raise ValueError(f"Could not extract params/bse/pvalues/ci from model_output: {e}")

    # Make a DataFrame for the coefficient table
    coef_df = pd.DataFrame({
        'coef': params,
        'se': bse,
        'z': params / bse,
        'pvalue': pvalues,
        'ci_lower': ci.iloc[:, 0],
        'ci_upper': ci.iloc[:, 1]
    })
    # Rate ratios (exponentiated coefficients) and CIs
    coef_df['rate_ratio'] = np.exp(coef_df['coef'])
    coef_df['rr_ci_lower'] = np.exp(coef_df['ci_lower'])
    coef_df['rr_ci_upper'] = np.exp(coef_df['ci_upper'])

    # Helper to find actual parameter names robustly (interaction naming can vary)
    param_names = list(params.index)

    def find_param(candidates):
        for c in candidates:
            if c in param_names:
                return c
        return None

    # Candidate names based on formula used in the model code
    age_name = find_param(['age'])
    sex_name = find_param(['sex_m'])
    help_name = find_param(['help_y'])
    help_age_name = find_param(['help_y:age', 'age:help_y'])
    help_sex_name = find_param(['help_y:sex_m', 'sex_m:help_y'])

    # Convenience to get coef value (0 if absent) and name mapping
    def coef_val(name):
        return float(params[name]) if (name is not None and name in params.index) else 0.0

    # Combined estimates (linear combinations) using covariance matrix for SEs
    try:
        cov = res.cov_params()
        # Ensure cov is a DataFrame with param names as index/cols
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, index=param_names, columns=param_names)
    except Exception:
        # Fallback: diagonal only if cov not available
        cov = pd.DataFrame(np.diag(bse.values**2), index=param_names, columns=param_names)

    def linear_combination(weights):
        """
        weights: dict mapping param name -> multiplier
        returns (estimate, se, z, p, ci_lower, ci_upper, rate_ratio, rr_ci_lower, rr_ci_upper)
        """
        # Build weight vector aligned to param_names
        w = np.array([weights.get(n, 0.0) for n in param_names], dtype=float)
        est = float(np.dot(w, params.values))
        # variance = w' * cov * w
        try:
            var = float(w @ cov.values @ w)
            se_lc = np.sqrt(var) if var >= 0 else np.nan
        except Exception:
            se_lc = np.nan
        z = est / se_lc if (se_lc and not np.isnan(se_lc)) else np.nan
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if (not np.isnan(z)) else np.nan
        ci_l = est - 1.96 * se_lc if not np.isnan(se_lc) else np.nan
        ci_u = est + 1.96 * se_lc if not np.isnan(se_lc) else np.nan
        rr = float(np.exp(est)) if not np.isnan(est) else np.nan
        rr_l = float(np.exp(ci_l)) if not np.isnan(ci_l) else np.nan
        rr_u = float(np.exp(ci_u)) if not np.isnan(ci_u) else np.nan
        return {
            'estimate': est, 'se': se_lc, 'z': z, 'p': p,
            'ci_lower': ci_l, 'ci_upper': ci_u,
            'rate_ratio': rr, 'rr_ci_lower': rr_l, 'rr_ci_upper': rr_u
        }

    combined = {}

    # Age effect when help = 0 (just the age main effect)
    if age_name is not None:
        combined['age_help0'] = linear_combination({age_name: 1.0})
    else:
        combined['age_help0'] = None

    # Age effect when help = 1 (age + interaction)
    weights = {}
    if age_name is not None:
        weights[age_name] = 1.0
    if help_age_name is not None:
        weights[help_age_name] = 1.0
    if weights:
        combined['age_help1'] = linear_combination(weights)
    else:
        combined['age_help1'] = None

    # Sex (male vs female) effect when help = 0
    if sex_name is not None:
        combined['sex_help0'] = linear_combination({sex_name: 1.0})
    else:
        combined['sex_help0'] = None

    # Sex effect when help = 1 (sex + sex:help interaction)
    weights = {}
    if sex_name is not None:
        weights[sex_name] = 1.0
    if help_sex_name is not None:
        weights[help_sex_name] = 1.0
    if weights:
        combined['sex_help1'] = linear_combination(weights)
    else:
        combined['sex_help1'] = None

    # Help effect at sample mean age and mean sex (if original data available)
    mean_age = None
    mean_sex = None
    df_available = None
    try:
        # statsmodels keeps the original data frame for formula-based models
        df_available = getattr(res.model.data, 'frame', None) or getattr(res.model.data, 'orig_exog', None)
        if isinstance(df_available, pd.DataFrame):
            if 'age' in df_available.columns:
                mean_age = float(df_available['age'].mean())
            if 'sex_m' in df_available.columns:
                mean_sex = float(df_available['sex_m'].mean())
    except Exception:
        df_available = None

    if (mean_age is not None) or (mean_sex is not None):
        ma = mean_age if mean_age is not None else 0.0
        ms = mean_sex if mean_sex is not None else 0.0
        weights = {}
        if help_name is not None:
            weights[help_name] = 1.0
        if help_age_name is not None:
            weights[help_age_name] = ma
        if help_sex_name is not None:
            weights[help_sex_name] = ms
        if weights:
            combined['help_at_mean_age_sex'] = {
                'mean_age': ma,
                'mean_sex': ms,
                **linear_combination(weights)
            }
        else:
            combined['help_at_mean_age_sex'] = None
    else:
        combined['help_at_mean_age_sex'] = None

    # Package final object
    result_object = {
        'model_choice': getattr(res, 'model_choice', getattr(res.model.family, 'name', None)),
        'overdispersion': getattr(res, 'overdispersion', None),
        'coef_table': coef_df.reset_index().rename(columns={'index': 'parameter'}).to_dict(orient='records'),
        'combined_effects': combined,
        'param_names': param_names
    }

    # Human-readable description
    description_lines = [
        "Extracted coefficients are on the log(rate) scale because the model is a count model",
        "with an offset (log_seconds). Exponentiating coefficients gives rate ratios (nuts/sec).",
        "",
        "coef_table: list of parameters with estimate, SE, z, p-value, 95% CI, and rate ratios.",
        "",
        "combined_effects includes:",
        "- age_help0: effect of age (log-rate change per year) when help = 0.",
        "- age_help1: effect of age when help = 1 (age main + age:help interaction).",
        "- sex_help0: effect of being male (vs female) when help = 0.",
        "- sex_help1: effect of being male when help = 1 (sex main + sex:help interaction).",
        "- help_at_mean_age_sex: effect of receiving help evaluated at the sample mean age and mean sex",
        "  (this uses help main + help:age * mean_age + help:sex * mean_sex) if the original data frame",
        "  was available in the fitted model object; it gives the log-rate difference and rate ratio.",
        "",
        "Interpretation guidance:",
        "- A coefficient (estimate) > 0 means an increased nut-opening rate (exp(coef) > 1).",
        "- A coefficient < 0 means a decreased rate (exp(coef) < 1).",
        "- p-values indicate whether effects differ from zero; p < 0.05 often taken as statistical evidence.",
        "",
        "Returned object contains both numeric tables and combined contrasts for direct interpretation."
    ]
    description = "\n".join(description_lines)

    return {"object": result_object, "description": description}