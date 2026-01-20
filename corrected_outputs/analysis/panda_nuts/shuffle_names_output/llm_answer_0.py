def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels OLS RegressionResultsWrapper
    (with clustered SEs) for the model:
      Efficiency ~ age_years + C(Sex) * received_help + C(hammer_type)
    
    Returns a dictionary with:
      - "object": dictionary of extracted numeric results (coefficients, SEs,
                  p-values, 95% CIs) for age, main sex effect, received_help,
                  interaction, and the marginal effect of received_help for
                  Females and Males.
      - "description": short interpretation of what these numbers mean.
    
    Notes:
      - For the marginal (combined) effects (e.g., effect of received_help for
        Males = main received_help + interaction), the standard error is
        computed using the covariance matrix (delta method). P-values for these
        combined effects use a normal approximation.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    params = res.params  # pandas Series indexed by param names
    cov = res.cov_params()  # covariance matrix (DataFrame or ndarray)
    pvals = res.pvalues
    try:
        conf = res.conf_int(alpha=0.05)
    except Exception:
        # fallback: compute from params and cov
        se_all = np.sqrt(np.diag(cov))
        z = params / se_all
        z = np.asarray(z)
        crit = stats.norm.ppf(0.975)
        conf = np.column_stack((params - crit * se_all, params + crit * se_all))
        # Make conf a DataFrame-like with same index
        import pandas as _pd
        conf = _pd.DataFrame(conf, index=params.index, columns=[0, 1])

    # Helper to safely get parameter values (returns 0 if param not present).
    def get_param(name):
        return params[name] if name in params.index else 0.0

    def has_param(name):
        return name in params.index

    # Common parameter name possibilities
    # Main terms
    name_age = 'age_years'
    name_help = 'received_help'
    # Sex effect (treatment coding likely created a term like 'C(Sex)[T.Male]')
    sex_term = None
    for nm in params.index:
        if nm.startswith('C(Sex)'):
            # pick the main sex contrast term (not interaction)
            if ':' not in nm and 'received_help' not in nm:
                sex_term = nm
                break

    # Interaction term name: could appear as 'C(Sex)[T.Male]:received_help' or 'received_help:C(Sex)[T.Male]'
    inter_name = None
    for nm in params.index:
        if 'received_help' in nm and 'C(Sex)' in nm:
            inter_name = nm
            break

    # Extract basic stats for a parameter (if present)
    def param_stats(nm):
        if not has_param(nm):
            return None
        coef = float(params[nm])
        se = float(np.sqrt(cov.loc[nm, nm])) if (isinstance(cov, (np.ndarray,)) is False) else float(np.sqrt(cov[params.index.get_loc(nm), params.index.get_loc(nm)]))
        p = float(pvals[nm]) if nm in pvals.index else float(2 * (1 - stats.norm.cdf(abs(coef / se))))
        ci_low = float(conf.loc[nm, 0]) if nm in conf.index else None
        ci_high = float(conf.loc[nm, 1]) if nm in conf.index else None
        return {'coef': coef, 'se': se, 'p_value': p, 'ci_95_low': ci_low, 'ci_95_high': ci_high}

    results = {}

    # Age effect
    if has_param(name_age):
        results['age_years'] = param_stats(name_age)
    else:
        results['age_years'] = {'note': f'Parameter "{name_age}" not found in model.'}

    # Sex main effect (Male vs reference (likely Female))
    if sex_term:
        results['sex_main'] = {'term_name': sex_term, **param_stats(sex_term)}
    else:
        results['sex_main'] = {'note': 'No C(Sex) main-effect parameter found in model.'}

    # Received_help main effect (this is the effect for the reference sex, typically Female)
    if has_param(name_help):
        results['received_help_main'] = param_stats(name_help)
    else:
        results['received_help_main'] = {'note': f'Parameter "{name_help}" not found in model.'}

    # Interaction
    if inter_name:
        results['interaction'] = {'term_name': inter_name, **param_stats(inter_name)}
    else:
        results['interaction'] = {'note': 'No interaction parameter between Sex and received_help found in model.'}

    # Compute marginal effect of received_help for each sex:
    # For reference sex (likely Female): effect = coef(received_help)
    # For other sex (likely Male): effect = coef(received_help) + coef(interaction)
    # Compute SE via delta method using covariance matrix.
    def combined_effect(base_name, add_name, label):
        # base_name: main received_help param name
        # add_name: interaction param name (can be None)
        if not has_param(base_name):
            return {'note': f'Base parameter "{base_name}" not found.'}
        b_base = float(params[base_name])
        if add_name and has_param(add_name):
            b_add = float(params[add_name])
            coef = b_base + b_add
            # variance = var(base) + var(add) + 2*cov(base, add)
            var_base = float(cov.loc[base_name, base_name])
            var_add = float(cov.loc[add_name, add_name])
            covar = float(cov.loc[base_name, add_name])
            se = float(np.sqrt(var_base + var_add + 2 * covar))
            # p-value via normal approx
            z = coef / se if se > 0 else np.nan
            p = float(2 * (1 - stats.norm.cdf(abs(z)))) if se > 0 else np.nan
            # 95% CI
            crit = stats.norm.ppf(0.975)
            ci_low = coef - crit * se
            ci_high = coef + crit * se
            return {'coef': coef, 'se': se, 'p_value': p, 'ci_95_low': ci_low, 'ci_95_high': ci_high,
                    'components': {base_name: float(b_base), add_name: float(b_add)}}
        else:
            # no interaction term; marginal effect equals base
            return param_stats(base_name)

    # Determine reference sex name (informational)
    reference_sex = 'reference level (likely the first alphabetic category, e.g., "Female")'
    if sex_term:
        # infer reference from naming: sex_term like 'C(Sex)[T.Male]' means Male is compared to reference
        import re
        m = re.match(r'C\(Sex\)\[T\.(.+)\]', sex_term)
        if m:
            other = m.group(1)
            # so reference is not 'other' -- unknown exact name, but we can state it
            reference_sex = f'reference sex (the category not shown in "{sex_term}")'

    results['received_help_effect_for_reference_sex'] = combined_effect(name_help, None, 'reference')  # direct
    # For non-reference sex (the one appearing in sex_term, if present), use interaction
    if inter_name:
        results['received_help_effect_for_other_sex'] = combined_effect(name_help, inter_name, 'other_sex')
    else:
        results['received_help_effect_for_other_sex'] = {'note': 'No interaction; effect is same for both sexes (see received_help_main).'}

    # Additional raw parameters for user's convenience
    results['raw_params'] = params.to_dict()

    # Prepare a concise description
    description_lines = [
        "Returned values are cluster-robust coefficient estimates (coef), their standard errors (se),",
        "two-sided p-values (p_value), and 95% confidence intervals (ci_95_low/ci_95_high).",
        "Key entries:",
        "- age_years: estimated change in nuts/min per additional year of age.",
        "- sex_main: coefficient for the non-reference sex vs reference (term name given).",
        "- received_help_main: effect of receiving help for the reference sex (typically Female).",
        "- interaction: additional effect of receiving help for the other sex (so male effect = received_help_main + interaction).",
        "- received_help_effect_for_reference_sex and _for_other_sex: marginal effects of receiving help for each sex computed via the covariance matrix (delta method).",
        "P-values for the combined (marginal) effects use a normal approximation based on the delta-method standard error."
    ]
    description = " ".join(description_lines)

    return {"object": results, "description": description}