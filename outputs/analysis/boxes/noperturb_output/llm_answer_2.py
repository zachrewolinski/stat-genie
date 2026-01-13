def extract_final_answer(model_output):
    """
    Extracts age-related effects (linear and quadratic) and culture-specific age slopes
    from a fitted statsmodels GLMResults-like object (possibly with cluster-robust cov).
    
    Returns a dict with keys:
      - "object": dict of extracted numeric results (overall age linear & quadratic;
                  per-culture age slopes with SE, z, p, 95% CI, and OR + CI;
                  joint test p-value for age-by-culture interactions if available)
      - "description": human-readable explanation tying the numbers to the research question.
    """
    import re
    import numpy as np
    from collections import OrderedDict
    try:
        from scipy import stats
    except Exception:
        # fall back to normal approx using math.erfc if scipy not available
        import math
        class _Approx:
            @staticmethod
            def pvalue_from_z(z):
                return 2 * 0.5 * math.erfc(abs(z) / math.sqrt(2))
        stats = None

    res = model_output  # alias
    
    # Extract parameter vector, covariance matrix, and names
    params = getattr(res, 'params', None)
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        # some result wrappers expose normalized_cov_params or similar
        cov = getattr(res, 'normalized_cov_params', None)
    if params is None:
        raise ValueError("model_output does not have .params")
    param_names = list(params.index if hasattr(params, 'index') else params.keys())
    # Convert params to plain dict of floats for safety
    params_dict = {name: float(params[name]) for name in param_names}

    # Helper to get cov element (returns 0 if not present)
    def cov_elem(a, b):
        if cov is None:
            return 0.0
        # cov may be a DataFrame or ndarray
        try:
            # If DataFrame-like
            return float(cov.loc[a, b])
        except Exception:
            try:
                # If ndarray, need index positions
                ai = param_names.index(a)
                bi = param_names.index(b)
                return float(cov[ai, bi])
            except Exception:
                return 0.0

    results = OrderedDict()

    # Overall linear age effect (age_c)
    if 'age_c' in params_dict:
        beta_age = params_dict['age_c']
        # SE from cov
        se_age = np.sqrt(max(0.0, cov_elem('age_c', 'age_c')))
        z_age = beta_age / se_age if se_age > 0 else np.nan
        if stats is not None:
            p_age = 2 * (1 - stats.norm.cdf(abs(z_age)))
        else:
            p_age = _Approx.pvalue_from_z(z_age)
        ci_low_age = beta_age - 1.96 * se_age
        ci_high_age = beta_age + 1.96 * se_age
        results['age_c'] = {
            'beta_logodds': float(beta_age),
            'se': float(se_age),
            'z': float(z_age),
            'p': float(p_age),
            '95%CI_logodds': [float(ci_low_age), float(ci_high_age)],
            'OR_per_unit_age': float(np.exp(beta_age)),
            '95%CI_OR': [float(np.exp(ci_low_age)), float(np.exp(ci_high_age))]
        }
    else:
        results['age_c'] = None

    # Quadratic term
    if 'age_c2' in params_dict:
        beta_q = params_dict['age_c2']
        se_q = np.sqrt(max(0.0, cov_elem('age_c2', 'age_c2')))
        z_q = beta_q / se_q if se_q > 0 else np.nan
        if stats is not None:
            p_q = 2 * (1 - stats.norm.cdf(abs(z_q)))
        else:
            p_q = _Approx.pvalue_from_z(z_q)
        ci_low_q = beta_q - 1.96 * se_q
        ci_high_q = beta_q + 1.96 * se_q
        results['age_c2'] = {
            'beta_logodds': float(beta_q),
            'se': float(se_q),
            'z': float(z_q),
            'p': float(p_q),
            '95%CI_logodds': [float(ci_low_q), float(ci_high_q)]
        }
    else:
        results['age_c2'] = None

    # Identify culture-specific interaction parameters of form: age_c:C(Culture)[T.<site>]
    interaction_pattern = re.compile(r'^age_c:C\(Culture\)\[T\.(.+)\]$')
    interaction_params = {}
    for name in param_names:
        m = interaction_pattern.match(name)
        if m:
            site = m.group(1)
            interaction_params[site] = name

    # Identify culture main effect names (for reference)
    culture_main_pattern = re.compile(r'^C\(Culture\)\[T\.(.+)\]$')
    culture_levels = set()
    for name in param_names:
        m = culture_main_pattern.match(name)
        if m:
            culture_levels.add(m.group(1))
    # The base/reference culture is the one not appearing as a main-effect param.
    # If interactions exist, include those sites plus the base (reference)
    per_culture = OrderedDict()
    if len(interaction_params) == 0:
        # No interactions: same age slope across cultures
        # Try to list cultures if available, but slope is same for all
        per_culture['(all cultures - no interactions)'] = {
            'slope_logodds': results['age_c'],
            'note': 'No age-by-culture interactions in model; linear age effect applies to all cultures equally.'
        }
    else:
        # Determine base culture name if possible (from model design info)
        # If we can access model.data.orig_exog or design_info, attempt to extract categories
        base_culture = None
        try:
            design_info = res.model.data.design_info
            # design_info.column_names contains columns; categories could be in term_names or factor_infos
            # Try to find levels for Culture factor
            for term in design_info.factor_infos:
                if 'Culture' in str(term):
                    # Not robust — skip detailed extraction
                    pass
        except Exception:
            pass
        # The set of cultures present = interaction_params.keys() union culture_levels union maybe base
        all_reported = set(interaction_params.keys()) | culture_levels
        # We don't know base label if it doesn't appear in params; mark as 'reference'
        # Compute slope for each culture: slope = beta_age + beta_interaction (if present)
        for site in sorted(all_reported):
            inter_name = interaction_params.get(site, None)
            beta_inter = params_dict.get(inter_name, 0.0) if inter_name is not None else 0.0
            slope = params_dict.get('age_c', 0.0) + beta_inter
            # compute SE using var(age_c) + var(inter) + 2*cov
            var_age = cov_elem('age_c', 'age_c')
            var_inter = cov_elem(inter_name, inter_name) if inter_name is not None else 0.0
            cov_ai = cov_elem('age_c', inter_name) if inter_name is not None else 0.0
            se_slope = np.sqrt(max(0.0, var_age + var_inter + 2.0 * cov_ai))
            z_slope = slope / se_slope if se_slope > 0 else np.nan
            if stats is not None:
                p_slope = 2 * (1 - stats.norm.cdf(abs(z_slope)))
            else:
                p_slope = _Approx.pvalue_from_z(z_slope)
            ci_low = slope - 1.96 * se_slope
            ci_high = slope + 1.96 * se_slope
            per_culture[site] = {
                'slope_logodds': float(slope),
                'se': float(se_slope),
                'z': float(z_slope),
                'p': float(p_slope),
                '95%CI_logodds': [float(ci_low), float(ci_high)],
                'OR_per_unit_age': float(np.exp(slope)),
                '95%CI_OR': [float(np.exp(ci_low)), float(np.exp(ci_high))]
            }

        # Also check for a reference (baseline) culture that may not have been in all_reported.
        # If we can detect the reference culture by inspecting model.exog_names vs param names...
        # As fallback, label baseline as 'reference (omitted)' and compute its slope as age_c alone.
        per_culture['reference (omitted)'] = {
            'slope_logodds': float(params_dict.get('age_c', 0.0)),
            'se': float(np.sqrt(max(0.0, cov_elem('age_c', 'age_c')))),
            'note': 'This is the baseline culture (the omitted category for C(Culture)).'
        }

    results['per_culture_age_slopes'] = per_culture

    # Joint test: are all age-by-culture interactions equal to zero?
    joint_test = None
    try:
        interaction_param_names = [interaction_params[s] for s in interaction_params]
        if len(interaction_param_names) > 0:
            # Build restriction matrix R that tests each interaction coeff = 0 jointly
            k = len(param_names)
            m = len(interaction_param_names)
            R = np.zeros((m, k))
            for i, pname in enumerate(interaction_param_names):
                j = param_names.index(pname)
                R[i, j] = 1.0
            # Use wald_test
            wt = res.wald_test(R)
            # wt may have statistic and pvalue attributes (WaldTestStatistic)
            stat = getattr(wt, 'statistic', None)
            pval = getattr(wt, 'pvalue', None)
            # If not directly available, try .pvalue or .df_denom...
            joint_test = {
                'wald_statistic': float(stat) if stat is not None and np.size(stat) == 1 else stat,
                'p': float(pval) if pval is not None else None,
                'tested_parameters': interaction_param_names,
                'note': 'Joint test H0: all age-by-culture interaction coefficients = 0'
            }
    except Exception:
        joint_test = {'note': 'Joint Wald test for interactions could not be computed with available result object.'}

    results['joint_interaction_test'] = joint_test

    # Prepare description string explaining how to interpret these numbers
    description_lines = []
    description_lines.append(
        "This output reports (1) the overall linear age effect (age_c) and quadratic effect (age_c2) "
        "on the log-odds of choosing the majority-demonstrated option, and (2) culture-specific linear age slopes "
        "(computed as age_c + age_c:C(Culture)[T.<site>] where applicable)."
    )
    description_lines.append(
        "For each coefficient we give the estimate on the log-odds scale, its standard error, z-statistic, p-value, "
        "and a 95% confidence interval. For linear slopes we also report the odds ratio (exp(beta)) per one-unit increase in centered age, "
        "with a 95% CI."
    )
    description_lines.append(
        "If age-by-culture interactions are present, the 'per_culture_age_slopes' table contains the estimated "
        "age slope for each culture and the baseline (omitted) culture. The 'joint_interaction_test' gives a Wald test "
        "p-value for whether all age-by-culture interactions are simultaneously zero (i.e., whether developmental slopes differ across cultures)."
    )
    description_lines.append(
        "Interpretation guidance: a positive slope_logodds indicates that with increasing age children are more likely to choose the majority option "
        "(higher reliance on majority preference); negative indicates decreasing reliance. Statistical significance can be read from the p-values or 95% CIs."
    )

    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }