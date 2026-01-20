def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and simple slopes for the
    age x help interaction from a statsmodels MixedLMResults (or MixedLMResultsWrapper).

    Returns:
      {
        "object": dict containing numeric results for terms and simple slopes,
        "description": human-readable interpretation of the key results
      }
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output  # expected to be a statsmodels MixedLMResults-like object

    # Basic extracts
    params = res.params.copy()        # pandas Series
    try:
        bse = res.bse.copy()
    except Exception:
        bse = res.bse  # fallback
    try:
        pvalues = res.pvalues.copy()
    except Exception:
        pvalues = res.pvalues
    try:
        ci = res.conf_int().copy()
    except Exception:
        ci = res.conf_int()

    cov = res.cov_params()  # covariance matrix of fixed-effect params

    # Helper to robustly find parameter names (interaction naming can vary)
    def find_param(name_fragments):
        # name_fragments: list of substrings that must be in the parameter name
        matches = [n for n in params.index if all(f in n for f in name_fragments)]
        return matches[0] if matches else None

    # Identify the parameter names
    age_name = find_param(['age_c'])  # should be 'age_c'
    sex_name = find_param(['sex_male'])
    help_name = find_param(['help_yes'])
    # interaction name contains both 'age_c' and 'help_yes'
    interaction_name = find_param(['age_c', 'help_yes'])

    # Collect results for main terms we care about
    def term_info(name):
        if name is None:
            return None
        return {
            'name': name,
            'estimate': float(params[name]),
            'se': float(bse[name]) if name in bse.index else None,
            'p_value': float(pvalues[name]) if name in pvalues.index else None,
            'ci_2.5': float(ci.loc[name, 0]) if name in ci.index else None,
            'ci_97.5': float(ci.loc[name, 1]) if name in ci.index else None,
            'significant_p05': (float(pvalues[name]) < 0.05) if name in pvalues.index else None
        }

    results = {
        'age': term_info(age_name),
        'sex_male': term_info(sex_name),
        'help_yes': term_info(help_name),
        'age_x_help_interaction': term_info(interaction_name),
        'notes': 'Estimates are changes in nuts opened per minute. Age is centered.'
    }

    # Simple slopes: effect of age when help_no (help=0) and help_yes (help=1)
    # slope_no = coef(age_c)
    if age_name is not None:
        slope_no = float(params[age_name])
        se_no = float(bse[age_name]) if age_name in bse.index else None
        p_no = float(pvalues[age_name]) if age_name in pvalues.index else None
    else:
        slope_no = se_no = p_no = None

    # slope_yes = coef(age_c) + coef(age_c:help_yes)
    if age_name is not None and interaction_name is not None:
        slope_yes = float(params[age_name] + params[interaction_name])

        # Compute SE for slope_yes using covariance matrix:
        # Var(a + b) = Var(a) + Var(b) + 2Cov(a,b)
        try:
            var_a = float(cov.loc[age_name, age_name])
            var_b = float(cov.loc[interaction_name, interaction_name])
            cov_ab = float(cov.loc[age_name, interaction_name])
            se_yes = float(np.sqrt(var_a + var_b + 2.0 * cov_ab))
            # z and p-value (Wald)
            z_yes = slope_yes / se_yes if se_yes != 0 else np.nan
            p_yes = float(2.0 * (1.0 - norm.cdf(abs(z_yes))))
        except Exception:
            se_yes = None
            p_yes = None
    else:
        slope_yes = se_yes = p_yes = None

    results['simple_slopes'] = {
        'slope_age_when_no_help': {
            'estimate': slope_no,
            'se': se_no,
            'p_value': p_no,
            'interpretation': 'Change in nuts/min for a one-unit increase in centered age when no help was received.'
        },
        'slope_age_when_help': {
            'estimate': slope_yes,
            'se': se_yes,
            'p_value': p_yes,
            'interpretation': 'Change in nuts/min for a one-unit increase in centered age when help was received.'
        }
    }

    # Build short textual description
    desc_lines = []
    # Age main effect
    if results['age'] is not None:
        desc_lines.append(
            f"Age (centered): estimate={results['age']['estimate']:.3f}, "
            f"SE={results['age']['se']:.3f}, p={results['age']['p_value']:.3f}."
        )
    else:
        desc_lines.append("Age term not found in model output.")

    # Sex
    if results['sex_male'] is not None:
        desc_lines.append(
            f"Sex (male vs female): estimate={results['sex_male']['estimate']:.3f}, "
            f"SE={results['sex_male']['se']:.3f}, p={results['sex_male']['p_value']:.3f}."
        )
    else:
        desc_lines.append("Sex term not found in model output.")

    # Help main effect
    if results['help_yes'] is not None:
        desc_lines.append(
            f"Help (received vs not): estimate={results['help_yes']['estimate']:.3f}, "
            f"SE={results['help_yes']['se']:.3f}, p={results['help_yes']['p_value']:.3f}."
        )
    else:
        desc_lines.append("Help term not found in model output.")

    # Interaction
    if results['age_x_help_interaction'] is not None:
        desc_lines.append(
            f"Age x Help interaction: estimate={results['age_x_help_interaction']['estimate']:.3f}, "
            f"SE={results['age_x_help_interaction']['se']:.3f}, p={results['age_x_help_interaction']['p_value']:.3f}."
        )
        # add simple slopes summary
        ss = results['simple_slopes']
        desc_lines.append(
            f"Simple slope of age when no help: estimate={ss['slope_age_when_no_help']['estimate']:.3f}, "
            f"p={ss['slope_age_when_no_help']['p_value']}; when help: "
            f"estimate={ss['slope_age_when_help']['estimate']:.3f}, p={ss['slope_age_when_help']['p_value']}."
        )
    else:
        desc_lines.append("No age x help interaction term found; simple slopes not computed.")

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }