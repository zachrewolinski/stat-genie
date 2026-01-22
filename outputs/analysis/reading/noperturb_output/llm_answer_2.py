def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of Reader View on log_speed,
    and specifically the interaction with dyslexia status.

    Returns a dict with keys:
      - "object": dict of extracted numeric results:
          * main_reader_view: effect of reader_view for non-dyslexic (coef, se, p, CI, pct_change)
          * interaction: coefficient for reader_view:dyslexia (coef, se, p, CI)
          * reader_view_for_dyslexic: combined effect for dyslexic = main + interaction
                                   (coef, se, p, CI, pct_change)
          * raw_params: full params Series (for reference)
      - "description": human-readable interpretation of the above.

    The function is defensive about parameter name variants (e.g. categorical coding
    names like "dyslexia_bin[T.1]" or "reader_view:dyslexia_bin[T.1]").
    """
    import numpy as np
    import math

    # Get fixed-effect parameters and covariance
    try:
        params = model_output.params  # pandas Series
    except Exception as e:
        raise ValueError("Could not extract params from model_output: " + str(e))

    try:
        cov = model_output.cov_params()
    except Exception as e:
        cov = None  # we'll avoid computing combined SE if not available

    # Helper to find parameter name that contains all include_substrings and none of exclude_substrings
    def find_param_name(include_substrings, exclude_substrings=None):
        exclude_substrings = exclude_substrings or []
        for name in params.index:
            name_str = str(name)
            if all(sub in name_str for sub in include_substrings) and not any(sub in name_str for sub in exclude_substrings):
                return name_str
        return None

    # Find main reader_view parameter (prefer exact 'reader_view' otherwise a name containing reader_view but not dyslexia)
    main_name = None
    if 'reader_view' in params.index:
        main_name = 'reader_view'
    else:
        main_name = find_param_name(['reader_view'], ['dyslexia', 'dyslexia_bin', ':'])
        # allow fallback to any name containing 'reader_view' if still None
        if main_name is None:
            main_name = find_param_name(['reader_view'], [])

    # Find interaction term name: something that contains both reader_view and dyslexia (or dyslexia_bin)
    interaction_name = find_param_name(['reader_view', 'dyslexia']) or find_param_name(['reader_view', 'dyslexia_bin'])
    # Also consider common categorical naming
    if interaction_name is None:
        # try detecting patterns like 'reader_view:dyslexia_bin[T.1]' or 'reader_view:C(dyslexia_bin)[T.1]'
        for name in params.index:
            s = str(name)
            if 'reader_view' in s and ('dyslexia' in s or 'dyslexia_bin' in s):
                interaction_name = s
                break

    # Validate found names
    if main_name is None:
        raise ValueError("Could not locate a parameter corresponding to the main effect of 'reader_view' in model params: "
                         + ", ".join(map(str, params.index)))

    # Extract main effect estimates
    main_coef = float(params[main_name])
    main_se = float(model_output.bse[main_name]) if hasattr(model_output, 'bse') and main_name in model_output.bse.index else (float(np.sqrt(cov.loc[main_name, main_name])) if cov is not None and main_name in cov.index else None)

    # p-value for main effect
    main_p = None
    if hasattr(model_output, 'pvalues') and main_name in model_output.pvalues.index:
        main_p = float(model_output.pvalues[main_name])
    elif main_se is not None and main_se > 0:
        z = main_coef / main_se
        try:
            from scipy.stats import norm
            main_p = float(2 * norm.sf(abs(z)))
        except Exception:
            main_p = float(math.erfc(abs(z) / math.sqrt(2)))
    # CI approx (normal)
    if main_se is not None:
        ci_lower_main = main_coef - 1.96 * main_se
        ci_upper_main = main_coef + 1.96 * main_se
    else:
        ci_lower_main = ci_upper_main = None

    # Interaction estimates (if present)
    interaction_coef = interaction_se = interaction_p = ci_lower_int = ci_upper_int = None
    if interaction_name is not None:
        interaction_coef = float(params[interaction_name])
        interaction_se = float(model_output.bse[interaction_name]) if hasattr(model_output, 'bse') and interaction_name in model_output.bse.index else (float(np.sqrt(cov.loc[interaction_name, interaction_name])) if cov is not None and interaction_name in cov.index else None)
        if hasattr(model_output, 'pvalues') and interaction_name in model_output.pvalues.index:
            interaction_p = float(model_output.pvalues[interaction_name])
        elif interaction_se is not None and interaction_se > 0:
            z = interaction_coef / interaction_se
            try:
                from scipy.stats import norm
                interaction_p = float(2 * norm.sf(abs(z)))
            except Exception:
                interaction_p = float(math.erfc(abs(z) / math.sqrt(2)))
        if interaction_se is not None:
            ci_lower_int = interaction_coef - 1.96 * interaction_se
            ci_upper_int = interaction_coef + 1.96 * interaction_se

    # Combined effect for dyslexic individuals = main_coef + interaction_coef (if interaction exists)
    combined = None
    if interaction_coef is not None and cov is not None and main_name in cov.index and interaction_name in cov.index:
        comb_coef = main_coef + interaction_coef
        # variance of sum = var(a)+var(b)+2cov(a,b)
        var_sum = cov.loc[main_name, main_name] + cov.loc[interaction_name, interaction_name] + 2.0 * cov.loc[main_name, interaction_name]
        comb_se = float(np.sqrt(var_sum)) if var_sum >= 0 else None
        # p-value
        if comb_se is not None and comb_se > 0:
            z = comb_coef / comb_se
            try:
                from scipy.stats import norm
                comb_p = float(2 * norm.sf(abs(z)))
            except Exception:
                comb_p = float(math.erfc(abs(z) / math.sqrt(2)))
        else:
            comb_p = None
        comb_ci_lower = comb_coef - 1.96 * comb_se if comb_se is not None else None
        comb_ci_upper = comb_coef + 1.96 * comb_se if comb_se is not None else None
        combined = {
            'coef': comb_coef,
            'se': comb_se,
            'p_value': comb_p,
            'ci_95_lower': comb_ci_lower,
            'ci_95_upper': comb_ci_upper,
            # convert from log-scale to percent change in speed: (exp(coef)-1)*100
            'pct_change': (np.exp(comb_coef) - 1.0) * 100 if comb_coef is not None else None
        }
    elif interaction_coef is None:
        # If no interaction term present, the main effect applies to both groups
        combined = {
            'coef': main_coef,
            'se': main_se,
            'p_value': main_p,
            'ci_95_lower': ci_lower_main,
            'ci_95_upper': ci_upper_main,
            'pct_change': (np.exp(main_coef) - 1.0) * 100 if main_coef is not None else None
        }
    else:
        # interaction exists but covariance not available; compute combined coef but not SE/p
        comb_coef = main_coef + interaction_coef
        combined = {
            'coef': comb_coef,
            'se': None,
            'p_value': None,
            'ci_95_lower': None,
            'ci_95_upper': None,
            'pct_change': (np.exp(comb_coef) - 1.0) * 100 if comb_coef is not None else None
        }

    # Prepare output object
    output_object = {
        'main_reader_view': {
            'param_name': main_name,
            'coef': main_coef,
            'se': main_se,
            'p_value': main_p,
            'ci_95_lower': ci_lower_main,
            'ci_95_upper': ci_upper_main,
            'pct_change': (np.exp(main_coef) - 1.0) * 100 if main_coef is not None else None,
            'interpretation': (
                "Effect of Reader View for the reference group (likely non-dyslexic if dyslexia_bin=0). "
                "Dependent variable is log(speed + 1), so coef is approximately the log multiplicative change; "
                "pct_change gives percent change in speed."
            )
        },
        'interaction': {
            'param_name': interaction_name,
            'coef': interaction_coef,
            'se': interaction_se,
            'p_value': interaction_p,
            'ci_95_lower': ci_lower_int,
            'ci_95_upper': ci_upper_int,
            'interpretation': (
                "Interaction term: additional effect of Reader View when dyslexia_bin=1. "
                "If positive, Reader View effect is larger for dyslexic participants by this amount (in log units)."
            ) if interaction_name is not None else "No interaction term found in model parameters."
        },
        'reader_view_for_dyslexic': combined,
        'raw_params': params.to_dict()
    }

    # Human readable description
    if combined and combined.get('p_value') is not None:
        if combined['p_value'] < 0.05:
            conclusion = ("Reader View has a statistically significant effect on reading speed for dyslexic participants "
                          f"(combined coef = {combined['coef']:.4f}, p = {combined['p_value']:.3g}). "
                          f"This corresponds to an estimated {combined['pct_change']:.1f}% change in speed.")
        else:
            conclusion = ("Reader View does not show a statistically significant effect on reading speed for dyslexic participants "
                          f"(combined coef = {combined['coef']:.4f}, p = {combined['p_value']:.3g}).")
    else:
        conclusion = ("Could not compute a p-value for the combined effect on dyslexic participants (missing SE/covariance). "
                      "Returned coefficients and approximate percent-change; consider inspecting model summary and covariance matrix.")

    description = (
        "Extracted estimates for the Reader View main effect and its interaction with dyslexia. "
        "Key objects:\n"
        "- main_reader_view: effect of Reader View for the reference group (likely non-dyslexic).\n"
        "- interaction: additional effect when dyslexia_bin=1.\n"
        "- reader_view_for_dyslexic: combined effect (main + interaction) representing Reader View effect for dyslexic participants; "
        "contains coef, SE, p-value (if covariance available), 95% CI, and percent change on original speed scale.\n\n"
        "Conclusion summary: " + conclusion
    )

    return {"object": output_object, "description": description}