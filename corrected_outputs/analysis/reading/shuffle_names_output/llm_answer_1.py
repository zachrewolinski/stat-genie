def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View on reading speed (WPM),
    both the main effect (reference dyslexia category) and the effect
    within each dyslexia category (via interaction terms).

    Returns a dictionary:
      - "object": dict with keys:
          - "main_effect": {coef, se, p_value, ci_low, ci_high}
          - "by_dyslexia": { category_name: {coef, se, p_value, ci_low, ci_high}, ...}
      - "description": short text interpretation of the returned numbers.
    """
    import re
    import numpy as np
    from scipy import stats

    res = model_output  # statsmodels RegressionResultsWrapper

    params = res.params
    pvalues = res.pvalues
    conf = res.conf_int()  # default 95% conf int for parameters
    cov = res.cov_params()  # covariance matrix (should reflect HC3 used at fit time)

    # Find main reader_view_on parameter name
    # Prefer exact match 'reader_view_on', otherwise try to find a param that is just the main effect
    main_name = None
    if 'reader_view_on' in params.index:
        main_name = 'reader_view_on'
    else:
        # fallback: find any param that equals 'reader_view_on' ignoring possible stray whitespace
        for name in params.index:
            if name.strip() == 'reader_view_on':
                main_name = name
                break
    if main_name is None:
        # try to find a parameter that contains reader_view_on but not dyslexia_cat (defensive)
        for name in params.index:
            if 'reader_view_on' in name and 'dyslexia_cat' not in name:
                main_name = name
                break

    if main_name is None:
        raise KeyError("Could not locate the main effect parameter for 'reader_view_on' in model parameters.")

    # Extract main effect statistics
    main_coef = float(params[main_name])
    main_se = float(np.sqrt(cov.loc[main_name, main_name]))
    main_p = float(pvalues[main_name]) if main_name in pvalues.index else None
    z_main = main_coef / main_se if main_se != 0 else np.nan
    ci_low_main, ci_high_main = conf.loc[main_name].tolist() if main_name in conf.index else (main_coef - 1.96 * main_se, main_coef + 1.96 * main_se)

    results = {
        "main_effect": {
            "param_name": main_name,
            "coef": main_coef,
            "se": main_se,
            "z": z_main,
            "p_value": main_p,
            "ci_95_low": float(ci_low_main),
            "ci_95_high": float(ci_high_main),
            "interpretation": "Estimated change in WPM when Reader View is ON for the reference dyslexia category"
        },
        "by_dyslexia": {}
    }

    # Find all interaction parameters that link reader_view_on with dyslexia_cat
    # Accept forms like:
    #   'reader_view_on:C(dyslexia_cat)[T.Dyslexia]'
    #   'C(dyslexia_cat)[T.Dyslexia]:reader_view_on'
    interaction_names = [
        name for name in params.index
        if ('reader_view_on' in name) and ('dyslexia_cat' in name)
    ]

    # Parse category labels from the interaction parameter names
    # Regex to extract the label inside [T.<label>]
    inter_map = {}  # maps category_label -> interaction_param_name
    bracket_re = re.compile(r'\[T\.([^\]]+)\]')
    for name in interaction_names:
        m = bracket_re.search(name)
        if m:
            label = m.group(1)
        else:
            # fallback: try to split by ':' and take the part that contains 'T.'
            parts = name.split(':')
            label = None
            for p in parts:
                if 'T.' in p:
                    # take after T.
                    try:
                        label = p.split('T.', 1)[1].strip(' ]')
                    except Exception:
                        label = p
                    break
            if label is None:
                # last resort: use full param name as label
                label = name
        inter_map[label] = name

    # For each dyslexia category we can compute the Reader View effect:
    # - reference category: effect = main_coef
    # - for other categories: effect = main_coef + coef(interaction)
    # Compute standard errors for sums using covariance matrix
    from math import sqrt

    # Add reference category entry:
    results["by_dyslexia"]["(reference)"] = {
        "coef": main_coef,
        "se": main_se,
        "p_value": main_p,
        "ci_95_low": float(ci_low_main),
        "ci_95_high": float(ci_high_main),
        "note": "Reference dyslexia category used by the model (effect equals main reader_view_on coefficient)."
    }

    # Compute combined effects for each interaction (non-reference categories)
    for label, inter_name in inter_map.items():
        inter_coef = float(params[inter_name])
        combined_coef = main_coef + inter_coef

        # Compute variance of sum: var(a+b) = var(a) + var(b) + 2cov(a,b)
        try:
            var_main = float(cov.loc[main_name, main_name])
            var_inter = float(cov.loc[inter_name, inter_name])
            cov_main_inter = float(cov.loc[main_name, inter_name])
            var_comb = var_main + var_inter + 2.0 * cov_main_inter
            if var_comb < 0:
                # numerical issues: floor at small positive
                var_comb = max(var_comb, 0.0)
            combined_se = float(sqrt(var_comb))
        except Exception:
            # fallback: cannot compute combined se from covariance; mark as NaN
            combined_se = float('nan')

        # compute z and p-value for combined effect using normal approx
        if not np.isnan(combined_se) and combined_se != 0:
            z_comb = combined_coef / combined_se
            p_comb = float(2.0 * (1.0 - stats.norm.cdf(abs(z_comb))))
        else:
            z_comb = float('nan')
            p_comb = None

        # 95% CI using normal approx
        if not np.isnan(combined_se):
            ci_low = combined_coef - stats.norm.ppf(0.975) * combined_se
            ci_high = combined_coef + stats.norm.ppf(0.975) * combined_se
        else:
            ci_low = float('nan')
            ci_high = float('nan')

        results["by_dyslexia"][label] = {
            "interaction_param": inter_name,
            "coef_interaction": inter_coef,
            "coef": combined_coef,
            "se": combined_se,
            "z": z_comb,
            "p_value": p_comb,
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "interpretation": f"Estimated change in WPM when Reader View is ON for dyslexia category '{label}'"
        }

    # Short human-readable description
    description_lines = [
        "This output returns the estimated effect(s) of Reader View (reader_view_on) on reading speed (WPM).",
        "- 'main_effect' is the coefficient for reader_view_on and represents the effect for the model's reference dyslexia category.",
        "- 'by_dyslexia' gives the estimated Reader View effect within each dyslexia category:",
        "    * '(reference)' = main effect (reference group).",
        "    * other entries = main effect + interaction coefficient (with robust SE and p-value computed using the model covariance).",
        "Interpretation: coefficients are in WPM; positive values mean Reader View increases reading speed for that group, negative means it decreases speed.",
        "Use the p-values and 95% CIs to assess statistical significance and precision."
    ]
    description = " ".join(description_lines)

    return {"object": results, "description": description}