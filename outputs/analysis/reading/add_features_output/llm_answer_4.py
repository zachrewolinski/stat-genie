def extract_final_answer(model_output):
    """
    Extracts the effect of activating Reader View on (log) reading speed for:
      - readers without dyslexia (dyslexia_bin = 0)
      - readers with dyslexia (dyslexia_bin = 1)
    when the model includes an interaction between reader_view and dyslexia_bin.

    Returns:
      {
        "object": {
          "dyslexia_0": {
            "coef": ...,         # effect on log_speed when dyslexia_bin=0
            "se": ...,
            "t": ...,
            "pvalue": ...,
            "ci_lower": ...,
            "ci_upper": ...,
            "percent_change": ...,           # (exp(coef)-1)*100
            "percent_change_ci": (low_pct, high_pct)
          },
          "dyslexia_1": { ... same fields for dyslexia_bin=1 ... },
          "params_used": [... list of parameter names ...]
        },
        "description": "Brief explanation of the values and how to interpret them."
      }

    Notes:
    - Positive coef on log_speed means faster reading (since DV is log(speed)).
    - Percent change translates the log-unit effect into multiplicative change in raw speed:
        percent_change = (exp(coef) - 1) * 100
    - The function relies on model_output supporting .params (indexable), .t_test(contrast),
      and returning robust covariance (as in statsmodels RegressionResultsWrapper).
    """
    import numpy as np

    res = model_output  # statsmodels results object

    # Obtain parameter values as an ndarray
    try:
        params_array = np.asarray(res.params)
    except Exception:
        # If params attribute is missing or not convertible, raise informative error
        raise AttributeError("The provided model_output does not have a usable .params attribute.")

    # Determine parameter names robustly
    param_names = None
    # If params is a pandas Series, it will have an index
    try:
        # Try to get names from params.index (works for pandas Series)
        params_attr = getattr(res, "params", None)
        if hasattr(params_attr, "index"):
            param_names = list(params_attr.index)
    except Exception:
        param_names = None

    if param_names is None:
        # Try model.exog_names (common in statsmodels)
        try:
            if hasattr(res, "model") and hasattr(res.model, "exog_names"):
                param_names = list(res.model.exog_names)
        except Exception:
            param_names = None

    if param_names is None:
        # Try attribute param_names (some objects provide it)
        try:
            if hasattr(res, "param_names"):
                param_names = list(res.param_names)
        except Exception:
            param_names = None

    if param_names is None:
        # Fallback: create generic names param_0 ... param_n-1
        n_params = params_array.size
        param_names = [f"param_{i}" for i in range(n_params)]

    n_params = len(param_names)

    # Helper to find exact param name for a variable (handles naming conventions)
    def find_param_name(varname):
        # Exact match first
        if varname in param_names:
            return varname
        # Otherwise, try to find any parameter that contains the varname as substring
        matches = [n for n in param_names if varname in n]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            # Ambiguous: return the exact match if present, otherwise raise
            raise KeyError(f"Multiple parameter names matched '{varname}': {matches}.")
        else:
            raise KeyError(f"No parameter name matched '{varname}'. Available params: {param_names}")

    # Helper to access covariance entries robustly
    def cov_at(cov, i, j):
        # cov may be a pandas DataFrame or numpy ndarray
        if hasattr(cov, "iloc"):
            return cov.iloc[i, j]
        else:
            return cov[i, j]

    # Locate main reader_view term and the interaction term
    try:
        reader_name = find_param_name('reader_view')
    except KeyError as e:
        raise KeyError("Could not find a parameter corresponding to 'reader_view' in the model params.") from e

    # Find interaction parameter that contains both reader_view and dyslexia_bin
    interaction_candidates = [n for n in param_names if ('reader_view' in n and 'dyslexia_bin' in n)]
    if len(interaction_candidates) == 1:
        interaction_name = interaction_candidates[0]
    elif len(interaction_candidates) == 0:
        # No explicit interaction param found: treat only main effect available
        interaction_name = None
    else:
        # Multiple matches (unexpected)
        raise KeyError(f"Multiple interaction parameter names matched reader_view and dyslexia_bin: {interaction_candidates}")

    # Build contrasts and use model_output.t_test to get robust std errors / p-values for linear combos
    results_dict = {}
    results_dict['params_used'] = param_names

    # Effect for dyslexia = 0 is simply the reader_view coefficient
    contrast0 = np.zeros(n_params)
    pos_reader = param_names.index(reader_name)
    contrast0[pos_reader] = 1.0

    tt0 = res.t_test(contrast0)  # statsmodels ContrastResults

    # Extract scalar effect, se, t, pvalue robustly
    def scalar_from_result_field(field):
        # field may be array-like or scalar
        try:
            arr = np.asarray(field)
            return float(np.atleast_1d(arr).ravel()[0])
        except Exception:
            return float(field)

    coef0 = scalar_from_result_field(tt0.effect)
    # Try to get sd from tt0; fallback to covariance
    try:
        se0 = scalar_from_result_field(tt0.sd)
    except Exception:
        cov = res.cov_params()
        var0 = cov_at(cov, pos_reader, pos_reader)
        se0 = float(np.sqrt(var0))
    t0 = scalar_from_result_field(tt0.tvalue)
    p0 = scalar_from_result_field(tt0.pvalue)
    ci0 = tt0.conf_int(alpha=0.05)
    # ci0 is typically an array shape (1,2)
    ci0_low, ci0_high = float(ci0[0, 0]), float(ci0[0, 1])
    pct0 = (np.exp(coef0) - 1.0) * 100.0
    pct0_ci = ((np.exp(ci0_low) - 1.0) * 100.0, (np.exp(ci0_high) - 1.0) * 100.0)

    results_dict['dyslexia_0'] = {
        'coef': coef0,
        'se': se0,
        't': t0,
        'pvalue': p0,
        'ci_lower': ci0_low,
        'ci_upper': ci0_high,
        'percent_change': pct0,
        'percent_change_ci': pct0_ci
    }

    # Effect for dyslexia = 1: reader_view + interaction
    if interaction_name is not None:
        contrast1 = np.zeros(n_params)
        pos_inter = param_names.index(interaction_name)
        contrast1[pos_reader] = 1.0
        contrast1[pos_inter] = 1.0
        tt1 = res.t_test(contrast1)
        coef1 = scalar_from_result_field(tt1.effect)
        try:
            se1 = scalar_from_result_field(tt1.sd)
        except Exception:
            cov = res.cov_params()
            var1 = cov_at(cov, pos_reader, pos_reader) + cov_at(cov, pos_inter, pos_inter) + 2 * cov_at(cov, pos_reader, pos_inter)
            se1 = float(np.sqrt(var1))
        t1 = scalar_from_result_field(tt1.tvalue)
        p1 = scalar_from_result_field(tt1.pvalue)
        ci1 = tt1.conf_int(alpha=0.05)
        ci1_low, ci1_high = float(ci1[0, 0]), float(ci1[0, 1])
        pct1 = (np.exp(coef1) - 1.0) * 100.0
        pct1_ci = ((np.exp(ci1_low) - 1.0) * 100.0, (np.exp(ci1_high) - 1.0) * 100.0)

        results_dict['dyslexia_1'] = {
            'coef': coef1,
            'se': se1,
            't': t1,
            'pvalue': p1,
            'ci_lower': ci1_low,
            'ci_upper': ci1_high,
            'percent_change': pct1,
            'percent_change_ci': pct1_ci
        }
    else:
        # No interaction term present; effect is same for dyslexia=1 as dyslexia=0 (model does not allow moderation)
        results_dict['dyslexia_1'] = {
            'coef': coef0,
            'se': se0,
            't': t0,
            'pvalue': p0,
            'ci_lower': ci0_low,
            'ci_upper': ci0_high,
            'percent_change': pct0,
            'percent_change_ci': pct0_ci,
            'note': "No reader_view:dyslexia_bin interaction present in model; effect assumed identical to dyslexia=0."
        }

    # Interpretation description
    description_lines = [
        "This output reports the estimated effect of activating Reader View on log(reading speed).",
        "- 'dyslexia_0' gives the effect when dyslexia_bin = 0 (readers without dyslexia).",
        "- 'dyslexia_1' gives the effect when dyslexia_bin = 1 (readers with dyslexia); computed as reader_view + interaction if present.",
        "- 'coef' is the change in log_speed. Positive values indicate faster reading (since DV is log(speed)).",
        "- 'percent_change' converts the log effect to percent change in raw speed: (exp(coef)-1)*100.",
        "- 'pvalue' and the 95% CI indicate statistical uncertainty. A conventional threshold for evidence of an effect is p < 0.05.",
        "To answer whether Reader View improves reading speed:",
        " - Check the sign and p-value for the appropriate group (dyslexia_0 and/or dyslexia_1).",
        " - If coef > 0 and p < 0.05, that is evidence Reader View increases reading speed for that group.",
        " - If the interaction (difference between dyslexia_1 and dyslexia_0) is statistically significant (p < 0.05), then the effect differs meaningfully by dyslexia status."
    ]
    description = " ".join(description_lines)

    return {"object": results_dict, "description": description}