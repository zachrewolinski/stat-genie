def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the predictors
    of interest from a statsmodels results object (MixedLMResultsWrapper or OLS/RegressionResultsWrapper).
    Also compute the marginal effect (slope) of age when Help_Y = 0 and when Help_Y = 1
    using the interaction term, if present.

    Returns:
      {
        "object": {
           "terms": {
              "age": {coef, se, p, ci_lower, ci_upper, significant},
              "Sex_M": { ... },
              "Help_Y": { ... },
              "age:Help_Y": { ... or None}
           },
           "age_slope_when_Help0": {value, interpretation},
           "age_slope_when_Help1": {value, interpretation},
           "notes": "any notes about missing terms or assumptions"
        },
        "description": "Plain-language interpretation of the key results"
      }
    """
    import numpy as np

    # Helper to safely pull arrays/values and convert to native Python floats
    def _safe_get(series_like, key):
        try:
            val = series_like[key]
            # convert numpy / pandas types to native floats
            if isinstance(val, (np.floating, np.integer)):
                return float(val)
            return float(np.asarray(val).item()) if np.asarray(val).size == 1 else val
        except Exception:
            return None

    # Attempt to obtain params, bse, pvalues, conf_int
    params = None
    bse = None
    pvalues = None
    conf_int = None

    # Many statsmodels result objects expose these attributes
    if hasattr(model_output, "params"):
        params = model_output.params
    if hasattr(model_output, "bse"):
        bse = model_output.bse
    if hasattr(model_output, "pvalues"):
        pvalues = model_output.pvalues
    # conf_int() is a method
    try:
        conf_int = model_output.conf_int()
    except Exception:
        conf_int = None

    if params is None:
        raise ValueError("The provided model_output does not expose 'params'; cannot extract results.")

    # Parameter names we want to extract
    # The exact interaction parameter name may vary ordering ('age:Help_Y' or 'Help_Y:age'),
    # so we search for a parameter name that contains both substrings.
    def find_param_name(target):
        # exact match first
        if target in params.index:
            return target
        # otherwise do contains match
        target_parts = target.split(':')
        for name in params.index:
            name_lower = name.lower()
            if all(part.lower() in name_lower for part in target_parts):
                return name
        return None

    name_age = find_param_name('age')
    name_sex = find_param_name('Sex_M') or find_param_name('sex_m') or find_param_name('SexM') or find_param_name('sex')
    name_help = find_param_name('Help_Y') or find_param_name('help_y') or find_param_name('HelpY') or find_param_name('help')
    # find interaction (contains both age and help)
    name_inter = None
    for name in params.index:
        low = name.lower()
        if 'age' in low and ('help_y' in low or 'helpy' in low or 'help' in low):
            name_inter = name
            break

    terms = {}
    notes = []

    def collect_stats(param_name, label):
        if param_name is None:
            notes.append(f"Parameter for {label} not found in model parameters.")
            return None
        coef = _safe_get(params, param_name)
        se = _safe_get(bse, param_name) if bse is not None else None
        p = _safe_get(pvalues, param_name) if pvalues is not None else None
        if conf_int is not None:
            # conf_int may be a DataFrame or ndarray with two columns
            try:
                ci_row = conf_int.loc[param_name]
                ci_lower = float(ci_row[0])
                ci_upper = float(ci_row[1])
            except Exception:
                # fallback: use se to compute approx 95% CI
                if (coef is not None) and (se is not None):
                    ci_lower = float(coef - 1.96 * se)
                    ci_upper = float(coef + 1.96 * se)
                else:
                    ci_lower = ci_upper = None
        else:
            if (coef is not None) and (se is not None):
                ci_lower = float(coef - 1.96 * se)
                ci_upper = float(coef + 1.96 * se)
            else:
                ci_lower = ci_upper = None

        significant = None
        if p is not None:
            try:
                significant = bool(p < 0.05)
            except Exception:
                significant = None

        return {
            "param_name": param_name,
            "coef": coef,
            "se": se,
            "p_value": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant_p_lt_0.05": significant
        }

    terms['age'] = collect_stats(name_age, 'age')
    terms['Sex_M'] = collect_stats(name_sex, 'Sex_M')
    terms['Help_Y'] = collect_stats(name_help, 'Help_Y')
    terms['age:Help_Y'] = collect_stats(name_inter, 'age:Help_Y')

    # Compute marginal slopes for age when Help_Y = 0 and when Help_Y = 1
    age_coef = terms['age']['coef'] if terms['age'] is not None else None
    inter_coef = terms['age:Help_Y']['coef'] if terms['age:Help_Y'] is not None else 0.0  # treat missing as 0

    age_slope_help0 = None
    age_slope_help1 = None
    try:
        if age_coef is not None:
            age_slope_help0 = float(age_coef)  # slope of age when Help_Y = 0
            age_slope_help1 = float(age_coef + (inter_coef if inter_coef is not None else 0.0))
        else:
            notes.append("Could not compute age slopes because age coefficient is missing.")
    except Exception:
        age_slope_help0 = age_slope_help1 = None
        notes.append("Error computing marginal age slopes.")

    # Prepare a concise plain-language description
    # Focus on direction, magnitude in log1p units, and interaction presence.
    desc_lines = []
    if terms['age'] is not None and terms['age']['coef'] is not None:
        desc_lines.append(
            f"Age effect (when Help_Y=0): coefficient = {terms['age']['coef']:.4f} "
            f"(95% CI [{terms['age']['ci_lower']:.4f}, {terms['age']['ci_upper']:.4f}], "
            f"p = {terms['age']['p_value']:.3g})"
        )
    else:
        desc_lines.append("Age effect: parameter not found or not estimable.")

    if terms['Help_Y'] is not None and terms['Help_Y']['coef'] is not None:
        desc_lines.append(
            f"Main effect of receiving help (Help_Y): coefficient = {terms['Help_Y']['coef']:.4f} "
            f"(95% CI [{terms['Help_Y']['ci_lower']:.4f}, {terms['Help_Y']['ci_upper']:.4f}], "
            f"p = {terms['Help_Y']['p_value']:.3g})"
        )
    else:
        desc_lines.append("Help_Y effect: parameter not found or not estimable.")

    if terms['age:Help_Y'] is not None and terms['age:Help_Y']['coef'] is not None:
        desc_lines.append(
            f"Age × Help_Y interaction: coefficient = {terms['age:Help_Y']['coef']:.4f} "
            f"(95% CI [{terms['age:Help_Y']['ci_lower']:.4f}, {terms['age:Help_Y']['ci_upper']:.4f}], "
            f"p = {terms['age:Help_Y']['p_value']:.3g})"
        )
        desc_lines.append(
            f"Implied age slope when Help_Y=0: {age_slope_help0:.4f}; "
            f"when Help_Y=1: {age_slope_help1:.4f}."
        )
        desc_lines.append(
            "Interpretation: coefficients are in log1p(efficiency) units. A difference of Δ in this scale "
            "corresponds to a multiplicative change of exp(Δ) in (1 + nuts_opened_per_second)."
        )
    else:
        desc_lines.append("No estimated interaction term found; age effect does not vary with Help_Y in this model.")

    # Summarize significance for sex
    if terms['Sex_M'] is not None and terms['Sex_M']['coef'] is not None:
        desc_lines.append(
            f"Sex (Male vs Female): coefficient = {terms['Sex_M']['coef']:.4f} "
            f"(95% CI [{terms['Sex_M']['ci_lower']:.4f}, {terms['Sex_M']['ci_upper']:.4f}], "
            f"p = {terms['Sex_M']['p_value']:.3g})"
        )
    else:
        desc_lines.append("Sex effect: parameter not found or not estimable.")

    description = " ".join(desc_lines)

    result_object = {
        "terms": terms,
        "age_slope_when_Help0": age_slope_help0,
        "age_slope_when_Help1": age_slope_help1,
        "notes": notes
    }

    return {
        "object": result_object,
        "description": description
    }