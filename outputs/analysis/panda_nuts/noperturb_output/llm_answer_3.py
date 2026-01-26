def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, and multiplicative effects
    (exp(coef)) for the predictors of interest from a fitted statsmodels model output
    (MixedLMResultsWrapper or OLSResults, or similar).

    Returns a dictionary with:
      - "object": dict keyed by variable name ('age_c', 'sex_m', 'help_yes') containing numeric stats
      - "description": textual explanation of the returned numbers and how to interpret them

    Interpretation notes:
      - The dependent variable is log_rate (log of nut-cracking rate). Coefficients are on the log scale.
        exp(coef) gives the multiplicative change in the nut-cracking rate for a one-unit increase
        in the predictor (or switching 0->1 for binary predictors). (exp(coef)-1)*100 gives percent change.
      - age_c is in years (mean-centered): coef = log-ratio per additional year.
      - sex_m is binary: 1 = male, 0 = female/other; coef compares males to reference.
      - help_yes is binary: 1 = received help, 0 = did not; coef compares helped vs not.
    """
    import numpy as np
    from math import exp
    # Try to import a normal distribution for p-value fallback
    try:
        from scipy.stats import norm
    except Exception:
        # Minimal fallback: use math.erfc for approximate tail (not ideal), but scipy is usually available.
        norm = None

    # Extract parameter estimates and standard errors
    if not (hasattr(model_output, 'params') and hasattr(model_output, 'bse')):
        raise ValueError("model_output must have 'params' and 'bse' attributes (statsmodels results object expected).")

    params = model_output.params
    bse = model_output.bse

    # Ensure we have indexable names and positions
    if hasattr(params, 'index'):
        param_names = list(params.index)
        get_param = lambda name: float(params[name])
        get_se = lambda name: float(bse[name])
    else:
        # params might be a numpy array; try to get names from model if possible
        try:
            param_names = list(model_output.model.exog_names)
        except Exception:
            raise ValueError("Unable to determine parameter names from the model output.")
        params = np.asarray(params)
        bse = np.asarray(bse)
        get_param = lambda name: float(params[param_names.index(name)])
        get_se = lambda name: float(bse[param_names.index(name)])

    # p-values: use model's pvalues if present, otherwise normal approx
    if hasattr(model_output, 'pvalues'):
        pvalues_available = True
        pvals = model_output.pvalues
        get_p = (lambda name: float(pvals[name])) if hasattr(pvals, 'index') else (lambda name: float(pvals[param_names.index(name)]))
    else:
        # compute z and use normal approx
        pvalues_available = False
        if norm is None:
            raise ImportError("scipy is required to compute p-values when model output lacks them.")
        def get_p(name):
            z = get_param(name) / get_se(name)
            return float(2.0 * norm.sf(abs(z)))

    # Confidence intervals: try model.conf_int(), otherwise use param +/- 1.96*SE
    if hasattr(model_output, 'conf_int'):
        try:
            ci_df = model_output.conf_int()
            have_ci_df = True
        except Exception:
            have_ci_df = False
            ci_df = None
    else:
        have_ci_df = False
        ci_df = None

    def get_ci(name):
        if have_ci_df:
            # ci_df may be a DataFrame or ndarray
            if hasattr(ci_df, 'loc'):
                # DataFrame
                low = float(ci_df.loc[name].iloc[0])
                high = float(ci_df.loc[name].iloc[1])
            else:
                # ndarray: need position
                pos = param_names.index(name)
                low = float(ci_df[pos, 0])
                high = float(ci_df[pos, 1])
        else:
            coef = get_param(name)
            se = get_se(name)
            low = float(coef - 1.96 * se)
            high = float(coef + 1.96 * se)
        return low, high

    variables_of_interest = ['age_c', 'sex_m', 'help_yes']
    results = {}
    for v in variables_of_interest:
        if v in param_names:
            coef = get_param(v)
            se = get_se(v)
            p = get_p(v)
            ci_low, ci_high = get_ci(v)
            ratio = exp(coef)  # multiplicative effect on the rate
            pct_change = (ratio - 1.0) * 100.0
            results[v] = {
                'coef': coef,
                'se': se,
                'p_value': p,
                'ci_2.5%': ci_low,
                'ci_97.5%': ci_high,
                'exp(coef)_rate_ratio': ratio,
                'percent_change_in_rate': pct_change
            }
        else:
            results[v] = None  # predictor not present in model

    # Provide some metadata about the model used / fallback
    model_type = type(model_output).__name__
    fallback_note = getattr(model_output, 'fallback_note', None)

    description_lines = [
        f"Extracted estimated coefficients, standard errors, 95% CIs, and p-values for predictors from the fitted model ({model_type}).",
        "Because the dependent variable is log_rate, coefficients are on the log scale:",
        "- exp(coef) gives the multiplicative change in nut-cracking rate for a one-unit increase in the predictor (or 0->1 for binaries).",
        "- (exp(coef) - 1) * 100 gives the percent change in rate.",
        "Variables reported:",
        "- age_c: change in log-rate per additional year of age (age is mean-centered).",
        "- sex_m: difference (male vs female/other).",
        "- help_yes: difference (received help vs not)."
    ]
    if fallback_note:
        description_lines.append(f"Note: model object contains fallback note: {fallback_note}")
    if not pvalues_available:
        description_lines.append("P-values were computed using normal (Wald) approximation since the model object lacked p-values.")
    description = " ".join(description_lines)

    return {"object": results, "description": description}