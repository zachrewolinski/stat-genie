def extract_final_answer(model_output):
    """
    Extract and summarize the effect of 'livebait' on fish caught per person-hour
    from the provided model_output (dict). Returns a dict with:
      - "object": dict of numeric results (coefficients, rate ratios, CIs, baseline rates, etc.)
      - "description": plain-language interpretation of those numbers.

    The function prefers Negative Binomial results (nb_results) when available
    (since overdispersion was detected). Falls back to Poisson otherwise.
    """
    import numpy as np
    from math import isfinite

    def fmt_num(x, fmt="{:.4f}"):
        try:
            if x is None:
                return "NA"
            if isinstance(x, (int, float)) and isfinite(x):
                return fmt.format(x)
            return str(x)
        except Exception:
            return str(x)

    results = {}

    # Determine which model to use (prefer NB if present)
    if isinstance(model_output, dict) and 'nb_results' in model_output and model_output['nb_results'] is not None:
        model = model_output['nb_results']
        model_type = 'NegativeBinomial (GLM)'
    elif isinstance(model_output, dict) and 'poisson_results' in model_output and model_output['poisson_results'] is not None:
        model = model_output['poisson_results']
        model_type = 'Poisson (GLM)'
    else:
        return {
            "object": None,
            "description": "No suitable model result found in model_output. Expected 'nb_results' or 'poisson_results'."
        }

    # Try to extract livebait coefficient and related statistics
    try:
        params = model.params
        pvalues = model.pvalues
        bse = model.bse
        ci_df = model.conf_int()  # DataFrame with two columns [0]=lower, [1]=upper
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not read parameters from the chosen model ({model_type}): {e}"
        }

    # Find the parameter name for the intercept (statsmodels may use 'Intercept' or 'const')
    intercept_name = 'Intercept' if 'Intercept' in params.index else ('const' if 'const' in params.index else None)

    if 'livebait' not in params.index:
        return {
            "object": None,
            "description": "The model does not contain a 'livebait' coefficient."
        }

    # Extract values for livebait
    try:
        coef_log = float(params['livebait'])
    except Exception:
        coef_log = None
    try:
        se = float(bse['livebait']) if ('livebait' in bse.index) else None
    except Exception:
        se = None
    try:
        pval = float(pvalues['livebait']) if ('livebait' in pvalues.index) else None
    except Exception:
        pval = None
    try:
        ci_log_lower = float(ci_df.loc['livebait', 0])
        ci_log_upper = float(ci_df.loc['livebait', 1])
    except Exception:
        ci_log_lower = ci_log_upper = None

    # Convert log-coefficient to rate ratio (multiplicative effect on rate per person-hour)
    try:
        rate_ratio = float(np.exp(coef_log)) if coef_log is not None else None
    except Exception:
        rate_ratio = None
    try:
        rr_ci_lower = float(np.exp(ci_log_lower)) if ci_log_lower is not None else None
        rr_ci_upper = float(np.exp(ci_log_upper)) if ci_log_upper is not None else None
    except Exception:
        rr_ci_lower = rr_ci_upper = None

    # Baseline rate (when livebait=0 and other covariates = 0) from intercept
    if intercept_name is not None:
        try:
            intercept_log = float(params[intercept_name])
            intercept_ci_lower = float(ci_df.loc[intercept_name, 0])
            intercept_ci_upper = float(ci_df.loc[intercept_name, 1])
            baseline_rate = float(np.exp(intercept_log))  # fish per person-hour when covariates zero
            baseline_ci_lower = float(np.exp(intercept_ci_lower))
            baseline_ci_upper = float(np.exp(intercept_ci_upper))
        except Exception:
            intercept_log = intercept_ci_lower = intercept_ci_upper = None
            baseline_rate = baseline_ci_lower = baseline_ci_upper = None
    else:
        intercept_log = None
        intercept_ci_lower = intercept_ci_upper = None
        baseline_rate = None
        baseline_ci_lower = None
        baseline_ci_upper = None

    # Absolute increase in fish per person-hour attributable to livebait (baseline * (RR - 1))
    try:
        if baseline_rate is not None and isfinite(baseline_rate) and rate_ratio is not None:
            abs_increase = baseline_rate * (rate_ratio - 1.0)
            abs_increase_ci_lower = baseline_ci_lower * (rr_ci_lower - 1.0) if (baseline_ci_lower is not None and rr_ci_lower is not None) else None
            abs_increase_ci_upper = baseline_ci_upper * (rr_ci_upper - 1.0) if (baseline_ci_upper is not None and rr_ci_upper is not None) else None
        else:
            abs_increase = abs_increase_ci_lower = abs_increase_ci_upper = None
    except Exception:
        abs_increase = abs_increase_ci_lower = abs_increase_ci_upper = None

    # Dispersion if available in model_output
    dispersion = model_output.get('dispersion', None)

    # Assemble object to return
    extracted = {
        "model_used": model_type,
        "dispersion_reported": float(dispersion) if dispersion is not None else None,
        "livebait_coef_log": coef_log,
        "livebait_se": se,
        "livebait_pvalue": pval,
        "livebait_95ci_log": [ci_log_lower, ci_log_upper],
        "livebait_rate_ratio": rate_ratio,
        "livebait_rate_ratio_95ci": [rr_ci_lower, rr_ci_upper],
        "baseline_rate_fish_per_person_hour (livebait=0, covariates=0)": baseline_rate,
        "baseline_rate_95ci": [baseline_ci_lower, baseline_ci_upper],
        "absolute_increase_fish_per_person_hour_with_livebait": abs_increase,
        "absolute_increase_95ci": [abs_increase_ci_lower, abs_increase_ci_upper],
        "significant_at_0.05": (pval is not None) and (pval < 0.05)
    }

    # Friendly textual description
    descr_lines = []
    descr_lines.append(f"Using the {model_type} results (preferred when overdispersion is present).")
    if dispersion is not None:
        try:
            descr_lines.append(f"Reported dispersion statistic: {float(dispersion):.3g}.")
        except Exception:
            descr_lines.append(f"Reported dispersion statistic: {dispersion}.")
    descr_lines.append(
        f"The estimated log rate ratio for 'livebait' is {fmt_num(coef_log, '{:.4f}')} "
        f"(SE = {fmt_num(se, '{:.4f}')}, p = {fmt_num(pval, '{:.3g}')})." )
    descr_lines.append(
        f"This corresponds to a rate ratio = exp(coef) = {fmt_num(rate_ratio, '{:.3f}')} "
        f"with 95% CI [{fmt_num(rr_ci_lower, '{:.3f}')}, {fmt_num(rr_ci_upper, '{:.3f}')}]"
    )
    if baseline_rate is not None:
        descr_lines.append(
            f"The baseline (livebait=0, other covariates=0) rate is estimated at "
            f"{fmt_num(baseline_rate, '{:.3f}')} fish per person-hour (95% CI [{fmt_num(baseline_ci_lower, '{:.3f}')}, {fmt_num(baseline_ci_upper, '{:.3f}')}])."
        )
        descr_lines.append(
            f"Thus, using live bait is associated with an estimated absolute change of "
            f"{fmt_num(abs_increase, '{:.3f}')} fish per person-hour (95% CI [{fmt_num(abs_increase_ci_lower, '{:.3f}')}, {fmt_num(abs_increase_ci_upper, '{:.3f}')}])."
        )
    if extracted["significant_at_0.05"]:
        descr_lines.append("The effect is statistically significant at the 0.05 level.")
    else:
        descr_lines.append("The effect is NOT statistically significant at the 0.05 level.")
    description = " ".join(descr_lines)

    return {
        "object": extracted,
        "description": description
    }