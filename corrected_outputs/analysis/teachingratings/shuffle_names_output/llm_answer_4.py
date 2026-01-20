def extract_final_answer(model_output):
    """
    Extract key statistics for the coefficient on 'Beauty_z' from a fitted statsmodels
    RegressionResultsWrapper (with clustered SEs as used in the model).
    
    Returns a dictionary:
      - "object": a dict with numeric results (coef, se, t, p, 95% CI, % of 1-5 scale, nobs)
      - "description": a short plain-language interpretation of the estimate and its significance
    """
    import numpy as np

    res = model_output
    param_name = 'Beauty_z'

    # Helper to safely extract confidence interval for the parameter
    def _get_conf_int(res, name):
        try:
            # try DataFrame-like indexing first
            ci = res.conf_int().loc[name]
            return float(ci[0]), float(ci[1])
        except Exception:
            # fallback to ndarray indexing using params index
            try:
                ci_arr = res.conf_int()
                params_index = list(res.params.index)
                idx = params_index.index(name)
                return float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                return (None, None)

    # Extract statistics, with informative errors if missing
    try:
        coef = float(res.params[param_name])
    except Exception as e:
        raise KeyError(f"Could not find parameter '{param_name}' in model_output.params") from e

    try:
        se = float(res.bse[param_name])
    except Exception:
        # If bse not available for the name, set to None
        se = None

    try:
        tval = float(res.tvalues[param_name])
    except Exception:
        tval = None

    try:
        pval = float(res.pvalues[param_name])
    except Exception:
        pval = None

    ci_lower, ci_upper = _get_conf_int(res, param_name)

    # Number of observations if available
    nobs = None
    try:
        nobs = int(getattr(res, 'nobs'))
    except Exception:
        try:
            # older/newer statsmodels might store as attribute or property
            nobs = int(res.df_resid + res.df_model + 1)
        except Exception:
            nobs = None

    # Interpret magnitude relative to the 1-5 evaluation scale (range = 4)
    percent_of_4pt_scale = None
    try:
        percent_of_4pt_scale = (coef / 4.0) * 100.0
    except Exception:
        percent_of_4pt_scale = None

    # Statistical significance at conventional alpha = 0.05 (if p-value available)
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Build the returned object
    result_object = {
        "parameter": param_name,
        "coef": round(coef, 4),
        "se (clustered)": round(se, 4) if se is not None else None,
        "t": round(tval, 4) if tval is not None else None,
        "p": round(pval, 4) if pval is not None else None,
        "95% CI": (round(ci_lower, 4) if ci_lower is not None else None,
                   round(ci_upper, 4) if ci_upper is not None else None),
        "percent_of_1-5_scale": round(percent_of_4pt_scale, 2) if percent_of_4pt_scale is not None else None,
        "nobs": nobs,
        "significant_at_0.05": significant
    }

    # Compose a brief, clear description
    significance_text = ("statistically significant (p < 0.05)"
                         if significant is True else
                         "not statistically significant at the 0.05 level"
                         if significant is False else
                         "statistical significance could not be determined (p-value missing)")

    description = (
        f"The estimated effect of a 1 standard-deviation increase in instructor attractiveness "
        f"('Beauty_z') on the course evaluation score is {result_object['coef']} points "
        f"(SE = {result_object['se (clustered)']}, t = {result_object['t']}, p = {result_object['p']}). "
        f"The 95% confidence interval is [{result_object['95% CI'][0]}, {result_object['95% CI'][1]}]. "
        f"This corresponds to {result_object['percent_of_1-5_scale']}% of the 4-point range of the 1–5 evaluation scale. "
        f"The estimate was computed using clustered standard errors by ProfessorID and is {significance_text}."
    )

    return {"object": result_object, "description": description}