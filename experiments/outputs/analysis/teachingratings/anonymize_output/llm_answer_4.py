def extract_final_answer(model_output):
    """
    Extracts the coefficient and inference for the 'Beauty_z' predictor from a
    statsmodels RegressionResultsWrapper object.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coef, se, t, p, 95% CI, nobs)
      - "description": a short plain-language interpretation of the result
    """
    import numpy as np

    result = {
        "object": None,
        "description": None
    }

    if model_output is None:
        result["object"] = None
        result["description"] = "No model output provided."
        return result

    try:
        params = model_output.params
        bse = model_output.bse
        tvals = model_output.tvalues
        pvals = model_output.pvalues
        nobs = int(getattr(model_output, "nobs", np.nan))
    except Exception as e:
        result["object"] = None
        result["description"] = f"Unable to extract basic statistics from model_output: {e}"
        return result

    varname = "Beauty_z"
    if varname not in params.index:
        result["object"] = None
        result["description"] = f"Variable '{varname}' not found in model output."
        return result

    coef = float(params.loc[varname])
    se = float(bse.loc[varname]) if varname in bse.index else float(np.nan)
    t = float(tvals.loc[varname]) if varname in tvals.index else float(np.nan)
    p = float(pvals.loc[varname]) if varname in pvals.index else float(np.nan)

    # Confidence interval: handle both DataFrame/ndarray outputs from conf_int()
    try:
        ci_all = model_output.conf_int()
        try:
            # If conf_int returns a DataFrame/Series with index
            ci_lower, ci_upper = ci_all.loc[varname]
        except Exception:
            # Otherwise it's an ndarray; find the index of varname in params
            idx = list(params.index).index(varname)
            ci_lower, ci_upper = ci_all[idx]
        ci_lower = float(ci_lower)
        ci_upper = float(ci_upper)
    except Exception:
        ci_lower = ci_upper = float(np.nan)

    # Build the object dictionary with extracted numeric results
    obj = {
        "coef": coef,
        "std_err": se,
        "t_value": t,
        "p_value": p,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_obs": nobs,
        "note": "Coefficient is the change in EvalScore (scale 1-5) per 1 SD increase in instructor beauty (Beauty_z is standardized)."
    }

    # Plain-language interpretation
    # Comment on statistical significance using conventional alpha=0.05
    sig_text = "statistically significant" if (not np.isnan(p) and p < 0.05) else "not statistically significant"
    description = (
        f"The estimated effect of instructor beauty (one standard deviation increase) on course evaluation score is "
        f"{coef:.3f} (SE = {se:.3f}, t = {t:.2f}, p = {p:.3g}; 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
        f"This effect is {sig_text} at the 0.05 level. "
        f"Interpreted on the 1–5 evaluation scale: a 1 SD increase in rated beauty is associated with an average change of "
        f"{coef:.3f} points in the evaluation score. Number of observations = {nobs}."
    )

    result["object"] = obj
    result["description"] = description
    return result