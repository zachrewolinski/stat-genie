def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, 95% CI, and incidence rate ratio (IRR = exp(coef))
    for the skin-tone variables ('skin_dark' and 'skin_avg') from a statsmodels-like
    robust results object (the object returned by get_robustcov_results).
    Returns a dictionary with keys "object" (detailed numeric results) and
    "description" (concise interpretation addressing whether darker-skinned players
    are more likely to receive red cards).
    """
    import numpy as np
    import pandas as pd

    # Helper to try multiple attribute/access patterns
    def _get_attr(obj, attr):
        return getattr(obj, attr, None)

    # Try to obtain parameter names and arrays/series for params, bse, pvalues
    params = _get_attr(model_output, "params")
    bse = _get_attr(model_output, "bse")
    pvalues = _get_attr(model_output, "pvalues")

    # If any are None, try digging into ._results or .results if present
    if params is None or bse is None or pvalues is None:
        inner = _get_attr(model_output, "_results") or _get_attr(model_output, "results")
        if inner is not None:
            params = params or _get_attr(inner, "params")
            bse = bse or _get_attr(inner, "bse")
            pvalues = pvalues or _get_attr(inner, "pvalues")

    # Get parameter names
    if isinstance(params, (pd.Series, pd.DataFrame)):
        names = list(params.index)
    elif hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
        names = list(model_output.model.exog_names)
    else:
        # fallback: try to coerce to list of strings by length of params
        try:
            length = len(params)
            names = [f"param_{i}" for i in range(length)]
        except Exception:
            names = []

    # Convert params, bse, pvalues to numpy arrays and to pandas Series for easy lookup
    try:
        params_s = pd.Series(params, index=names)
    except Exception:
        params_s = pd.Series(np.asarray(params), index=names)

    try:
        bse_s = pd.Series(bse, index=names)
    except Exception:
        bse_s = pd.Series(np.asarray(bse), index=names)

    try:
        pvalues_s = pd.Series(pvalues, index=names)
    except Exception:
        pvalues_s = pd.Series(np.asarray(pvalues), index=names)

    # Confidence intervals: try model_output.conf_int() (common API)
    ci = None
    conf_int_fn = _get_attr(model_output, "conf_int") or _get_attr(model_output, "conf_int()")
    try:
        # prefer calling conf_int() if callable
        if callable(_get_attr(model_output, "conf_int")):
            ci_raw = model_output.conf_int()
        else:
            ci_raw = _get_attr(model_output, "conf_int")()
    except Exception:
        # try alternative access patterns
        try:
            ci_raw = model_output.conf_int
        except Exception:
            ci_raw = None

    if ci_raw is not None:
        # If DataFrame-like
        if isinstance(ci_raw, pd.DataFrame):
            # columns may be [0,1] or ['lower','upper']
            if list(ci_raw.columns)[0] in [0, "lower"]:
                lower_col = ci_raw.columns[0]
                upper_col = ci_raw.columns[1]
            else:
                lower_col, upper_col = ci_raw.columns[:2]
            ci = pd.DataFrame({
                "lower": ci_raw[lower_col],
                "upper": ci_raw[upper_col]
            })
        else:
            # assume numpy array with rows aligned to names
            try:
                ci = pd.DataFrame(ci_raw, index=names, columns=["lower", "upper"])
            except Exception:
                ci = None

    # Variables of interest in order of preference
    vars_of_interest = []
    if "skin_dark" in names:
        vars_of_interest.append("skin_dark")
    if "skin_avg" in names:
        vars_of_interest.append("skin_avg")
    # If none found, try partial matches
    if not vars_of_interest:
        for n in names:
            if "skin" in n:
                vars_of_interest.append(n)
    # Build results dict
    results = {}
    for var in vars_of_interest:
        coef = float(params_s.get(var, np.nan))
        se = float(bse_s.get(var, np.nan))
        pval = float(pvalues_s.get(var, np.nan))
        # CI (on coefficient scale)
        if ci is not None and var in ci.index:
            ci_lower = float(ci.loc[var, "lower"])
            ci_upper = float(ci.loc[var, "upper"])
        else:
            # approximate 95% CI from coef +/- 1.96*se if se available
            if not np.isnan(se):
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                ci_lower = np.nan
                ci_upper = np.nan
        irr = float(np.exp(coef)) if not np.isnan(coef) else np.nan
        irr_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else np.nan
        irr_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else np.nan

        results[var] = {
            "coef": coef,
            "se": se,
            "p_value": pval,
            "coef_95ci": (ci_lower, ci_upper),
            "IRR": irr,
            "IRR_95ci": (irr_ci_lower, irr_ci_upper)
        }

    # Formulate concise interpretation addressing the research question.
    # Primary focus: skin_dark binary indicator (contrast dark vs light). If present, use it.
    conclusion = ""
    alpha = 0.05
    primary_var = "skin_dark" if "skin_dark" in results else (vars_of_interest[0] if vars_of_interest else None)

    if primary_var is None:
        conclusion = "No skin-tone variable (e.g., 'skin_dark' or 'skin_avg') found in the model output."
    else:
        r = results[primary_var]
        coef = r["coef"]
        irr = r["IRR"]
        p = r["p_value"]
        ci = r["IRR_95ci"]
        # Interpret direction and significance
        if np.isnan(p):
            conclusion = (f"Could not determine p-value for {primary_var}; "
                          "numeric summaries returned in 'object'.")
        else:
            sig = (p < alpha)
            dir_text = "higher" if irr > 1 else "lower" if irr < 1 else "no difference"
            if sig:
                conclusion = (
                    f"Statistically significant effect for '{primary_var}' (p = {p:.3g}). "
                    f"Estimated IRR = {irr:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}]), "
                    f"meaning players with higher values on '{primary_var}' receive red cards at a "
                    f"{dir_text} rate per game compared to the reference. "
                )
                # Map to plain-language for skin_dark
                if primary_var == "skin_dark":
                    if irr > 1:
                        conclusion += "This indicates darker-skinned players are more likely to receive red cards."
                    elif irr < 1:
                        conclusion += "This indicates darker-skinned players are less likely to receive red cards."
                    else:
                        conclusion += "No practical difference in red-card rates was detected."
            else:
                conclusion = (
                    f"No statistically significant effect for '{primary_var}' (p = {p:.3g}). "
                    f"Estimated IRR = {irr:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}]). "
                    "This provides no strong evidence that darker-skinned players are more likely to receive red cards."
                )

    return {
        "object": results,
        "description": conclusion
    }