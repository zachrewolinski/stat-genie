def extract_final_answer(model_output):
    """
    Extracts statistics for the 'is_human' coefficient from a fitted statsmodels results object
    (this may be a GLMResultsWrapper or a robustified results object returned by
    get_robustcov_results). Returns a dictionary with:
      - "object": a dict of numeric results for the 'is_human' effect
      - "description": a short interpretation in the context of whether humans have higher AMTL

    The numeric results include coefficient (log-odds), SE, test statistic, p-value,
    95% CI on the coefficient scale, odds ratio and its 95% CI (when the model uses a logit link).
    """
    import numpy as np

    # Helper to safely get attributes from the results object
    def _get_attr(res, name):
        return getattr(res, name, None)

    res = model_output

    # params might be a pandas Series or numpy array with index
    params = _get_attr(res, "params")
    if params is None:
        raise ValueError("The provided model_output has no 'params' attribute. "
                         "Provide a statsmodels results object (possibly from get_robustcov_results).")

    # Find the parameter name that corresponds to 'is_human'
    # Look for an exact match first, else any name containing 'is_human'
    try:
        param_index = list(params.index)
    except Exception:
        # params may be a plain ndarray without index (unlikely for statsmodels), handle defensively
        param_index = None

    target_name = None
    if param_index is not None:
        if "is_human" in param_index:
            target_name = "is_human"
        else:
            # find any name containing 'is_human'
            matches = [n for n in param_index if "is_human" in str(n)]
            if len(matches) >= 1:
                # choose the first match
                target_name = matches[0]

    if target_name is None:
        raise ValueError("Could not find a parameter for 'is_human' in model parameters. "
                         f"Available parameter names: {param_index}")

    # Extract coefficient and related statistics
    coef = float(params[target_name])

    # Standard error: try multiple attribute names (.bse or .std_errors)
    bse = None
    if _get_attr(res, "bse") is not None:
        bse_series = res.bse
        try:
            bse = float(bse_series[target_name])
        except Exception:
            # If indexing fails, try converting to array by position
            try:
                pos = list(params.index).index(target_name)
                bse = float(np.asarray(bse_series)[pos])
            except Exception:
                bse = None
    if bse is None and _get_attr(res, "std_errors") is not None:
        try:
            std_errs = res.std_errors
            bse = float(std_errs[target_name])
        except Exception:
            bse = None

    # Test statistic and p-value
    zstat = None
    pvalue = None
    if _get_attr(res, "tvalues") is not None:
        try:
            zstat = float(res.tvalues[target_name])
        except Exception:
            pass
    if zstat is None and _get_attr(res, "tvalue") is not None:
        try:
            zstat = float(res.tvalue[target_name])
        except Exception:
            pass
    if _get_attr(res, "pvalues") is not None:
        try:
            pvalue = float(res.pvalues[target_name])
        except Exception:
            pass

    # 95% confidence interval for coefficient
    ci_lower = ci_upper = None
    try:
        ci = res.conf_int()
        # conf_int() typically returns a DataFrame/ndarray with rows indexed by param names
        # Attempt to index by the target name
        try:
            ci_row = ci.loc[target_name]
            ci_lower = float(ci_row[0])
            ci_upper = float(ci_row[1])
        except Exception:
            # fallback: use positional index
            pos = list(params.index).index(target_name)
            ci_row = np.asarray(ci)[pos]
            ci_lower = float(ci_row[0])
            ci_upper = float(ci_row[1])
    except Exception:
        # If conf_int not available, approximate from coef +/- 1.96*SE if SE available
        if bse is not None:
            ci_lower = float(coef - 1.96 * bse)
            ci_upper = float(coef + 1.96 * bse)

    # Determine model link function name if available (to decide whether OR is meaningful)
    link_name = None
    try:
        link_name = res.model.family.link.__class__.__name__.lower()
    except Exception:
        try:
            # sometimes name attribute exists
            link_name = res.model.family.link.name.lower()
        except Exception:
            link_name = None

    # If link appears to be logit, compute odds ratio and CI on OR scale
    odds_ratio = or_ci_lower = or_ci_upper = None
    try:
        if link_name is not None and "logit" in link_name:
            odds_ratio = float(np.exp(coef))
            if (ci_lower is not None) and (ci_upper is not None):
                or_ci_lower = float(np.exp(ci_lower))
                or_ci_upper = float(np.exp(ci_upper))
        else:
            # If link unknown, attempt exp transform but flag that interpretation may be inappropriate
            if link_name is None:
                # still compute exp for interpretability, but note it in description
                odds_ratio = float(np.exp(coef))
                if (ci_lower is not None) and (ci_upper is not None):
                    or_ci_lower = float(np.exp(ci_lower))
                    or_ci_upper = float(np.exp(ci_upper))
    except Exception:
        odds_ratio = or_ci_lower = or_ci_upper = None

    # Determine significance at alpha = 0.05 if p-value available
    significant = None
    if pvalue is not None:
        significant = bool(pvalue < 0.05)

    # Build the object to return
    result_object = {
        "parameter_name": target_name,
        "coef_log_odds": coef,
        "std_error": bse,
        "test_statistic": zstat,
        "p_value": pvalue,
        "ci_95_coef": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "ci_95_odds_ratio": [or_ci_lower, or_ci_upper],
        "link_name": link_name,
        "significant_at_0.05": significant
    }

    # Build a concise description / interpretation
    if (pvalue is not None) and (coef is not None):
        direction = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
        if link_name is not None and "logit" in link_name:
            if significant is True:
                interp = (f"The model coefficient for '{target_name}' is {coef:.3f} (SE={bse:.3f}, p={pvalue:.3g}), "
                         f"which corresponds to an odds ratio of {odds_ratio:.3f} (95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}]). "
                         f"This indicates that, after adjusting for age, prob_male, and tooth class, "
                         f"modern humans have {direction} odds of AMTL compared to non-human primates (p < 0.05).")
            else:
                interp = (f"The model coefficient for '{target_name}' is {coef:.3f} (SE={bse:.3f}, p={pvalue:.3g}), "
                         f"odds ratio {odds_ratio:.3f} (95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}]). "
                         f"The effect is not statistically significant at α=0.05, so we cannot conclude a difference in AMTL between modern humans and non-human primates after adjustment.")
        else:
            # Non-logit link or unknown link
            if significant is True:
                interp = (f"The coefficient for '{target_name}' is {coef:.3f} (SE={bse:.3f}, p={pvalue:.3g}), "
                         f"95% CI for the coefficient: [{ci_lower:.3f}, {ci_upper:.3f}]. "
                         f"This indicates a statistically significant difference in the modeled scale; positive coef means {direction} modeled AMTL for humans. "
                         f"Odds ratio (exp(coef)) = {odds_ratio:.3f} provided for rough interpretation (link = {link_name}).")
            else:
                interp = (f"The coefficient for '{target_name}' is {coef:.3f} (SE={bse:.3f}, p={pvalue:.3g}), "
                         f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]. "
                         f"The effect is not statistically significant at α=0.05. Odds ratio (exp(coef)) = {odds_ratio:.3f} (link = {link_name}) may be computed but interpret cautiously.")
    else:
        interp = ("Could not fully extract p-value or coefficient details for 'is_human'. "
                  "Returned numeric values (where available) in the 'object' field for inspection.")

    return {"object": result_object, "description": interp}