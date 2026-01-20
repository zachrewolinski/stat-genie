def extract_final_answer(model_output):
    """
    Extract statistics for the effect of IsHuman from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with keys:
      - "object": dict of numeric results (coefficient, SE, z, p, CI, odds ratio, OR CI, decision)
      - "description": human-readable interpretation of the IsHuman effect in context

    The function is robust to parameter-name variations that include 'IsHuman' (e.g., 'IsHuman', 'IsHuman[T.True]', etc.).
    """
    import math

    # Helper to raise clear error if model_output doesn't look like a statsmodels results object
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (no .params).")

    params = model_output.params
    bse = None
    pvalues = None
    conf_int = None

    # Try to get standard errors and p-values reliably
    try:
        bse = model_output.bse
    except Exception:
        # fallback: compute from cov_params if available
        if hasattr(model_output, "cov_params"):
            cov = model_output.cov_params()
            # create a copy-like structure for bse
            try:
                bse = params.copy()
                for k in params.index:
                    bse[k] = float((cov.loc[k, k]) ** 0.5)
            except Exception:
                # if params isn't indexable like a pandas Series, build dict
                bse = {}
                for k in params.index:
                    bse[k] = float((cov.loc[k, k]) ** 0.5)
        else:
            raise

    try:
        pvalues = model_output.pvalues
    except Exception:
        # pvalues should be present; if not, compute approx from z
        pvalues = {}
        for k in params.index:
            z = float(params[k]) / float(bse[k])
            # two-sided p-value from normal
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            pvalues[k] = p
        # convert to a pandas-like series if possible
        try:
            import pandas as pd
            pvalues = pd.Series(pvalues)
        except Exception:
            # leave as dict if pandas not available
            pass

    try:
        conf_int = model_output.conf_int()
    except Exception:
        # approximate 95% CI using normal approx
        conf_int = {}
        for k in params.index:
            coef = float(params[k])
            se = float(bse[k])
            lower = coef - 1.96 * se
            upper = coef + 1.96 * se
            conf_int[k] = (lower, upper)
        try:
            import pandas as pd
            conf_int = pd.DataFrame(list(conf_int.values()), index=list(conf_int.keys()), columns=[0, 1])
        except Exception:
            # leave as dict if pandas not available
            pass

    # Identify the parameter name corresponding to IsHuman (be flexible)
    ishuman_keys = [k for k in params.index if "IsHuman" in str(k)]
    if len(ishuman_keys) == 0:
        raise ValueError("No parameter matching 'IsHuman' found in model parameters. Available params: "
                         + ", ".join([str(k) for k in params.index]))

    # If multiple matches, prefer an exact 'IsHuman' name, otherwise pick the first match.
    if "IsHuman" in params.index:
        key = "IsHuman"
    else:
        key = ishuman_keys[0]

    coef = float(params[key])
    # retrieve se robustly
    try:
        se = float(bse[key])
    except Exception:
        # bse might be a dict
        se = float(bse.get(key, float("nan")))

    z_value = coef / se if se != 0 else float("nan")

    # p-value retrieval: robust to dict/Series
    pval = float("nan")
    try:
        # try indexing first (works for pandas Series and dict)
        pval = float(pvalues[key])
    except Exception:
        try:
            # if pvalues has .get (dict-like)
            pval = float(pvalues.get(key, float("nan")))
        except Exception:
            # fallback to model_output.pvalues if available
            try:
                pval = float(model_output.pvalues.get(key, float("nan")))
            except Exception:
                pval = float("nan")

    # Confidence interval: conf_int is a DataFrame-like or dict-like with columns [0,1]
    try:
        # If conf_int is a DataFrame or has .loc
        row = conf_int.loc[key]
        # row may be a Series or list-like
        try:
            ci_low = float(row[0])
            ci_high = float(row[1])
        except Exception:
            # try position-based
            ci_low = float(row.iloc[0])
            ci_high = float(row.iloc[1])
    except Exception:
        # conf_int might be a dict or other structure handled above
        try:
            ci_low, ci_high = conf_int[key]
            ci_low = float(ci_low)
            ci_high = float(ci_high)
        except Exception:
            ci_low, ci_high = (float("nan"), float("nan"))

    # Odds ratio and its CI
    try:
        or_coef = math.exp(coef)
        or_ci_low = math.exp(ci_low)
        or_ci_high = math.exp(ci_high)
    except Exception:
        or_coef = or_ci_low = or_ci_high = float("nan")

    # Decision rule: evidence that humans have higher AMTL if coef > 0 and p < 0.05
    evidence = (coef > 0) and (pval < 0.05)
    if evidence:
        decision_text = ("There is statistically significant evidence (two-sided p = {:.4g}) that modern humans "
                         "(Homo) have higher antemortem tooth loss (AMTL) than the non-human primates in the dataset, "
                         "after controlling for age, sex, and tooth class.").format(pval)
    else:
        # Give more specific reason
        if not (pval < 0.05):
            decision_text = ("There is NOT statistically significant evidence that modern humans have higher AMTL "
                             "(two-sided p = {:.4g}); the coefficient is {:.4g} (SE = {:.4g}).").format(pval, coef, se)
        else:
            # p < 0.05 but coef <= 0
            decision_text = ("Although the IsHuman effect is statistically significant (two-sided p = {:.4g}), "
                             "the coefficient is non-positive ({:.4g}), which does not indicate higher AMTL in humans.").format(pval, coef)

    # Build the object to return (numeric summary)
    result_object = {
        "parameter_name": key,
        "coefficient_log_odds": coef,
        "std_error": se,
        "z_value": z_value,
        "p_value": pval,
        "coef_95CI_log_odds": [ci_low, ci_high],
        "odds_ratio": or_coef,
        "odds_ratio_95CI": [or_ci_low, or_ci_high],
        "decision_higher_AMTL_in_humans": bool(evidence)
    }

    # Human-readable description summarizing the numeric results and decision
    description = (
        f"Model parameter '{key}': coefficient (log-odds) = {coef:.4g}, SE = {se:.4g}, z = {z_value:.4g}, "
        f"two-sided p = {pval:.4g}. 95% CI for coefficient (log-odds): [{ci_low:.4g}, {ci_high:.4g}]. "
        f"Equivalently, odds ratio = {or_coef:.4g} (95% CI: [{or_ci_low:.4g}, {or_ci_high:.4g}]). "
        + decision_text
    )

    return {"object": result_object, "description": description}