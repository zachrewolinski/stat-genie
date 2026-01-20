def extract_final_answer(model_output):
    """
    Extracts the estimated effect of the 'female' indicator from a fitted statsmodels
    binary model result (e.g., Logit/GLM result or a robust-covariance wrapper).
    Returns a dictionary with:
      - "object": a dict of numeric results (coefficient, SE, test stat, p-value,
                  95% CI on log-odds, odds ratio and its 95% CI, significance flag, nobs)
      - "description": a short interpretation of what these numbers mean for the task.

    The function is robust to different parameter naming conventions (e.g., 'female',
    'Female', 'female[T.True]', 'C(female)[T.1]', 'sex', 'gender', etc.) and will
    attempt to find a parameter whose name contains any of a set of keyword candidates.
    If the target parameter cannot be located or some statistics cannot be extracted,
    the function returns a best-effort "object" with None for missing numeric entries
    and an explanatory "description" instead of raising an error.
    """
    import math

    res = model_output

    # Helper to obtain list of parameter names
    def _get_param_names(res_obj):
        # Try params index first
        try:
            params = getattr(res_obj, "params", None)
            if params is not None:
                # pandas Series / Index-like
                if hasattr(params, "index"):
                    return list(params.index)
                # statsmodels sometimes has params as numpy array and model.exog_names present
        except Exception:
            pass
        # Fallback to model.exog_names
        try:
            mn = getattr(res_obj, "model", None)
            if mn is not None and hasattr(mn, "exog_names"):
                return list(mn.exog_names)
        except Exception:
            pass
        # As last resort try to inspect params if it's dict-like
        try:
            params = getattr(res_obj, "params", None)
            if isinstance(params, dict):
                return list(params.keys())
        except Exception:
            pass
        return []

    param_names = _get_param_names(res)

    # Try to find a parameter corresponding to "female" or synonyms
    def _find_female_param(names):
        if not names:
            return None
        keywords_priority = [
            ["female"],
            ["sex"],
            ["gender"],
            ["woman", "women", "female"],
            ["male"],  # include male as last resort (we will note interpretation)
        ]
        # Normalize name strings to compare
        names_str = [str(n) for n in names]
        # Exact case-insensitive match first
        for n in names_str:
            if n.lower() == "female":
                return n
        # Then multi-stage search by priority groups
        for group in keywords_priority:
            for target in group:
                for n in names_str:
                    if target in n.lower():
                        return n
        # No match found
        return None

    female_name = _find_female_param(param_names)

    # Helper to safely extract a numeric value (by name or by positional index)
    def _get_numeric(obj, name, names_list):
        if obj is None or name is None:
            return None
        # If it's a pandas Series / has index with name
        try:
            if hasattr(obj, "index") and name in obj.index:
                return float(obj[name])
        except Exception:
            pass
        # dict-like
        try:
            if isinstance(obj, dict) and name in obj:
                return float(obj[name])
        except Exception:
            pass
        # positional lookup using names_list
        try:
            if names_list:
                pos = list(names_list).index(name)
                return float(obj[pos])
        except Exception:
            pass
        # DataFrame-like with columns/loc
        try:
            if hasattr(obj, "loc") and name in getattr(obj, "index", []):
                return float(obj.loc[name])
        except Exception:
            pass
        return None

    # If we couldn't locate a plausible parameter name, return explanatory result instead of raising
    if female_name is None:
        description = (
            "Could not locate a model parameter that looks like 'female', 'sex', or 'gender'. "
            f"Available parameter names were: {param_names!r}. "
            "Please ensure the fitted model includes a binary indicator for sex/gender or pass a model "
            "object with accessible parameter names."
        )
        result_object = {
            "parameter_name": None,
            "coef_log_odds": None,
            "std_error": None,
            "test_statistic": None,
            "p_value": None,
            "ci_log_odds_95pct": [None, None],
            "odds_ratio": None,
            "odds_ratio_ci_95pct": [None, None],
            "significant_at_0.05": None,
            "n_observations": None,
        }
        return {"object": result_object, "description": description}

    # Coefficient (log-odds)
    if not hasattr(res, "params"):
        description = "The provided model_output does not have a 'params' attribute."
        result_object = {
            "parameter_name": str(female_name),
            "coef_log_odds": None,
            "std_error": None,
            "test_statistic": None,
            "p_value": None,
            "ci_log_odds_95pct": [None, None],
            "odds_ratio": None,
            "odds_ratio_ci_95pct": [None, None],
            "significant_at_0.05": None,
            "n_observations": None,
        }
        return {"object": result_object, "description": description}

    coef = _get_numeric(res.params, female_name, param_names)
    if coef is None:
        description = (
            f"Could not extract the coefficient for parameter '{female_name}'. "
            f"Available parameters: {param_names!r}."
        )
        result_object = {
            "parameter_name": str(female_name),
            "coef_log_odds": None,
            "std_error": None,
            "test_statistic": None,
            "p_value": None,
            "ci_log_odds_95pct": [None, None],
            "odds_ratio": None,
            "odds_ratio_ci_95pct": [None, None],
            "significant_at_0.05": None,
            "n_observations": None,
        }
        return {"object": result_object, "description": description}

    # Standard error
    bse = None
    if hasattr(res, "bse"):
        bse = _get_numeric(res.bse, female_name, param_names)
    if bse is None:
        # try to compute from cov_params if available: se = sqrt(diag(cov))[pos]
        try:
            cov_func = getattr(res, "cov_params", None)
            if cov_func is not None:
                cov = cov_func()
                try:
                    if hasattr(cov, "loc"):
                        var = float(cov.loc[female_name, female_name])
                    else:
                        pos = list(param_names).index(female_name)
                        var = float(cov[pos, pos])
                    bse = float(math.sqrt(var))
                except Exception:
                    bse = None
        except Exception:
            bse = None
    # If still None, proceed but note in description later

    # Test statistic: try tvalues/zvalues or compute coef / se
    test_stat = None
    for attr in ("tvalues", "zvalues", "tvalue", "zvalue"):
        if hasattr(res, attr):
            arr = getattr(res, attr)
            ts = _get_numeric(arr, female_name, param_names)
            if ts is not None:
                test_stat = float(ts)
                break
    if test_stat is None and bse not in (None, 0):
        try:
            test_stat = float(coef) / float(bse)
        except Exception:
            test_stat = None

    # p-value
    pval = None
    if hasattr(res, "pvalues"):
        pval = _get_numeric(res.pvalues, female_name, param_names)
    if pval is None and test_stat is not None:
        # Compute two-sided p-value from normal approximation without requiring scipy
        try:
            z = abs(float(test_stat))
            # Phi(z) = 0.5*(1 + erf(z/sqrt(2)))
            phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            pval = float(2.0 * (1.0 - phi))
        except Exception:
            pval = None

    # 95% confidence interval on coefficient (log-odds)
    lower = None
    upper = None
    if hasattr(res, "conf_int"):
        try:
            ci = res.conf_int()
            if hasattr(ci, "loc"):
                # DataFrame-like: columns 0 and 1
                lower = float(ci.loc[female_name, 0])
                upper = float(ci.loc[female_name, 1])
            else:
                pos = list(param_names).index(female_name)
                lower = float(ci[pos, 0])
                upper = float(ci[pos, 1])
        except Exception:
            lower = None
            upper = None
    if (lower is None or upper is None) and bse is not None:
        try:
            lower = float(coef) - 1.96 * float(bse)
            upper = float(coef) + 1.96 * float(bse)
        except Exception:
            lower = None
            upper = None

    # Odds ratio and its CI (only if coef available)
    odds_ratio = None
    odds_ratio_ci_lower = None
    odds_ratio_ci_upper = None
    try:
        if coef is not None:
            odds_ratio = float(math.exp(coef))
        if lower is not None:
            odds_ratio_ci_lower = float(math.exp(lower))
        if upper is not None:
            odds_ratio_ci_upper = float(math.exp(upper))
    except Exception:
        odds_ratio = odds_ratio_ci_lower = odds_ratio_ci_upper = None

    # Significance flag at alpha=0.05
    significant = None
    try:
        if pval is not None:
            significant = bool(pval < 0.05)
    except Exception:
        significant = None

    # Sample size if available
    nobs = None
    try:
        if hasattr(res, "nobs"):
            nobs = int(res.nobs)
    except Exception:
        nobs = None
    if nobs is None:
        try:
            mn = getattr(res, "model", None)
            if mn is not None and hasattr(mn, "nobs"):
                nobs = int(mn.nobs)
        except Exception:
            nobs = None

    # Build result object with safe casts
    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    result_object = {
        "parameter_name": str(female_name),
        "coef_log_odds": _safe_float(coef),
        "std_error": _safe_float(bse),
        "test_statistic": _safe_float(test_stat),
        "p_value": _safe_float(pval),
        "ci_log_odds_95pct": [_safe_float(lower), _safe_float(upper)],
        "odds_ratio": _safe_float(odds_ratio),
        "odds_ratio_ci_95pct": [_safe_float(odds_ratio_ci_lower), _safe_float(odds_ratio_ci_upper)],
        "significant_at_0.05": (None if significant is None else bool(significant)),
        "n_observations": (None if nobs is None else int(nobs)),
    }

    # Short human-readable description
    if coef is None:
        description = (
            f"Parameter '{female_name}' was located but the coefficient could not be extracted. "
            f"Result object contains available fields; missing values indicate statistics that could not be derived."
        )
    else:
        # Compose informative description depending on what's available
        coef_val = result_object["coef_log_odds"]
        pval_val = result_object["p_value"]
        or_val = result_object["odds_ratio"]
        ci_or = result_object["odds_ratio_ci_95pct"]
        ci_log = result_object["ci_log_odds_95pct"]
        se_val = result_object["std_error"]

        if pval_val is not None and significant is not None and significant:
            direction = "higher" if coef_val > 0 else "lower"
            # approximate percent change in odds
            try:
                if coef_val > 0:
                    pct_change = (or_val - 1.0) * 100.0
                else:
                    pct_change = (1.0 - or_val) * 100.0
            except Exception:
                pct_change = None
            pct_str = f"approximately {pct_change:.1f}% " if pct_change is not None else ""
            description = (
                f"The model estimates that '{female_name}' is associated with a statistically significant change "
                f"in the odds of the outcome (coef={coef_val:.4f}, p={pval_val:.3g}). "
                f"Odds ratio = {or_val:.3f} (95% CI: {ci_or[0]:.3f} to {ci_or[1]:.3f}) — the group coded by '{female_name}' has "
                f"{pct_str}{direction} odds compared with the reference, controlling for covariates. "
                f"(Log-odds 95% CI: [{ci_log[0]:.4f}, {ci_log[1]:.4f}].)"
            )
        else:
            # Non-significant or lacking p-value
            if pval_val is None:
                p_part = "p-value could not be determined"
            else:
                p_part = f"p={pval_val:.3g} (not significant at 0.05)"
            se_part = f" standard error={se_val:.4f}," if se_val is not None else ""
            description = (
                f"The model estimates an association between '{female_name}' and the outcome (coef={coef_val:.4f}), "
                f"but {p_part}. The odds ratio is {('NA' if or_val is None else f'{or_val:.3f}')} "
                f"with 95% CI [{('NA' if ci_or[0] is None else f'{ci_or[0]:.3f}')}, {('NA' if ci_or[1] is None else f'{ci_or[1]:.3f}')}]. "
                f"{se_part} sample size: {nobs if nobs is not None else 'unknown'}."
            )

    return {"object": result_object, "description": description}