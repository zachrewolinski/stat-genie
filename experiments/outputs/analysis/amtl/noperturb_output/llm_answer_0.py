import numpy as np

def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, test statistic, p-value, 95% CI, and odds ratio
    (with CI) for the 'IsHuman' predictor from a statsmodels result-like object (e.g., RobustResults).
    Returns a dictionary with keys:
      - "object": dict of numeric results and a boolean 'humans_higher' indicating whether the
                  result supports that Homo sapiens have higher AMTL (coef > 0) and 'significant'
                  indicating p < 0.05.
      - "description": short interpretation of the extracted statistics in the study context.
    """
    res = model_output
    var = "IsHuman"

    # Basic sanity checks
    if not hasattr(res, "params"):
        raise AttributeError("model_output does not have 'params' attribute; expected a statsmodels result object.")

    # Build parameter index list for name->position lookups
    try:
        params_index = list(res.params.index)
    except Exception:
        # params is not indexable by name; fallback to sequence of positions
        try:
            params_index = list(range(len(res.params)))
        except Exception:
            raise RuntimeError("Unable to determine parameter names/positions from model_output.params.")

    if var not in params_index:
        raise KeyError(f"'{var}' not found in model coefficients. Available vars: {params_index}")

    # Helper to safely get element by name or by position from various container types
    def _safe_get(container, name, params_index):
        """
        Try container[name]; if that fails, find position of name in params_index and
        return container[pos]. Works for pandas Series/DataFrame, numpy arrays, lists.
        """
        # Direct name-based access (works for pandas Series/DataFrame or dict)
        try:
            return container[name]
        except Exception:
            pass

        # Position-based access
        try:
            pos = params_index.index(name)
        except ValueError:
            raise KeyError(f"'{name}' not found in params index {params_index}")

        arr = np.asarray(container)
        # If 1D, return arr[pos]; if 2D and pos is row index, return arr[pos]
        try:
            return arr[pos]
        except Exception as e:
            raise RuntimeError(f"Could not index into container to retrieve '{name}': {e}")

    # Extract coefficient
    try:
        coef = float(_safe_get(res.params, var, params_index))
    except Exception as e:
        raise RuntimeError(f"Could not extract coefficient for '{var}': {e}")

    # Extract standard error (bse) with fallbacks
    se = None
    if hasattr(res, "bse"):
        try:
            se_candidate = _safe_get(res.bse, var, params_index)
            se = float(se_candidate)
        except Exception:
            se = None

    # If se still None, try to derive from conf_int
    if se is None:
        try:
            ci_mat = res.conf_int()
            # try name-based access first
            try:
                ci_row = ci_mat.loc[var]
                ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
            except Exception:
                # fallback by position
                pos = params_index.index(var)
                ci_arr = np.asarray(ci_mat)
                ci_lower, ci_upper = float(ci_arr[pos, 0]), float(ci_arr[pos, 1])
            se = (ci_upper - ci_lower) / (2 * 1.96)
        except Exception:
            raise RuntimeError("Could not extract standard errors or confidence intervals from model_output.")

    # Test statistic (z/t): compute as coef / se
    try:
        stat = float(coef / se) if se != 0 else float("nan")
    except Exception:
        stat = float("nan")

    # p-value extraction
    pval = float("nan")
    if hasattr(res, "pvalues"):
        try:
            pval_candidate = _safe_get(res.pvalues, var, params_index)
            pval = float(pval_candidate)
        except Exception:
            pval = float("nan")

    # Confidence interval (95%)
    try:
        ci_mat = res.conf_int()
        try:
            ci_row = ci_mat.loc[var]
            ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        except Exception:
            pos = params_index.index(var)
            ci_arr = np.asarray(ci_mat)
            ci_lower, ci_upper = float(ci_arr[pos, 0]), float(ci_arr[pos, 1])
    except Exception:
        # fallback using se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Odds ratio and its CI (since model is binomial-logit)
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Decision: do humans have higher AMTL?
    humans_higher_direction = coef > 0
    significant = (not np.isnan(pval)) and (pval < 0.05)
    humans_higher = humans_higher_direction and significant

    # Build return object
    result_obj = {
        "variable": var,
        "coefficient_log_odds": coef,
        "std_error": se,
        "statistic": stat,
        "p_value": pval,
        "ci_95_log_odds": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_95": [or_ci_lower, or_ci_upper],
        "humans_higher_direction": humans_higher_direction,
        "significant_at_0.05": significant,
        "humans_higher": humans_higher  # True only if coef>0 AND p<0.05
    }

    # Short interpretation
    if np.isnan(pval):
        significance_text = "p-value not available"
    else:
        significance_text = ("statistically significant (p < 0.05)"
                             if significant else "not statistically significant (p >= 0.05)")

    direction_text = ("positive coefficient (higher log-odds of AMTL for Homo sapiens)"
                      if humans_higher_direction else
                      "negative coefficient (lower log-odds of AMTL for Homo sapiens)")

    description = (
        f"IsHuman coefficient = {coef:.4f} (SE = {se:.4f}, stat = {stat:.3f}, p = {pval:.4g}). "
        f"95% CI on log-odds: [{ci_lower:.4f}, {ci_upper:.4f}]. Odds ratio = {odds_ratio:.3f} "
        f"95% CI: [{or_ci_lower:.3f}, {or_ci_upper:.3f}].\n"
        f"Interpretation: {direction_text}; {significance_text}. "
        f"In plain terms, a positive and significant coefficient means modern humans have a higher "
        f"frequency of AMTL than non-human primates after adjusting for age, sex (prob_male), tooth class, "
        f"and population. The 'humans_higher' boolean in the 'object' indicates whether both direction and "
        f"statistical significance support that conclusion."
    )

    return {"object": result_obj, "description": description}