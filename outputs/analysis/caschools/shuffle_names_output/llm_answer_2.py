import re

def extract_final_answer(model_output):
    """
    Extracts the coefficient, p-value, 95% CI, and related interpretive info
    for the StudentTeacherRatio variable from a statsmodels OLSResults-like object
    (including robustcov results).

    Returns a dict with keys:
      - "object": dict containing numerical results:
           {
             "coef": float or None,                # coefficient on StudentTeacherRatio
             "p_value": float or None,             # two-sided p-value
             "ci_lower": float or None,            # 95% CI lower bound
             "ci_upper": float or None,            # 95% CI upper bound
             "coef_per_10_students": float or None,# effect for a 10-student increase
             "n_obs": int or None,                 # number of observations used in the model
             "significant_0.05": bool or None      # whether p_value < 0.05
           }
      - "description": str explaining what the numbers mean in context.
    The function is defensive: if the StudentTeacherRatio parameter cannot be found,
    it returns a descriptive message and None values rather than raising.
    """

    def _normalize(name):
        return re.sub(r'[^a-z0-9]', '', str(name).lower())

    def _param_index_candidates(res):
        """
        Yield a list-like of parameter names/labels to search through in order.
        """
        candidates = []
        if hasattr(res, 'params'):
            try:
                idx = res.params.index
                # convert to list (pandas Index or similar)
                candidates.extend(list(idx))
            except Exception:
                # params exists but not indexable; ignore
                pass
        # Try model.exog_names if available (statsmodels)
        if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
            try:
                candidates.extend(list(res.model.exog_names))
            except Exception:
                pass
        # dedupe while preserving order
        seen = set()
        uniq = []
        for x in candidates:
            key = str(x)
            if key not in seen:
                uniq.append(x)
                seen.add(key)
        return uniq

    def _resolve_name(res, desired_name):
        """
        Try to find the best matching parameter name in the result object for desired_name.
        Matching strategy:
         - exact normalized match
         - normalized contains both 'student' and 'teacher'
         - normalized contains either 'student' or 'teacher'
         - otherwise None
        """
        desired_norm = _normalize(desired_name)
        candidates = _param_index_candidates(res)
        # exact normalized match
        for name in candidates:
            if _normalize(name) == desired_norm:
                return name
        # both words present
        for name in candidates:
            n = _normalize(name)
            if 'student' in n and 'teacher' in n:
                return name
        # either word present
        for name in candidates:
            n = _normalize(name)
            if 'student' in n or 'teacher' in n:
                return name
        return None

    def _get_param(res, name):
        resolved = _resolve_name(res, name)
        if resolved is None:
            raise KeyError(f"Could not find parameter matching '{name}' in model output.")
        # access value
        if hasattr(res, 'params'):
            try:
                return res.params[resolved]
            except Exception:
                # try position-based lookup if params is array-like and we have exog_names
                try:
                    if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                        exog = list(res.model.exog_names)
                        if resolved in exog:
                            idx = exog.index(resolved)
                            return res.params[idx]
                except Exception:
                    pass
        # fallback: direct attribute
        try:
            return getattr(res, resolved)
        except Exception:
            raise KeyError(f"Could not retrieve parameter value for '{resolved}'.")

    def _get_pvalue(res, name):
        resolved = _resolve_name(res, name)
        if resolved is None:
            raise KeyError(f"Could not find p-value matching '{name}' in model output.")
        if hasattr(res, 'pvalues'):
            try:
                return res.pvalues[resolved]
            except Exception:
                try:
                    # attempt positional lookup as fallback
                    if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                        exog = list(res.model.exog_names)
                        if resolved in exog:
                            idx = exog.index(resolved)
                            return res.pvalues[idx]
                except Exception:
                    pass
        raise KeyError(f"Could not retrieve p-value for '{resolved}'.")

    def _get_conf_int(res, name, alpha=0.05):
        resolved = _resolve_name(res, name)
        if resolved is None:
            raise KeyError(f"Could not find confidence interval matching '{name}' in model output.")
        if hasattr(res, 'conf_int'):
            try:
                ci = res.conf_int(alpha=alpha)
                # Most commonly returns a DataFrame or 2D ndarray
                # Try DataFrame-like access first
                try:
                    # If ci is DataFrame-like and supports .loc[name]
                    lower = float(ci.loc[resolved][0])
                    upper = float(ci.loc[resolved][1])
                    return lower, upper
                except Exception:
                    pass
                # If ci is array-like (ndarray) try to find index of resolved in params/exog_names
                try:
                    # find index
                    if hasattr(res, 'params') and resolved in list(res.params.index):
                        idx = list(res.params.index).index(resolved)
                        lower = float(ci[idx, 0])
                        upper = float(ci[idx, 1])
                        return lower, upper
                    if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                        exog = list(res.model.exog_names)
                        if resolved in exog:
                            idx = exog.index(resolved)
                            lower = float(ci[idx, 0])
                            upper = float(ci[idx, 1])
                            return lower, upper
                except Exception:
                    pass
                # As a last resort, if ci is 2D and has same order as params, try to map by order
                try:
                    if hasattr(res, 'params'):
                        idx = 0
                        # attempt to detect index position
                        try:
                            idx = list(res.params.index).index(resolved)
                            lower = float(ci[idx][0])
                            upper = float(ci[idx][1])
                            return lower, upper
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
        raise KeyError(f"Could not obtain confidence interval for '{resolved}'.")

    var_name = 'StudentTeacherRatio'
    # Prepare default return in case of failure
    result_obj = {
        "coef": None,
        "p_value": None,
        "ci_lower": None,
        "ci_upper": None,
        "coef_per_10_students": None,
        "n_obs": None,
        "significant_0.05": None,
        "variable": var_name
    }
    description = ""

    try:
        coef_raw = _get_param(model_output, var_name)
        coef = float(coef_raw)
        pval_raw = _get_pvalue(model_output, var_name)
        pval = float(pval_raw)
        ci_lower, ci_upper = _get_conf_int(model_output, var_name, alpha=0.05)
    except KeyError as e:
        # Gracefully return with explanatory description rather than raising
        description = (
            f"Could not extract the StudentTeacherRatio estimates from the provided model output. "
            f"Detail: {e}. The returned 'object' fields are None where extraction failed."
        )
        return {"object": result_obj, "description": description}

    # Number of observations if available
    n_obs = None
    if hasattr(model_output, 'nobs'):
        try:
            n_obs = int(model_output.nobs)
        except Exception:
            try:
                df_model = getattr(model_output, 'df_model', None)
                df_resid = getattr(model_output, 'df_resid', None)
                if df_model is not None and df_resid is not None:
                    n_obs = int(df_model + df_resid + 1)
            except Exception:
                n_obs = None

    coef_per_10 = coef * 10.0
    significant = (pval < 0.05)

    # Interpret direction: StudentTeacherRatio higher => more students per teacher.
    if significant:
        if coef < 0:
            direction = ("Statistically significant (p < 0.05). "
                         "The negative coefficient indicates that lower student-teacher ratios "
                         "(fewer students per teacher) are associated with higher district average test scores.")
        else:
            direction = ("Statistically significant (p < 0.05). "
                         "The positive coefficient indicates that higher student-teacher ratios "
                         "(more students per teacher) are associated with higher district average test scores.")
    else:
        if coef < 0:
            direction = ("Not statistically significant at the 5% level. "
                         "Point estimate is negative (lower ratio linked to higher scores), "
                         "but we cannot reject no effect.")
        else:
            direction = ("Not statistically significant at the 5% level. "
                         "Point estimate is positive (higher ratio linked to higher scores), "
                         "but we cannot reject no effect.")

    description = (
        f"Coefficient on StudentTeacherRatio = {coef:.4f}; two-sided p-value = {pval:.4g}; "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}].\n"
        f"Interpretation: A one-unit increase in StudentTeacherRatio (one more student per teacher) is associated "
        f"with a change of {coef:.4f} points in AvgTestScore, holding expenditure per student, percent free/reduced lunch, "
        f"percent English learners, and log enrollment constant. For a 10-student increase, the estimated change is "
        f"{coef_per_10:.4f} points. {direction} "
    )
    if n_obs is not None:
        description += f"Results are based on n = {n_obs} observations."

    result_obj.update({
        "coef": float(coef),
        "p_value": float(pval),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "coef_per_10_students": float(coef_per_10),
        "n_obs": n_obs,
        "significant_0.05": bool(significant),
        "variable": var_name
    })

    return {
        "object": result_obj,
        "description": description
    }