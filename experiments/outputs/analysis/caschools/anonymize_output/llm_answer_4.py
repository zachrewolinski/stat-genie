def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a statsmodels OLSResults
    (or robustcov results) object and returns a structured answer.

    Returns:
      {
        "object": {
          "coef": float or None,
          "std_err": float or None,
          "t_value": float or None,
          "p_value": float or None,
          "ci_lower": float or None,
          "ci_upper": float or None,
          "significant_0.05": bool or None,
          "conclusion": str   # short yes/no style conclusion about whether LOWER ratio -> HIGHER performance
        },
        "description": str    # human-readable interpretation in context
      }
    """
    import numpy as np

    res = model_output
    varname = "StudentTeacherRatio"

    # Try to get core attributes; raise if model_output is clearly incompatible
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        tvalues = getattr(res, "tvalues", None)
        conf = res.conf_int()
    except Exception as e:
        raise ValueError(
            "model_output does not appear to be a statsmodels results object with expected attributes."
        ) from e

    # Helper: normalize variable names for fuzzy matching
    def _normalize(name):
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    # Determine available parameter names (if any)
    params_index = None
    try:
        # pandas Index-like
        params_index = list(params.index)
    except Exception:
        # If params is array-like without names, we cannot look up by variable name
        params_index = None

    matched_name = None
    matched_idx = None

    if params_index:
        # Try exact match first
        if varname in params_index:
            matched_name = varname
            matched_idx = params_index.index(matched_name)
        else:
            # Build normalized mapping
            norm_map = { _normalize(n): n for n in params_index }
            target_norm = _normalize(varname)
            if target_norm in norm_map:
                matched_name = norm_map[target_norm]
                matched_idx = params_index.index(matched_name)
            else:
                # Try substring-based heuristics: look for names containing both 'student' and 'teacher'
                candidates = []
                for n in params_index:
                    nl = n.lower()
                    if ("student" in nl and "teacher" in nl) or ("stu" in nl and "teach" in nl):
                        candidates.append(n)
                if len(candidates) == 1:
                    matched_name = candidates[0]
                    matched_idx = params_index.index(matched_name)
                elif len(candidates) > 1:
                    # Prefer exact token match if present
                    for c in candidates:
                        if _normalize(c) == target_norm:
                            matched_name = c
                            matched_idx = params_index.index(matched_name)
                            break
                    # Otherwise take first candidate
                    if matched_name is None:
                        matched_name = candidates[0]
                        matched_idx = params_index.index(matched_name)
                else:
                    # Try more permissive: any name containing 'student' OR 'teacher'
                    candidates2 = [n for n in params_index if ("student" in n.lower() or "teacher" in n.lower())]
                    if len(candidates2) == 1:
                        matched_name = candidates2[0]
                        matched_idx = params_index.index(matched_name)

    # If we couldn't map a name, return a clear result describing the problem
    if matched_name is None:
        available = params_index if params_index is not None else "<unnamed parameter array>"
        description = (
            f"Variable '{varname}' not found in model results. Available parameter names: {available}. "
            "No statistics for StudentTeacherRatio could be extracted."
        )
        result_object = {
            "coef": None,
            "std_err": None,
            "t_value": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant_0.05": None,
            "conclusion": "Variable not found"
        }
        return {"object": result_object, "description": description}

    # Extract numeric values using the matched_name / matched_idx
    try:
        # params, bse, pvalues might be pandas Series or numpy arrays
        def _get_by_name_or_idx(obj, name, idx):
            # Try by label first
            try:
                return obj[name]
            except Exception:
                try:
                    return obj.iloc[idx]
                except Exception:
                    try:
                        return np.asarray(obj)[idx]
                    except Exception:
                        raise KeyError(f"Could not extract entry for '{name}' from object of type {type(obj)}")

        coef = float(_get_by_name_or_idx(params, matched_name, matched_idx))
        std_err = float(_get_by_name_or_idx(bse, matched_name, matched_idx))
        p_value = float(_get_by_name_or_idx(pvalues, matched_name, matched_idx))
        t_value = None
        if tvalues is not None:
            try:
                t_value = float(_get_by_name_or_idx(tvalues, matched_name, matched_idx))
            except Exception:
                t_value = None

        # Confidence interval: conf may be DataFrame-like with index labels or a plain ndarray
        try:
            # Try DataFrame/array with .loc
            try:
                ci_row = conf.loc[matched_name].values
            except Exception:
                # positional fallback
                ci_arr = np.asarray(conf)
                ci_row = ci_arr[matched_idx]
        except Exception:
            raise KeyError(f"Could not extract confidence interval row for '{matched_name}'")
        ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
    except KeyError as ke:
        # If any extraction fails unexpectedly, return a clear message
        description = (
            f"Failed to extract full statistics for matched variable '{matched_name}': {ke}. "
            "This may indicate an unexpected structure in the model result object."
        )
        result_object = {
            "coef": None,
            "std_err": None,
            "t_value": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "significant_0.05": None,
            "conclusion": "Extraction error"
        }
        return {"object": result_object, "description": description}

    # Statistical significance at alpha=0.05
    significant_0_05 = (p_value < 0.05)

    # Prepare human-readable description and conclusion
    if coef < 0 and significant_0_05:
        conclusion = "Yes"
        desc = (
            f"Yes — coefficient on {matched_name} = {coef:.4f} (SE = {std_err:.4f}, "
            f"t = {t_value:.3f} , p = {p_value:.3f}), indicating that a one-unit increase in "
            f"students-per-teacher is associated with a {abs(coef):.4f}-point decrease in AvgScore. "
            f"This effect is statistically significant at the 5% level. 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]."
        )
    elif coef < 0 and not significant_0_05:
        conclusion = "No (directionally consistent but not statistically significant)"
        desc = (
            f"The estimated coefficient on {matched_name} = {coef:.4f} (SE = {std_err:.4f}, "
            f"p = {p_value:.3f}) is negative, which would imply that lower student-teacher ratios "
            f"(fewer students per teacher) are associated with higher AvgScore, but this estimate "
            f"is not statistically significant at the 5% level. 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]."
        )
    elif coef > 0 and significant_0_05:
        conclusion = "No (evidence in the opposite direction)"
        desc = (
            f"No — coefficient on {matched_name} = {coef:.4f} (SE = {std_err:.4f}, "
            f"t = {t_value:.3f} , p = {p_value:.3f}), indicating that a one-unit increase in "
            f"students-per-teacher is associated with a {coef:.4f}-point increase in AvgScore. "
            f"This is statistically significant at the 5% level. 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]."
        )
    elif coef > 0 and not significant_0_05:
        conclusion = "No (no evidence of association)"
        desc = (
            f"The estimated coefficient on {matched_name} = {coef:.4f} (SE = {std_err:.4f}, "
            f"p = {p_value:.3f}) is positive but not statistically significant at the 5% level. "
            f"Thus there is no reliable evidence that student-teacher ratio is associated with AvgScore. "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]."
        )
    else:  # coef == 0 (very unlikely exactly zero)
        conclusion = "No (no association)"
        desc = (
            f"The estimated coefficient on {matched_name} is essentially zero (coef = {coef:.4f}), "
            f"with p = {p_value:.3f} and 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}], providing no evidence "
            f"of an association."
        )

    result_object = {
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_0.05": significant_0_05,
        "conclusion": conclusion
    }

    return {"object": result_object, "description": desc}