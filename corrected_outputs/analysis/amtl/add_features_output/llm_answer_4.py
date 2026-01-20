import numpy as np


def extract_final_answer(model_output):
    """
    Extracts the Homo sapiens vs reference (Pan) effect from a statsmodels GLM results object
    (possibly with cluster-robust covariances applied).

    Returns:
      {
        "object": {
          "param_name": str,
          "coef": float,
          "se": float,
          "z": float,                # or t-value depending on the result object
          "p_value": float,
          "ci_lower": float,
          "ci_upper": float,
          "odds_ratio": float,
          "or_ci_lower": float,
          "or_ci_upper": float,
          "significant": bool,
          "alpha": 0.05
        },
        "description": str  # brief interpretation in plain language
      }

    The description states whether Homo sapiens have higher/lower AMTL odds than Pan
    and whether the difference is statistically significant at alpha=0.05.
    """

    # Basic validation
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (missing .params).")

    params = model_output.params
    pvalues = getattr(model_output, "pvalues", None)
    bse = getattr(model_output, "bse", None)

    # Try to get a confidence interval table
    try:
        conf = model_output.conf_int()
    except Exception:
        conf = None

    # Helper to check membership safely
    def has_index_member(container, key):
        try:
            return key in container.index
        except Exception:
            try:
                # dict-like fallback
                return key in container
            except Exception:
                return False

    # Helper to get a value safely
    def get_val(container, key):
        try:
            return container[key]
        except Exception:
            try:
                return container.get(key)  # dict-like
            except Exception:
                return None

    # Identify the parameter name corresponding to Homo sapiens vs reference (Pan).
    param_candidates = []
    try:
        iter_names = list(params.index)
    except Exception:
        # If params doesn't have index (unlikely), try to treat it as mapping
        try:
            iter_names = list(params.keys())
        except Exception:
            iter_names = []

    for name in iter_names:
        try:
            lname = str(name).lower()
        except Exception:
            continue
        if "homo" in lname:
            param_candidates.append(name)

    # Prefer a candidate containing both 'homo' and 'sapiens' if present
    chosen_name = None
    if param_candidates:
        for n in param_candidates:
            if "sapiens" in str(n).lower():
                chosen_name = n
                break
        if chosen_name is None:
            chosen_name = param_candidates[0]

    if chosen_name is None:
        # If no Homo parameter found, return a descriptive error object
        available = ", ".join(map(str, iter_names)) if iter_names else "(none)"
        return {
            "object": None,
            "description": (
                "Could not find a model coefficient corresponding to 'Homo sapiens'. "
                "Parameter names present: " + available
            )
        }

    # Extract statistics for the chosen parameter
    coef_raw = get_val(params, chosen_name)
    try:
        coef = float(coef_raw)
    except Exception:
        raise ValueError(f"Could not convert coefficient for '{chosen_name}' to float (value: {coef_raw!r}).")

    # Standard error
    se_raw = get_val(bse, chosen_name) if bse is not None else None
    se = None
    try:
        se = float(se_raw) if se_raw is not None else None
    except Exception:
        se = None

    # p-value
    pval_raw = get_val(pvalues, chosen_name) if pvalues is not None else None
    pval = None
    try:
        pval = float(pval_raw) if pval_raw is not None else None
    except Exception:
        pval = None

    # z/t value: try tvalues then zvalues, else compute if se available
    zval = None
    tvalues = getattr(model_output, "tvalues", None)
    zvalues = getattr(model_output, "zvalues", None)
    if tvalues is not None and has_index_member(tvalues, chosen_name):
        try:
            zval = float(get_val(tvalues, chosen_name))
        except Exception:
            zval = None
    elif zvalues is not None and has_index_member(zvalues, chosen_name):
        try:
            zval = float(get_val(zvalues, chosen_name))
        except Exception:
            zval = None
    else:
        if se is not None and se != 0:
            zval = coef / se

    # Confidence interval on the log-odds scale
    ci_lower = ci_upper = None
    if conf is not None:
        try:
            if has_index_member(conf, chosen_name):
                # conf is expected to be a DataFrame with two columns
                # Some implementations return column labels as 0 and 1
                row = conf.loc[chosen_name]
                # row may be a Series with positional indices 0 and 1 or named columns
                try:
                    ci_lower = float(row.iloc[0])
                    ci_upper = float(row.iloc[1])
                except Exception:
                    # try by column names 'lower'/'upper' or similar
                    try:
                        ci_lower = float(row[0])
                        ci_upper = float(row[1])
                    except Exception:
                        ci_lower = ci_upper = None
        except Exception:
            ci_lower = ci_upper = None

    if ci_lower is None or ci_upper is None:
        if se is not None:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

    # Odds ratio and its CI (by exponentiating) if possible
    try:
        or_val = float(np.exp(coef))
    except Exception:
        or_val = None

    try:
        or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    except Exception:
        or_ci_lower = None

    try:
        or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        or_ci_upper = None

    # Determine statistical significance at alpha = 0.05 if p-value available
    alpha = 0.05
    significant = None
    if pval is not None:
        significant = (pval < alpha)

    # Build interpretation sentence:
    direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
    if significant is True:
        conclusion = (
            f"Homo sapiens have {direction} odds of AMTL compared to Pan, "
            f"and this difference is statistically significant (p = {pval:.3g})."
        )
    elif significant is False:
        conclusion = (
            f"Homo sapiens show {direction} odds of AMTL compared to Pan, "
            f"but the difference is not statistically significant (p = {pval:.3g})."
        )
    else:
        conclusion = (
            f"Homo sapiens show {direction} odds of AMTL compared to Pan. "
            "A p-value was not available to assess statistical significance."
        )

    # Compose the returned object with numeric values
    result_obj = {
        "param_name": chosen_name,
        "coef": coef,
        "se": se,
        "z": zval,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": or_val,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "significant": significant,
        "alpha": alpha
    }

    # Compose a concise description
    if or_ci_lower is not None and or_ci_upper is not None:
        description = (
            f"Parameter '{chosen_name}': coefficient = {coef:.4g} (log-odds). "
            f"Odds ratio = {or_val:.4g} with 95% CI [{or_ci_lower:.4g}, {or_ci_upper:.4g}]."
        )
    else:
        # If OR or CIs unavailable, include what we have
        description = (
            f"Parameter '{chosen_name}': coefficient = {coef:.4g} (log-odds). "
            f"Odds ratio = {or_val:.4g}." if or_val is not None else
            f"Parameter '{chosen_name}': coefficient = {coef:.4g} (log-odds)."
        )

    description += " " + conclusion

    return {"object": result_obj, "description": description}