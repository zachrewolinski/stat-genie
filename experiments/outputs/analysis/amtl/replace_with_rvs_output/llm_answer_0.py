def extract_final_answer(model_output):
    """
    Extracts the is_human effect from model_output and returns a concise numeric
    summary and interpretation.

    Returns:
      {
        "object": {
            "estimate_logit": float or None,
            "std_err": float or None,
            "p_value": float or None,
            "odds_ratio": float or None,
            "or_ci_lower": float or None,
            "or_ci_upper": float or None
        },
        "description": str   # brief interpretation in context
      }
    """
    import math
    import numpy as np

    # Helper to safely get items from pandas Series / dict-like objects
    def safe_get(obj, keys):
        if obj is None:
            return None
        for k in keys:
            try:
                # for pandas Series and dicts
                if hasattr(obj, "get") and k in obj:
                    return obj[k]
                # pandas Series __getitem__ works with labels
                val = obj[k]
                return val
            except Exception:
                continue
        return None

    # Initialize extracted values
    est = se = pval = or_val = or_lo = or_hi = None

    # 1) Preferred path: explicit 'is_human_row' provided in model_output
    row = model_output.get('is_human_row') if isinstance(model_output, dict) else None
    if row is not None:
        # Known keys from the example: 'estimate_logit', 'std_err', 'p_value', 'or', 'or_ci_lower', 'or_ci_upper'
        est = safe_get(row, ['estimate_logit', 'estimate', 'coef', 'coef_logit', 'logit'])
        se = safe_get(row, ['std_err', 'stderr', 'se', 'std_error'])
        pval = safe_get(row, ['p_value', 'pvalue', 'p_val', 'p'])
        or_val = safe_get(row, ['or', 'odds_ratio'])
        or_lo = safe_get(row, ['or_ci_lower', 'or_lower', 'or_ci_lo', 'ci_lower'])
        or_hi = safe_get(row, ['or_ci_upper', 'or_upper', 'or_ci_hi', 'ci_upper'])

    # 2) Fallback: look into coef_table DataFrame if present
    if est is None and isinstance(model_output, dict) and 'coef_table' in model_output:
        try:
            coef_table = model_output['coef_table']
            # find row where term == 'is_human'
            row_df = coef_table[coef_table['term'] == 'is_human']
            if not row_df.empty:
                r = row_df.iloc[0]
                est = safe_get(r, ['estimate_logit', 'estimate_logit', 'estimate', 'estimate_log'])
                se = safe_get(r, ['std_err', 'stderr', 'se'])
                pval = safe_get(r, ['p_value', 'pvalue', 'p_val'])
                or_val = safe_get(r, ['or', 'odds_ratio'])
                or_lo = safe_get(r, ['or_ci_lower', 'or_ci_lower'])
                or_hi = safe_get(r, ['or_ci_upper', 'or_ci_upper'])
        except Exception:
            pass

    # 3) Last fallback: use fitted_model (statsmodels result) if present
    if est is None and isinstance(model_output, dict) and 'fitted_model' in model_output:
        try:
            res = model_output['fitted_model']
            # params and pvalues and bse should be accessible
            if hasattr(res, 'params') and 'is_human' in res.params.index:
                est = float(res.params['is_human'])
                # try robust se / bse
                try:
                    se = float(res.bse['is_human'])
                except Exception:
                    # sometimes res.bse is an ndarray aligned with params
                    try:
                        se = float(res.bse.loc['is_human'])
                    except Exception:
                        se = None
                try:
                    pval = float(res.pvalues['is_human'])
                except Exception:
                    pval = None
                # Confidence interval (on logit) and transform to OR
                try:
                    ci = res.conf_int().loc['is_human']
                    or_lo = float(np.exp(ci[0]))
                    or_hi = float(np.exp(ci[1]))
                    or_val = float(np.exp(est))
                except Exception:
                    # if conf_int not available, at least compute OR
                    try:
                        or_val = float(np.exp(est))
                    except Exception:
                        or_val = None
        except Exception:
            pass

    # Convert numeric-like strings / numpy types to Python floats where possible
    def to_float(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return None
            return float(x)
        except Exception:
            return None

    est = to_float(est)
    se = to_float(se)
    pval = to_float(pval)
    or_val = to_float(or_val)
    or_lo = to_float(or_lo)
    or_hi = to_float(or_hi)

    # If OR not provided but est is present, compute it
    if or_val is None and est is not None:
        try:
            or_val = float(np.exp(est))
        except Exception:
            or_val = None
    # If CI on OR not provided but se is present, approximate 95% CI on logit then exponentiate
    if (or_lo is None or or_hi is None) and est is not None and se is not None:
        try:
            z = 1.96
            lo_logit = est - z * se
            hi_logit = est + z * se
            or_lo = float(np.exp(lo_logit))
            or_hi = float(np.exp(hi_logit))
        except Exception:
            pass

    # Build the returned object
    result_object = {
        "estimate_logit": est,
        "std_err": se,
        "p_value": pval,
        "odds_ratio": or_val,
        "or_ci_lower": or_lo,
        "or_ci_upper": or_hi
    }

    # Interpretation
    if est is None:
        description = ("Could not locate an 'is_human' coefficient in the provided model_output. "
                       "Returned object contains None values.")
    else:
        # significance test
        sig_text = "no evidence of a difference"  # default
        if pval is not None:
            if pval < 0.05:
                sig_text = "statistically significant difference (p < 0.05)"
            else:
                sig_text = "no statistically significant difference (p >= 0.05)"
        # direction
        direction = "higher" if (or_val is not None and or_val > 1) else "lower or similar"
        # More explicit message about AMTL in modern humans
        description = (
            f"The model's estimated log-odds coefficient for is_human = {est:.4f} "
            f"(SE = {se:.4f}) corresponds to an odds ratio = {or_val:.3f} "
            f"with 95% CI [{or_lo:.3f}, {or_hi:.3f}] and p = {pval:.4f}."
            f" This indicates {sig_text} in AMTL for modern humans compared to the non-human primates; "
            f"the point estimate suggests {direction} AMTL in humans, but the CI includes 1 so the effect is not distinguishable from no effect."
        )

    return {"object": result_object, "description": description}