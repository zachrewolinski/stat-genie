def extract_final_answer(model_output):
    """
    Extracts statistics for the independent variable (masfem_z) from the provided model_output.
    Returns a dict with keys:
      - "object": dict of extracted numeric results (coef, se, stat, pvalue, CI, IRR, IRR_CI, significance, model_used, variable)
      - "description": plain-English explanation of what those numbers mean for the hypothesis.
    """
    import numpy as np

    # Select preferred models in order
    preferred_keys = ['nb_glm_robust', 'nb_glm', 'poisson_glm', 'ols_log_outcome']
    model = None
    model_key = None
    for k in preferred_keys:
        if k in model_output and model_output[k] is not None:
            model = model_output[k]
            model_key = k
            break

    if model is None:
        return {
            "object": None,
            "description": "No usable model object found in model_output under keys: "
                           + ", ".join(preferred_keys)
        }

    # Variable names to try
    var_candidates = ['masfem_z', 'masfem']

    # Helper to safely extract from result object
    def safe_get_attr(obj, attr):
        return getattr(obj, attr) if hasattr(obj, attr) else None

    coef = se = pval = stat = ci = None
    used_var = None

    for var in var_candidates:
        try:
            params = safe_get_attr(model, 'params')
            if params is None:
                # some wrappers store params as a dict-like; try getattr
                params = model.__dict__.get('params', None)
            if params is None or var not in params:
                continue

            coef = float(params[var])

            # standard error
            bse = safe_get_attr(model, 'bse')
            if bse is not None and var in bse:
                se = float(bse[var])
            else:
                se = None

            # p-value
            pvals = safe_get_attr(model, 'pvalues')
            if pvals is not None and var in pvals:
                pval = float(pvals[var])
            else:
                pval = None

            # test statistic (t or z)
            tvals = safe_get_attr(model, 'tvalues')
            zvals = safe_get_attr(model, 'zvalues')
            if tvals is not None and var in tvals:
                stat = float(tvals[var])
            elif zvals is not None and var in zvals:
                stat = float(zvals[var])
            else:
                stat = (coef / se) if (se is not None and se != 0) else None

            # confidence interval: model.conf_int() may return array-like or DataFrame
            try:
                conf = safe_get_attr(model, 'conf_int')
                if callable(conf):
                    ci_mat = conf()
                    # conf_int can be ndarray or DataFrame; handle both
                    if hasattr(ci_mat, 'loc') and var in ci_mat.index:
                        row = ci_mat.loc[var].tolist()
                        ci = [float(row[0]), float(row[1])]
                    else:
                        # assume order of params matches params.index
                        # try to find index of var in params.index
                        try:
                            idx = list(params.index).index(var)
                            ci = [float(ci_mat[idx, 0]), float(ci_mat[idx, 1])]
                        except Exception:
                            # as fallback, try to interpret ci_mat as dict-like
                            try:
                                row = ci_mat[var]
                                ci = [float(row[0]), float(row[1])]
                            except Exception:
                                ci = None
                else:
                    ci = None
            except Exception:
                ci = None

            used_var = var
            break
        except Exception:
            continue

    if coef is None:
        return {
            "object": None,
            "description": "Variable 'masfem_z' (or 'masfem') was not found in the model parameters."
        }

    # For count models (NB/Poisson) exponentiate coefficient to get incidence rate ratio
    irr = None
    irr_ci = None
    # Determine if model is a count model by the key used
    if model_key in ('nb_glm_robust', 'nb_glm', 'poisson_glm'):
        try:
            irr = float(np.exp(coef))
            if ci is not None:
                irr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
        except Exception:
            irr = None
            irr_ci = None
    else:
        # For OLS on log(outcome+1), coef approximates % change for small values; still report exp(coef)
        try:
            irr = float(np.exp(coef))
            if ci is not None:
                irr_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
        except Exception:
            irr = None
            irr_ci = None

    significance = None
    if pval is not None:
        significance = bool(pval < 0.05)

    # Build object to return
    result_object = {
        "model_used": model_key,
        "variable": used_var,
        "coef": coef,
        "se": se,
        "statistic": stat,
        "p_value": pval,
        "conf_int_95": ci,
        "incidence_rate_ratio_or_exp_coef": irr,
        "irr_95_CI": irr_ci,
        "statistically_significant_p_lt_0.05": significance
    }

    # Build human-readable description / interpretation
    # Interpretation for count models: coef is log change in expected count per 1 SD increase in name femininity.
    interpretation_parts = []
    interpretation_parts.append(
        f"Model used: {model_key}. Extracted variable: {used_var}."
    )
    interpretation_parts.append(
        f"Coefficient (log scale) = {coef:.4f}" + (f", SE = {se:.4f}" if se is not None else "")
        + (f", stat = {stat:.3f}" if stat is not None else "") + (f", p = {pval:.3f}" if pval is not None else "")
    )
    if ci is not None:
        interpretation_parts.append(f"95% CI for coef = [{ci[0]:.4f}, {ci[1]:.4f}].")
    if irr is not None:
        interpretation_parts.append(f"Exponential effect (IRR or exp(coef)) = {irr:.4f}.")
        if irr_ci is not None:
            interpretation_parts.append(f"95% CI for IRR = [{irr_ci[0]:.4f}, {irr_ci[1]:.4f}].")
    # Directional interpretation relative to hypothesis:
    # For count models, positive coef -> higher expected fatalities as masfem_z increases.
    if coef > 0:
        dir_text = ("Positive coefficient: higher name femininity (one SD increase) is associated with "
                    "higher expected fatalities (holding controls constant).")
    elif coef < 0:
        dir_text = ("Negative coefficient: higher name femininity (one SD increase) is associated with "
                    "lower expected fatalities (holding controls constant).")
    else:
        dir_text = "Coefficient is exactly zero (no estimated association)."
    interpretation_parts.append(dir_text)

    if significance is not None:
        sig_text = ("The effect is statistically significant at alpha=0.05." if significance
                    else "The effect is NOT statistically significant at alpha=0.05.")
        interpretation_parts.append(sig_text)

    interpretation = " ".join(interpretation_parts)

    return {
        "object": result_object,
        "description": interpretation
    }