def extract_final_answer(model_output):
    """
    Extract statistics about the effect of beauty (Beauty_z and Beauty_z2)
    from a fitted statsmodels regression result (RegressionResultsWrapper).

    Returns a dict with:
      - "object": a dict containing coefficients, SEs, p-values, 95% CIs for Beauty_z
                  and Beauty_z2, marginal effects of Beauty_z at values
                  [-1, 0, 1] (with SE and p-value), and a joint (Wald/F) test
                  of the hypothesis that both Beauty_z and Beauty_z2 are zero.
      - "description": a short explanation of the numbers and how to interpret them.

    The function is defensive: it checks required attributes and falls back to
    reasonable approximations where necessary.
    """
    import numpy as np
    import math

    res = model_output

    # Ensure the object looks like a statsmodels RegressionResults-like object
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (missing .params)")

    params = res.params  # usually a pandas Series
    # Ensure we can get index/keys
    try:
        names = list(params.index)
    except Exception:
        # params might be ndarray; in that case we cannot map by name
        raise ValueError("model_output.params does not have an index of parameter names")

    # Required variable names
    var1 = 'Beauty_z'
    var2 = 'Beauty_z2'

    if var1 not in names or var2 not in names:
        raise ValueError(f"Required variables not found in model params. Expected {var1} and {var2} in {names}")

    # Extract coefficients
    coef1 = float(params[var1])
    coef2 = float(params[var2])

    # Helper to obtain covariance matrix as numpy array (if available)
    cov_arr = None
    cov_obj = getattr(res, "cov_params", None)
    if cov_obj is not None:
        try:
            # cov_params may be a method or an attribute (DataFrame/ndarray)
            cov_raw = cov_obj() if callable(cov_obj) else cov_obj
            cov_arr = np.asarray(cov_raw)
        except Exception:
            try:
                cov_arr = np.asarray(cov_obj)
            except Exception:
                cov_arr = None

    # Standard errors and p-values
    se1 = se2 = None
    p1 = p2 = None
    bse_obj = getattr(res, "bse", None)
    if bse_obj is not None:
        try:
            se1 = float(bse_obj[var1])
            se2 = float(bse_obj[var2])
        except Exception:
            try:
                se1 = float(np.asarray(bse_obj)[names.index(var1)])
                se2 = float(np.asarray(bse_obj)[names.index(var2)])
            except Exception:
                se1 = se2 = None

    pvals_obj = getattr(res, "pvalues", None)
    if pvals_obj is not None:
        try:
            p1 = float(pvals_obj[var1])
            p2 = float(pvals_obj[var2])
        except Exception:
            try:
                p1 = float(np.asarray(pvals_obj)[names.index(var1)])
                p2 = float(np.asarray(pvals_obj)[names.index(var2)])
            except Exception:
                p1 = p2 = None

    # If SEs missing but covariance matrix available, compute from diagonal
    if (se1 is None or se2 is None) and cov_arr is not None:
        try:
            idx1 = names.index(var1)
            idx2 = names.index(var2)
            se1 = float(math.sqrt(max(0.0, float(cov_arr[idx1, idx1]))))
            se2 = float(math.sqrt(max(0.0, float(cov_arr[idx2, idx2]))))
        except Exception:
            # leave as None if anything fails
            pass

    # 95% confidence intervals
    ci1 = (None, None)
    ci2 = (None, None)
    ci_obj = getattr(res, "conf_int", None)
    if ci_obj is not None:
        try:
            ci_raw = ci_obj(alpha=0.05) if callable(ci_obj) else ci_obj
            # conf_int may be DataFrame or ndarray; handle both
            if hasattr(ci_raw, "loc") and var1 in ci_raw.index:
                # assume two columns: lower, upper
                ci1 = (float(ci_raw.loc[var1].iloc[0]), float(ci_raw.loc[var1].iloc[1]))
                ci2 = (float(ci_raw.loc[var2].iloc[0]), float(ci_raw.loc[var2].iloc[1]))
            else:
                ci_arr = np.asarray(ci_raw)
                idx1 = names.index(var1)
                idx2 = names.index(var2)
                ci1 = (float(ci_arr[idx1, 0]), float(ci_arr[idx1, 1]))
                ci2 = (float(ci_arr[idx2, 0]), float(ci_arr[idx2, 1]))
        except Exception:
            ci1 = (None, None)
            ci2 = (None, None)

    # Marginal effect of Beauty (derivative) = beta1 + 2 * beta2 * z
    # Compute at z = -1, 0, +1 (interpretable because beauty is standardized)

    def me_stats(z):
        me = coef1 + 2.0 * coef2 * z
        se_me = None
        tstat = None
        pval = None
        if cov_arr is not None:
            try:
                idx1 = names.index(var1)
                idx2 = names.index(var2)
                # Extract 2x2 covariance submatrix using integer positions
                sub = cov_arr[[idx1, idx2], :][:, [idx1, idx2]]
                vec = np.array([1.0, 2.0 * z])
                var_me = float(vec @ sub @ vec.T)
                se_me = float(math.sqrt(max(var_me, 0.0)))
                if se_me > 0:
                    tstat = float(me / se_me)
                    # compute p-value using t-distribution if df_resid available, else normal approx
                    try:
                        from scipy import stats
                        df = None
                        if hasattr(res, "df_resid"):
                            try:
                                df = float(res.df_resid) if (res.df_resid is not None) else None
                            except Exception:
                                df = None
                        if df is not None and not np.isnan(df) and df > 0:
                            pval = float(2.0 * stats.t.sf(abs(tstat), df))
                        else:
                            pval = float(2.0 * stats.norm.sf(abs(tstat)))
                    except Exception:
                        # fallback to normal approx using math.erf
                        pval = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(tstat) / math.sqrt(2.0)))))
            except Exception:
                se_me = None
                tstat = None
                pval = None
        return {"z": float(z), "marginal_effect": float(me), "se": se_me, "t": tstat, "p_value": pval}

    marg_effects = {str(z): me_stats(z) for z in (-1.0, 0.0, 1.0)}

    # Joint test that both coefficients (Beauty_z and Beauty_z2) are zero
    joint_test = {"f_stat": None, "p_value": None, "df_denom": None, "df_num": None}
    try:
        # build restriction matrix R of shape (2, k_params)
        k = len(names)
        R = np.zeros((2, k))
        R[0, names.index(var1)] = 1.0
        R[1, names.index(var2)] = 1.0
        ftest = res.f_test(R)
        # ftest may be a Contrast object with attributes .fvalue, .pvalue
        fval = getattr(ftest, "fvalue", None)
        p_joint = getattr(ftest, "pvalue", None)
        # fvalue/pvalue might be arrays; convert to scalar if possible
        if hasattr(fval, "__iter__"):
            try:
                fval = float(np.asarray(fval).squeeze())
            except Exception:
                fval = None
        if hasattr(p_joint, "__iter__"):
            try:
                p_joint = float(np.asarray(p_joint).squeeze())
            except Exception:
                p_joint = None
        joint_test.update({"f_stat": fval, "p_value": p_joint})
        # df info if available
        try:
            dfnum = getattr(ftest, "df_num", None)
            dfden = getattr(ftest, "df_denom", None)
            joint_test.update({"df_num": float(dfnum) if dfnum is not None else None,
                               "df_denom": float(dfden) if dfden is not None else None})
        except Exception:
            pass
    except Exception:
        # if f_test failed, leave joint_test with None values
        pass

    result_object = {
        "coef": {
            var1: {"coef": coef1, "se": se1, "p_value": p1, "ci_95": ci1},
            var2: {"coef": coef2, "se": se2, "p_value": p2, "ci_95": ci2},
        },
        "marginal_effects_at": marg_effects,
        "joint_test_beauty_and_beauty2": joint_test,
        "model_param_names": names
    }

    # Prepare a short human-readable description explaining how to interpret the numbers.
    description_lines = [
        "This output contains coefficient estimates and inference for the linear (Beauty_z) and",
        "quadratic (Beauty_z2) terms for instructor attractiveness in the course-level",
        "student evaluation OLS model, plus marginal effects of beauty (the derivative)",
        "evaluated at z = -1, 0, +1 (since Beauty_z is standardized).",
        "",
        "How to interpret:",
        "- coef[Beauty_z]: the slope of EvalScore with respect to standardized beauty when",
        "  the quadratic term is zero (or approximately at the mean of Beauty_z).",
        "- coef[Beauty_z2]: if nonzero, indicates curvature. A positive coef means",
        "  the slope increases with beauty; a negative coef means diminishing or",
        "  non-linear (e.g., concave) relationship.",
        "- Marginal effects (marginal_effects_at) give the instantaneous effect of a one-SD",
        "  increase in beauty on EvalScore at specific beauty levels (z = -1, 0, +1).",
        "- Joint test (joint_test_beauty_and_beauty2) tests whether both coefficients are",
        "  simultaneously zero (no effect of beauty in any linear/quadratic form).",
        "",
        "Use the p-values to judge statistical significance (small p-values, e.g. < 0.05,",
        "indicate rejection of the null of no effect). Confidence intervals give a range",
        "of plausible values for each coefficient.",
        "",
        "The 'object' field contains numeric values you can programmatically inspect."
    ]

    description = "\n".join(description_lines)

    return {"object": result_object, "description": description}