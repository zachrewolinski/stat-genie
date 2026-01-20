def extract_final_answer(model_output):
    """
    Extracts coefficient estimates, standard errors, z-stats (or t-stats), p-values,
    95% confidence intervals, and multiplicative effects (exp(coef)) for the
    predictors of interest: 'age', 'sex_M', and 'help_Y' from a fitted
    statsmodels MixedLMResults or MixedLMResultsWrapper object.

    Returns a dict with keys:
      - "object": dict mapping each predictor -> its numeric summary (coef, se, z, p, CI, exp(coef), exp(CI), significant)
      - "description": brief interpretation of what the numbers mean in the study context
    """
    import numpy as np
    import pandas as pd
    from math import exp
    try:
        from scipy import stats
        _has_scipy = True
    except Exception:
        _has_scipy = False

    # Ensure we have the attributes we need (params, bse)
    if not hasattr(model_output, "params") or not hasattr(model_output, "bse"):
        raise ValueError("model_output does not look like a statsmodels results object with .params and .bse")

    params = model_output.params
    bse = model_output.bse

    # p-values: try to get from object, otherwise compute from normal approximation
    if hasattr(model_output, "pvalues"):
        pvalues = model_output.pvalues
    else:
        # compute z-stats and p-values using normal distribution
        z = params / bse
        if _has_scipy:
            pvalues = 2 * stats.norm.sf(np.abs(z))
        else:
            # approximate with numpy using erf if scipy not present
            from math import erf, sqrt
            def _norm_sf(x):
                # survival function approximation using erf
                return 0.5 * (1.0 - erf(x / sqrt(2.0)))
            pvalues = 2 * np.array([_norm_sf(abs(val)) for val in z])
        pvalues = pd.Series(pvalues, index=params.index)

    # confidence intervals: use model_output.conf_int() when available
    try:
        ci = model_output.conf_int()
        # conf_int returns DataFrame with two columns; ensure names 0 and 1
        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]
    except Exception:
        # fallback to normal approx 95% CI
        ci_lower = params - 1.96 * bse
        ci_upper = params + 1.96 * bse
        ci_lower = pd.Series(ci_lower, index=params.index)
        ci_upper = pd.Series(ci_upper, index=params.index)

    predictors = ['age', 'sex_M', 'help_Y']
    results = {}
    for pred in predictors:
        if pred not in params.index:
            results[pred] = {
                "present": False,
                "message": f"Predictor '{pred}' not found in model fixed effects."
            }
            continue

        coef = float(params[pred])
        se = float(bse[pred]) if pred in bse.index else None
        z_stat = float(coef / se) if (se is not None and se != 0) else None
        pval = float(pvalues[pred]) if pred in pvalues.index else None
        ci_l = float(ci_lower[pred])
        ci_u = float(ci_upper[pred])

        # multiplicative effect on the original rate (nuts/sec): exp(coef)
        mult = float(np.exp(coef))
        mult_ci_l = float(np.exp(ci_l))
        mult_ci_u = float(np.exp(ci_u))

        significance = (pval is not None) and (pval < 0.05)

        # short plain-language interpretation for this predictor
        if pred == 'age':
            interp = (f"Per additional year of age, the log rate of nuts opened/sec changes by {coef:.4f} "
                      f"(95% CI [{ci_l:.4f}, {ci_u:.4f}], p={pval:.3g}). "
                      f"On the original rate scale this corresponds to a multiplicative change of {mult:.3f} "
                      f"(95% CI [{mult_ci_l:.3f}, {mult_ci_u:.3f}]).")
        elif pred == 'sex_M':
            interp = (f"Being male (vs female) is associated with a change in log rate of nuts opened/sec of {coef:.4f} "
                      f"(95% CI [{ci_l:.4f}, {ci_u:.4f}], p={pval:.3g}). "
                      f"On the original rate scale this corresponds to a multiplicative factor of {mult:.3f} "
                      f"(95% CI [{mult_ci_l:.3f}, {mult_ci_u:.3f}]).")
        else:  # help_Y
            interp = (f"Receiving help (vs no help) is associated with a change in log rate of nuts opened/sec of {coef:.4f} "
                      f"(95% CI [{ci_l:.4f}, {ci_u:.4f}], p={pval:.3g}). "
                      f"On the original rate scale this corresponds to a multiplicative factor of {mult:.3f} "
                      f"(95% CI [{mult_ci_l:.3f}, {mult_ci_u:.3f}]).")

        results[pred] = {
            "present": True,
            "coef": coef,
            "se": se,
            "z_or_t": z_stat,
            "p_value": pval,
            "ci_95": [ci_l, ci_u],
            "exp_coef (multiplicative effect on nuts/sec)": mult,
            "exp_ci_95": [mult_ci_l, mult_ci_u],
            "significant_at_0.05": bool(significance),
            "interpretation": interp
        }

    # Optional: include model-level info (random intercept variance & residual)
    model_info = {}
    try:
        # random effects covariance (variance of random intercept)
        re_cov = getattr(model_output, "cov_re", None)
        if re_cov is not None:
            # if single random intercept, cov_re is 1x1 matrix-like
            try:
                var_re = float(np.asarray(re_cov).ravel()[0])
                model_info['random_intercept_variance'] = var_re
            except Exception:
                model_info['random_effects_covariance'] = re_cov
        # scale (residual variance)
        if hasattr(model_output, "scale"):
            model_info['residual_variance (scale)'] = float(model_output.scale)
    except Exception:
        pass

    description_lines = [
        "Extracted fixed-effect estimates for predictors 'age', 'sex_M', and 'help_Y' from the fitted mixed model.",
        "Coefficients are on the log(rate) scale where rate = (nuts_opened + 0.5) / seconds.",
        "exp(coef) gives the multiplicative change in that rate associated with a one-unit increase in the predictor (or the factor level vs reference).",
        "Significance is reported using two-sided p-values (or normal-approximation if p-values not provided by the model)."
    ]
    if model_info:
        description_lines.append("Also returned some model-level variance estimates when available.")

    description = " ".join(description_lines)

    return {
        "object": {
            "predictor_summaries": results,
            "model_info": model_info
        },
        "description": description
    }