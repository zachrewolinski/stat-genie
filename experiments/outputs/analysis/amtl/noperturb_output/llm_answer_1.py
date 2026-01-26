def extract_final_answer(model_output):
    """
    Extracts the effect of the 'Human' indicator from a fitted statsmodels GEE results object.

    Returns a dictionary with:
      - "object": dict with numeric results for the Human effect (coefficient, SE, z, p, 95% CI,
                  odds ratio and its CI, parameter name)
      - "description": brief interpretation answering whether modern humans have higher AMTL
                       than non-human primates after accounting for covariates.
    """
    import re
    import numpy as np

    res = model_output

    # Ensure required attributes exist
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    params = res.params
    param_names = list(params.index)

    # Find the parameter name that corresponds to the Human indicator.
    # This will match names like 'Human' or 'Human[T.True]' etc.
    human_params = [name for name in param_names if "Human" in name]
    if len(human_params) == 0:
        raise ValueError("No parameter matching 'Human' found in model_output params: found %r" % param_names)

    # If multiple matches, choose the first (common case is a single match).
    human_param = human_params[0]

    # Extract statistics (robust to availability of attributes)
    coef = float(params[human_param])
    # standard error
    if hasattr(res, "bse"):
        se = float(res.bse[human_param])
    elif hasattr(res, "standard_errors"):
        se = float(res.standard_errors[human_param])
    else:
        raise ValueError("Could not find standard errors on model output (expected .bse).")

    # z-value (compute if not provided)
    z_value = float(coef / se) if se != 0 else float("nan")

    # p-value
    if hasattr(res, "pvalues"):
        p_value = float(res.pvalues[human_param])
    else:
        # fallback: try to compute two-sided p from z
        from math import erf, sqrt
        # p = 2*(1 - Phi(|z|)); Phi via erf
        phi = 0.5 * (1.0 + erf(abs(z_value) / sqrt(2.0)))
        p_value = 2.0 * (1.0 - phi)

    # 95% confidence interval (on link/logit scale)
    if hasattr(res, "conf_int"):
        ci_df = res.conf_int()
        # conf_int returns a DataFrame; index should contain human_param
        if human_param in ci_df.index:
            ci_lower = float(ci_df.loc[human_param, 0])
            ci_upper = float(ci_df.loc[human_param, 1])
        else:
            # try columns by name if different ordering
            try:
                row = ci_df.loc[human_param]
                ci_lower = float(row.iloc[0])
                ci_upper = float(row.iloc[1])
            except Exception:
                raise ValueError("Could not extract confidence interval for parameter %r." % human_param)
    else:
        # approximate 95% CI from coef +/- 1.96*se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Transform to odds ratio scale (exp of coefficient) and CI
    or_est = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Conclusion logic at alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        if coef > 0:
            conclusion = (
                "Yes — the coefficient for %r is positive and statistically significant "
                "(coef = %.4f, p = %.4g). After adjusting for age, sex, and tooth class, "
                "modern humans have a higher probability of AMTL compared to non-human primates. "
                "Estimated odds ratio = %.3f (95%% CI: %.3f–%.3f)."
                % (human_param, coef, p_value, or_est, or_ci_lower, or_ci_upper)
            )
        else:
            conclusion = (
                "No — the coefficient for %r is negative and statistically significant "
                "(coef = %.4f, p = %.4g). After adjusting for age, sex, and tooth class, "
                "modern humans have a lower probability of AMTL compared to non-human primates. "
                "Estimated odds ratio = %.3f (95%% CI: %.3f–%.3f)."
                % (human_param, coef, p_value, or_est, or_ci_lower, or_ci_upper)
            )
    else:
        conclusion = (
            "No statistically significant difference detected — the coefficient for %r is %.4f "
            "(p = %.4g), which does not provide strong evidence that modern humans differ from "
            "non-human primates in AMTL frequency after adjusting for age, sex, and tooth class. "
            "Estimated odds ratio = %.3f (95%% CI: %.3f–%.3f)."
            % (human_param, coef, p_value, or_est, or_ci_lower, or_ci_upper)
        )

    result_object = {
        "param_name": human_param,
        "coef": coef,
        "std_error": se,
        "z_value": z_value,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": or_est,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "conclusion": conclusion
    }

    description = (
        "Extracted the coefficient and inference for the 'Human' indicator from the provided "
        "GEE model. The returned 'object' contains numeric results (coef, SE, z, p, 95% CI) on the "
        "logit scale and the exponentiated effect (odds ratio and its 95% CI). The 'conclusion' key "
        "summarizes whether modern humans show higher AMTL than non-human primates after adjusting "
        "for age_z, prob_male_z, and tooth_class, based on a 0.05 significance threshold."
    )

    return {"object": result_object, "description": description}