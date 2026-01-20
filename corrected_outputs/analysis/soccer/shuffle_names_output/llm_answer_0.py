def extract_final_answer(model_output):
    """
    Extracts the effect of the SkinDark variable from a fitted statsmodels GLM (or
    cluster-robust) results object and returns a dictionary with numeric results
    and a short interpretation.

    Returns:
      {
        "object": {
          "param_name": str,
          "coef": float,
          "se": float,
          "pvalue": float,
          "ci_lower": float,
          "ci_upper": float,
          "IRR": float,            # Incidence Rate Ratio = exp(coef)
          "IRR_ci_lower": float,
          "IRR_ci_upper": float,
          "significant": bool
        },
        "description": str
      }
    """
    import numpy as np
    import pandas as pd
    from math import isnan
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    # Ensure we can read parameter names
    if not hasattr(res, "params"):
        raise ValueError("Provided model_output does not have 'params' attribute.")

    params = res.params
    # Normalize params to a pandas Series for easy indexing
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Unable to coerce model params to pandas Series.")

    names = list(params.index)

    # Find the SkinDark parameter name (exact match preferred, otherwise substring)
    param_name = None
    if "SkinDark" in names:
        param_name = "SkinDark"
    else:
        # fallback: any parameter name that contains 'Skin' (case-insensitive)
        lower_names = [n.lower() for n in names]
        for n, ln in zip(names, lower_names):
            if "skin" in ln:
                param_name = n
                break

    if param_name is None:
        raise ValueError("Could not find a parameter name for skin tone (expected 'SkinDark' or name containing 'skin').")

    # Extract coefficient
    coef = float(params[param_name])

    # Extract standard error: prefer res.bse if available, otherwise derive from cov_params
    se = None
    if hasattr(res, "bse"):
        try:
            bse = res.bse
            if isinstance(bse, (pd.Series, dict)):
                se = float(bse[param_name])
            else:
                # array-like: align by index position
                pos = names.index(param_name)
                se = float(bse[pos])
        except Exception:
            se = None

    if se is None:
        # Try deriving from covariance matrix
        if hasattr(res, "cov_params"):
            try:
                cov = res.cov_params()
                # cov may be DataFrame or ndarray
                if isinstance(cov, pd.DataFrame):
                    se = float(np.sqrt(np.abs(cov.loc[param_name, param_name])))
                else:
                    pos = names.index(param_name)
                    se = float(np.sqrt(np.abs(cov[pos, pos])))
            except Exception:
                se = None

    if se is None or not np.isfinite(se):
        raise ValueError("Could not obtain a finite standard error for parameter '%s'." % param_name)

    # Extract p-value if provided; otherwise compute from z-stat
    pvalue = None
    if hasattr(res, "pvalues"):
        try:
            pv = res.pvalues
            if isinstance(pv, (pd.Series, dict)):
                pvalue = float(pv[param_name])
            else:
                pos = names.index(param_name)
                pvalue = float(pv[pos])
        except Exception:
            pvalue = None

    if pvalue is None:
        # compute two-sided p-value from normal approximation
        if stats is not None:
            z = coef / se
            pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
        else:
            # fallback: use large-sample normal approx via numpy (erf)
            z = coef / se
            # cdf of standard normal via erf
            cdf = 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))
            pvalue = float(2.0 * (1.0 - cdf))

    # Confidence intervals: try res.conf_int(), map to parameter
    ci_low = ci_high = None
    if hasattr(res, "conf_int"):
        try:
            ci = res.conf_int()
            # conf_int may be ndarray or DataFrame
            if isinstance(ci, pd.DataFrame):
                # columns are typically [0,1] or ['lower','upper']; use iloc to be safe
                try:
                    ci_row = ci.loc[param_name]
                except Exception:
                    # if index alignment failed, fallback to positional
                    pos = names.index(param_name)
                    ci_row = ci.iloc[pos]
                ci_low = float(ci_row.iloc[0])
                ci_high = float(ci_row.iloc[1])
            else:
                # numpy array: rows align with params order
                pos = names.index(param_name)
                ci_low = float(ci[pos, 0])
                ci_high = float(ci[pos, 1])
        except Exception:
            ci_low = ci_high = None

    if ci_low is None or ci_high is None or not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        # fallback: approximate CI from coef +/- 1.96*se
        z_crit = 1.96
        ci_low = float(coef - z_crit * se)
        ci_high = float(coef + z_crit * se)

    # Incidence Rate Ratio and its CI
    IRR = float(np.exp(coef))
    IRR_ci_lower = float(np.exp(ci_low))
    IRR_ci_upper = float(np.exp(ci_high))

    significant = (pvalue < 0.05)

    # Build descriptive conclusion
    if significant:
        if coef > 0:
            conclusion = (
                "Yes — the coefficient for '%s' is positive and statistically significant "
                "(coef = %.4g, p = %.3g). This implies dark-skinned players receive red cards at a "
                "higher rate than light-skinned players. IRR = %.3g (95%% CI: %.3g–%.3g)."
                % (param_name, coef, pvalue, IRR, IRR_ci_lower, IRR_ci_upper)
            )
        else:
            conclusion = (
                "Yes, but in the opposite direction: the coefficient for '%s' is negative and "
                "statistically significant (coef = %.4g, p = %.3g), implying dark-skinned players "
                "receive red cards at a lower rate than light-skinned players. IRR = %.3g (95%% CI: %.3g–%.3g)."
                % (param_name, coef, pvalue, IRR, IRR_ci_lower, IRR_ci_upper)
            )
    else:
        conclusion = (
            "No — the effect of '%s' is not statistically significant at the 0.05 level "
            "(coef = %.4g, p = %.3g). The estimated IRR is %.3g (95%% CI: %.3g–%.3g), "
            "so we cannot conclude a reliable difference in red card rates between dark- and light-skinned players."
            % (param_name, coef, pvalue, IRR, IRR_ci_lower, IRR_ci_upper)
        )

    result_object = {
        "param_name": param_name,
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "IRR": IRR,
        "IRR_ci_lower": IRR_ci_lower,
        "IRR_ci_upper": IRR_ci_upper,
        "significant": significant,
    }

    return {"object": result_object, "description": conclusion}