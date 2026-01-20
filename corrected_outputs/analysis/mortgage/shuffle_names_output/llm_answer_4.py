def extract_final_answer(model_output):
    """
    Extracts the estimated effect of the 'Female' indicator from a fitted statsmodels
    Logit (BinaryResultsWrapper) object and returns a concise numeric object plus
    an interpretation string.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, z, p, conf_int, odds_ratio, or_ci, nobs)
      - "description": human-readable explanation of what these numbers mean

    The function is defensive: if 'Female' is not in the model, or an extraction error
    occurs, it returns object=None and description explaining the error.
    """
    import numpy as np
    import pandas as pd

    try:
        results = model_output

        # Ensure this looks like a statsmodels results object
        if not hasattr(results, "params"):
            raise TypeError("model_output does not expose .params; expected a statsmodels results object")

        var = "Female"
        if var not in results.params.index:
            raise KeyError(f"Variable '{var}' not found in fitted model parameters: {list(results.params.index)}")

        # Extract basic statistics
        coef = float(results.params[var])
        se = float(results.bse[var]) if hasattr(results, "bse") else None
        p_value = float(results.pvalues[var]) if hasattr(results, "pvalues") else None

        # z-score (or t-like stat used by statsmodels for Logit)
        z_value = float(coef / se) if (se is not None and se != 0) else None

        # 95% confidence interval on coefficient (log-odds)
        try:
            ci_row = results.conf_int().loc[var].values
            ci_low, ci_high = float(ci_row[0]), float(ci_row[1])
        except Exception:
            # fallback if conf_int returns array-like without index
            ci = results.conf_int()
            # try to find row by position
            idx = list(results.params.index).index(var)
            ci_low, ci_high = float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])

        # Convert to odds ratio scale
        odds_ratio = float(np.exp(coef))
        or_ci_low, or_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

        # Sample size if available
        nobs = None
        if hasattr(results, "nobs") and results.nobs is not None:
            try:
                nobs = int(results.nobs)
            except Exception:
                nobs = None
        elif hasattr(results, "model") and hasattr(results.model, "nobs"):
            try:
                nobs = int(results.model.nobs)
            except Exception:
                nobs = None

        # Significance statement
        sig_threshold = 0.05
        if p_value is None:
            significance = "p-value not available"
        else:
            significance = "statistically significant (p < 0.05)" if p_value < sig_threshold else "not statistically significant (p >= 0.05)"

        # Build the numeric object to return
        numeric_object = {
            "variable": var,
            "coef_log_odds": round(coef, 6),
            "std_err": round(se, 6) if se is not None else None,
            "z_value": round(z_value, 6) if z_value is not None else None,
            "p_value": round(p_value, 6) if p_value is not None else None,
            "conf_int_log_odds": [round(ci_low, 6), round(ci_high, 6)],
            "odds_ratio": round(odds_ratio, 6),
            "conf_int_odds_ratio": [round(or_ci_low, 6), round(or_ci_high, 6)],
            "nobs": nobs
        }

        # Interpretation string
        direction = "increase" if odds_ratio > 1 else "decrease" if odds_ratio < 1 else "no change"
        description = (
            f"The estimated log-odds coefficient for '{var}' is {numeric_object['coef_log_odds']} "
            f"(SE = {numeric_object['std_err']}, z = {numeric_object['z_value']}, p = {numeric_object['p_value']}). "
            f"On the odds scale this corresponds to an odds ratio of {numeric_object['odds_ratio']} "
            f"with 95% CI [{numeric_object['conf_int_odds_ratio'][0]}, {numeric_object['conf_int_odds_ratio'][1]}]. "
            f"This implies that being female is associated with a {direction} in the odds of mortgage approval "
            f"compared with being male. The effect is {significance}. "
            + (f"Number of observations used: {nobs}." if nobs is not None else "")
        )

        return {"object": numeric_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting statistics for 'Female' from model_output: {repr(e)}"
        }