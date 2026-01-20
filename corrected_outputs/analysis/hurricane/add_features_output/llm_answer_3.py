def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, sample size,
    and transformed (percent) effects for the masfem_z and gender_female predictors
    from the supplied statsmodels results objects.

    Returns a dictionary with:
      - "object": dict with keys 'masfem' and 'gender' (if available) containing numeric stats
      - "description": a short interpretation of whether results support the hypothesis
                       that more feminine names are associated with larger fatalities
                       (consistent with fewer precautions)

    model_output is expected to be a dict-like object with keys 'model_masfem' and 'model_gender'
    whose values are statsmodels RegressionResultsWrapper objects. The function is defensive
    to missing keys/objects.
    """
    import math
    import numpy as np

    def summarize_model(res, varname):
        """
        Given a statsmodels results object and the variable name, return a summary dict.
        """
        if res is None:
            return None

        # Ensure the variable exists in params
        params = getattr(res, "params", None)
        if params is None or varname not in params.index:
            return None

        coef = float(params[varname])
        se = float(res.bse[varname]) if hasattr(res, "bse") and varname in res.bse.index else None
        t = float(res.tvalues[varname]) if hasattr(res, "tvalues") and varname in res.tvalues.index else None
        p = float(res.pvalues[varname]) if hasattr(res, "pvalues") and varname in res.pvalues.index else None
        ci = None
        try:
            ci_array = res.conf_int(alpha=0.05)
            if varname in res.params.index:
                lower, upper = float(ci_array.loc[varname, 0]), float(ci_array.loc[varname, 1])
                ci = (lower, upper)
        except Exception:
            ci = None

        # Transform effect from ln(1+deaths) to percent change in (1 + deaths)
        # percent change = (exp(coef) - 1) * 100
        pct_change = None
        pct_ci = None
        try:
            pct_change = (math.exp(coef) - 1.0) * 100.0
            if ci is not None:
                pct_ci = ((math.exp(ci[0]) - 1.0) * 100.0, (math.exp(ci[1]) - 1.0) * 100.0)
        except Exception:
            pct_change = None
            pct_ci = None

        # Sample size and (robust) R-squared if available
        nobs = int(res.nobs) if hasattr(res, "nobs") else None
        rsquared = float(res.rsquared) if hasattr(res, "rsquared") else None

        return {
            "variable": varname,
            "coef": coef,
            "std_error": se,
            "t_value": t,
            "p_value": p,
            "conf_int_95": ci,
            "percent_change_on_1_plus_deaths": pct_change,
            "percent_change_conf_int_95": pct_ci,
            "nobs": nobs,
            "r_squared": rsquared
        }

    # Normalize possible input shapes
    model_masfem = None
    model_gender = None
    if isinstance(model_output, dict):
        model_masfem = model_output.get("model_masfem", None)
        model_gender = model_output.get("model_gender", None)
    else:
        # If a single results object was passed in error, try to use it for masfem
        try:
            # attempt attribute access
            model_masfem = getattr(model_output, "model_masfem", None)
            model_gender = getattr(model_output, "model_gender", None)
        except Exception:
            model_masfem = model_output

    summary = {}
    summary["masfem"] = summarize_model(model_masfem, "masfem_z") if model_masfem is not None else None
    summary["gender"] = summarize_model(model_gender, "gender_female") if model_gender is not None else None

    # Short interpretation
    interpretations = []
    # Evaluate masfem result
    m = summary.get("masfem")
    if m is None:
        interpretations.append("masfem_z model not available or variable 'masfem_z' not found.")
    else:
        sign = "positive" if m["coef"] > 0 else ("negative" if m["coef"] < 0 else "approximately zero")
        signif = ""
        if m["p_value"] is not None:
            signif = "statistically significant (p < 0.05)" if m["p_value"] < 0.05 else "not statistically significant (p >= 0.05)"
        interpretations.append(
            f"masfem_z: coefficient = {m['coef']:.4f} ({sign}), {signif}. "
            f"This corresponds to a {m['percent_change_on_1_plus_deaths']:.1f}% change in (1 + fatalities) "
            f"per 1 SD increase in name femininity"
            + (f" (95% CI: {m['percent_change_conf_int_95'][0]:.1f}% to {m['percent_change_conf_int_95'][1]:.1f}%)."
               if m["percent_change_conf_int_95"] is not None else ".")
        )

    # Evaluate gender (binary) result
    g = summary.get("gender")
    if g is None:
        interpretations.append("gender_female model not available or variable 'gender_female' not found.")
    else:
        sign = "positive" if g["coef"] > 0 else ("negative" if g["coef"] < 0 else "approximately zero")
        signif = ""
        if g["p_value"] is not None:
            signif = "statistically significant (p < 0.05)" if g["p_value"] < 0.05 else "not statistically significant (p >= 0.05)"
        interpretations.append(
            f"gender_female: coefficient = {g['coef']:.4f} ({sign}), {signif}. "
            f"This corresponds to a {g['percent_change_on_1_plus_deaths']:.1f}% difference in (1 + fatalities) "
            f"for storms with female vs male names"
            + (f" (95% CI: {g['percent_change_conf_int_95'][0]:.1f}% to {g['percent_change_conf_int_95'][1]:.1f}%)."
               if g["percent_change_conf_int_95"] is not None else ".")
        )

    description = " ".join(interpretations)

    return {
        "object": summary,
        "description": description
    }