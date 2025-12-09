def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, 95% CI, number of observations, and R-squared
    for the femininity variables from the provided statsmodels RegressionResultsWrapper objects.

    Returns:
      {
        "object": {
          "coder": { ... } or None,
          "mturk": { ... } or None
        },
        "description": "Brief interpretation of the results in context."
      }
    """
    import math
    from scipy import stats
    results = {"coder": None, "mturk": None}

    def _extract_from_result(res, varname):
        if res is None:
            return None
        params = getattr(res, "params", None)
        if params is None or varname not in params.index:
            return None

        coef = float(params[varname])
        # robust se already used in fit (cov_type='HC3'), accessible via bse
        bse = float(res.bse[varname]) if (hasattr(res, "bse") and varname in res.bse.index) else None
        tval = float(res.tvalues[varname]) if (hasattr(res, "tvalues") and varname in res.tvalues.index) else None
        pval = float(res.pvalues[varname]) if (hasattr(res, "pvalues") and varname in res.pvalues.index) else None

        # Try to get conf_int; if not available in an indexable form, compute using t critical value
        try:
            ci = res.conf_int().loc[varname]
            ci_lower = float(ci[0])
            ci_upper = float(ci[1])
        except Exception:
            # fallback: use t critical with df_resid
            if bse is not None and hasattr(res, "df_resid"):
                df = float(res.df_resid)
                if math.isfinite(df) and df > 0:
                    t_crit = float(stats.t.ppf(0.975, df))
                    ci_lower = coef - t_crit * bse
                    ci_upper = coef + t_crit * bse
                else:
                    ci_lower = None
                    ci_upper = None
            else:
                ci_lower = None
                ci_upper = None

        nobs = int(res.nobs) if hasattr(res, "nobs") else None
        rsq = float(res.rsquared) if hasattr(res, "rsquared") else None

        return {
            "variable": varname,
            "coef": coef,
            "std_err": bse,
            "t": tval,
            "p_value": pval,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "nobs": nobs,
            "r_squared": rsq
        }

    # Extract for coder-rated femininity if present
    if isinstance(model_output, dict) and "model_coder" in model_output:
        try:
            results["coder"] = _extract_from_result(model_output["model_coder"], "Femininity_Coder_c")
        except Exception:
            results["coder"] = None

    # Extract for MTurk-rated femininity if present
    if isinstance(model_output, dict) and "model_mturk" in model_output:
        try:
            results["mturk"] = _extract_from_result(model_output["model_mturk"], "Femininity_MTURK_c")
        except Exception:
            results["mturk"] = None

    # Build a brief description interpreting the primary result (coder). If coder is missing, mention that.
    if results["coder"] is None:
        description = ("No estimate for coder-rated femininity (Femininity_Coder_c) could be extracted from the model output. "
                       "Ensure the model object contains that parameter.")
    else:
        r = results["coder"]
        # Interpret sign and statistical significance
        sign_desc = "negative" if r["coef"] < 0 else ("positive" if r["coef"] > 0 else "null (≈0)")
        sig_desc = "statistically significant" if (r["p_value"] is not None and r["p_value"] < 0.05) else "not statistically significant"
        description = (
            f"Primary estimate (coder-rated femininity, Femininity_Coder_c): coefficient = {r['coef']:.4g}, "
            f"SE = {r['std_err']:.4g}, p = {r['p_value']:.4g}, 95% CI = [{r['ci_95_lower']:.4g}, {r['ci_95_upper']:.4g}] "
            f"based on {r['nobs']} observations. The coefficient is {sign_desc} and {sig_desc}. "
            "A negative and statistically significant coefficient would support the hypothesis that more feminine names "
            "are associated with lower log(1+deaths) (i.e., fewer fatalities), consistent with less precautionary behavior. "
        )
        # Add note about robustness (MTurk) if present
        if results["mturk"] is not None:
            m = results["mturk"]
            m_sign = "negative" if m["coef"] < 0 else ("positive" if m["coef"] > 0 else "null (≈0)")
            m_sig = "statistically significant" if (m["p_value"] is not None and m["p_value"] < 0.05) else "not statistically significant"
            description += (f"Robustness (MTurk-rated femininity): coefficient = {m['coef']:.4g}, p = {m['p_value']:.4g} "
                            f"({m_sign}, {m_sig}), 95% CI = [{m['ci_95_lower']:.4g}, {m['ci_95_upper']:.4g}].")
        else:
            description += "No MTurk-rated robustness estimate was available in the provided output."

    return {"object": results, "description": description}