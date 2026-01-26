def extract_final_answer(model_output):
    """
    Extract key statistics about the femininity variable from the fitted model objects
    returned by the modeling function.

    Parameters
    ----------
    model_output : dict
        Dictionary expected to contain keys:
          - 'main_masfem_on_deaths' : statsmodels RegressionResultsWrapper or error string
          - 'robust_mturk_on_deaths' : statsmodels RegressionResultsWrapper or error string
          - 'masfem_on_damage' : statsmodels RegressionResultsWrapper or error string

    Returns
    -------
    dict
        {
          "object": {
              "main_masfem_on_deaths": { ... extracted stats ... } or {"error": "..."},
              "robust_mturk_on_deaths": { ... } or {"error": "..."},
              "masfem_on_damage": { ... } or {"error": "..."}
          },
          "description": "<plain-English interpretation of the main and robustness results>"
        }
    """
    import math

    def summarize_result(res, varname):
        """Return a dict of extracted statistics for variable `varname` from statsmodels result `res`."""
        try:
            # Ensure the parameter exists
            if varname not in res.params.index:
                return {"error": f"Variable '{varname}' not found in model parameters."}

            coef = float(res.params[varname])
            se = float(res.bse[varname])
            tval = float(res.tvalues[varname])
            pval = float(res.pvalues[varname])
            ci = res.conf_int().loc[varname]
            ci_low = float(ci[0])
            ci_high = float(ci[1])
            # Observations and R-squared if available
            nobs = int(res.nobs) if hasattr(res, "nobs") else None
            rsq = float(res.rsquared) if hasattr(res, "rsquared") else None

            # Because the dependent variable is log1p(deaths), approximate percent change:
            # exp(coef) - 1 gives approximate multiplicative change in (1 + deaths).
            approx_pct_change = (math.exp(coef) - 1) * 100.0

            return {
                "variable": varname,
                "coef": coef,
                "std_err": se,
                "t_value": tval,
                "p_value": pval,
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "nobs": nobs,
                "r_squared": rsq,
                "approx_pct_change_in_1_plus_deaths": approx_pct_change
            }
        except Exception as e:
            return {"error": f"Exception while summarizing model: {str(e)}"}

    results_summary = {}

    # Map model keys to the variable name to extract
    mapping = {
        "main_masfem_on_deaths": "masfem_std",
        "robust_mturk_on_deaths": "masfem_mturk_std",
        "masfem_on_damage": "masfem_std"
    }

    for key, varname in mapping.items():
        val = model_output.get(key, None)
        if isinstance(val, str):
            # The caller sometimes returns an error string in place of a model
            results_summary[key] = {"error": val}
        elif val is None:
            results_summary[key] = {"error": "Model output missing or None"}
        else:
            # Attempt to summarize the statsmodels results object
            results_summary[key] = summarize_result(val, varname)

    # Build a concise interpretation focused on the main model (and note robustness)
    def interpret_entry(entry):
        if not isinstance(entry, dict):
            return "No information."
        if "error" in entry:
            return entry["error"]
        coef = entry["coef"]
        p = entry["p_value"]
        pct = entry["approx_pct_change_in_1_plus_deaths"]
        direction = "negative (higher femininity → fewer deaths)" if coef < 0 else "positive (higher femininity → more deaths)"
        signif = "statistically significant (p < 0.05)" if p < 0.05 else ("borderline (0.05 ≤ p < 0.10)" if p < 0.10 else "not statistically significant (p ≥ 0.10)")
        return f"Coefficient = {coef:.4f} ({direction}), {signif}. Approx. {pct:.1f}% change in (1 + deaths) per 1 SD increase in femininity."

    main_interp = interpret_entry(results_summary.get("main_masfem_on_deaths", {}))
    mturk_interp = interpret_entry(results_summary.get("robust_mturk_on_deaths", {}))
    damage_interp = interpret_entry(results_summary.get("masfem_on_damage", {}))

    description_lines = [
        "Main model (LogDeaths ~ masfem_std + controls): " + main_interp,
        "Robustness A (alternative femininity measure masfem_mturk_std): " + mturk_interp,
        "Robustness B (outcome = LogDamage): " + damage_interp,
        "Notes: Because the outcome is log1p(deaths), exp(coef)-1 gives the approximate proportional change in (1 + deaths).",
        "A negative and statistically significant coefficient would support the hypothesis that more feminine names lead to fewer fatalities (fewer precautionary measures)."
    ]
    description = " ".join(description_lines)

    return {"object": results_summary, "description": description}