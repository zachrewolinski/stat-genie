def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of hurricane name femininity (NameFem)
    and binary gender (GenderBinary) from a statsmodels fitted-model output dict.

    Input:
        model_output: dict-like with keys 'main', 'robust_damage', 'gender_only'
                      values are statsmodels RegressionResultsWrapper objects.

    Output:
        dict with keys:
          - "object": dict with extracted numeric statistics for relevant coefficients
          - "description": short interpretation linking the statistics to the task hypothesis
    """
    import math
    import numpy as np

    def safe_get_result(res, varname):
        """Return dict with coefficient, se, t, p, 95% CI, and percent change for log outcome."""
        out = {"variable": varname, "present": False}
        if res is None:
            return out
        try:
            params = res.params
        except Exception:
            return out
        if varname not in params.index:
            return out
        out["present"] = True
        coef = float(params[varname])
        # standard error, tvalue, pvalue
        se = float(res.bse[varname]) if varname in res.bse.index else None
        t = float(res.tvalues[varname]) if varname in res.tvalues.index else None
        p = float(res.pvalues[varname]) if varname in res.pvalues.index else None
        # confidence interval
        try:
            ci = res.conf_int().loc[varname].tolist()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            # fallback if conf_int doesn't have index
            try:
                ci_array = res.conf_int().values
                # try to find row corresponding to varname via params index position
                idx = list(res.params.index).index(varname)
                ci_lower, ci_upper = float(ci_array[idx, 0]), float(ci_array[idx, 1])
            except Exception:
                ci_lower, ci_upper = None, None
        # For log outcome, approximate percent change in (outcome) per 1-unit change in predictor:
        # percent_change = (exp(coef) - 1) * 100
        try:
            pct_change = (math.exp(coef) - 1) * 100.0
        except Exception:
            pct_change = None
        # assemble
        out.update({
            "coef": coef,
            "se": se,
            "t": t,
            "p": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "percent_change_approx": pct_change
        })
        return out

    # Prepare results container
    results = {}

    # Extract from main model: NameFem and GenderBinary (effect on LogDeaths)
    res_main = model_output.get('main')
    results['main_NameFem'] = safe_get_result(res_main, 'NameFem')
    results['main_GenderBinary'] = safe_get_result(res_main, 'GenderBinary')

    # Extract from robustness using damages outcome (LogNDAM15): same predictor NameFem
    res_damage = model_output.get('robust_damage')
    results['damage_NameFem'] = safe_get_result(res_damage, 'NameFem')

    # Extract from gender-only model: GenderBinary effect on LogDeaths
    res_genderonly = model_output.get('gender_only')
    results['genderonly_GenderBinary'] = safe_get_result(res_genderonly, 'GenderBinary')

    # Build human-readable interpretation for the main effect
    interp_lines = []
    main_nf = results.get('main_NameFem', {})
    if main_nf.get("present"):
        coef = main_nf["coef"]
        p = main_nf["p"]
        ci_l = main_nf["ci_lower"]
        ci_u = main_nf["ci_upper"]
        pct = main_nf["percent_change_approx"]
        direction = "positive (higher femininity → higher log fatalities)" if coef > 0 else "negative (higher femininity → lower log fatalities)"
        signif = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p ≥ 0.05)"
        interp_lines.append(
            f"Main model — NameFem: coef={coef:.4g}, se={main_nf.get('se'):.4g} , p={p:.4g}; 95% CI [{ci_l:.4g}, {ci_u:.4g}]. "
            f"Direction: {direction}. {signif}."
        )
        if pct is not None:
            interp_lines.append(f"Approx. percent change in (alldeaths+1) per 1-unit increase in NameFem: {pct:.3g}%.")
    else:
        interp_lines.append("Main model — NameFem: variable not present in the fitted model output.")

    # Add interpretation for gender-only model
    go = results.get('genderonly_GenderBinary', {})
    if go.get("present"):
        coef = go["coef"]
        p = go["p"]
        ci_l = go["ci_lower"]
        ci_u = go["ci_upper"]
        direction = "female-named storms associated with higher log fatalities" if coef > 0 else "female-named storms associated with lower log fatalities"
        signif = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p ≥ 0.05)"
        interp_lines.append(
            f"Gender-only model — GenderBinary: coef={coef:.4g}, p={p:.4g}; 95% CI [{ci_l:.4g}, {ci_u:.4g}]. "
            f"Interpretation: {direction}. {signif}."
        )
    else:
        interp_lines.append("Gender-only model — GenderBinary: variable not present in the fitted model output.")

    # Brief conclusion relative to the hypothesis
    conclusion = ("Conclusion (data-driven): Inspect the sign and significance of NameFem in the main model. "
                  "If NameFem coef is positive and statistically significant, this is consistent with the hypothesis "
                  "that more feminine names are associated with higher fatalities (interpreted as less precaution). "
                  "If negative and significant, it contradicts the hypothesis. If not significant, the analysis "
                  "does not provide evidence for the hypothesized effect.")
    interp_lines.append(conclusion)

    return {
        "object": results,
        "description": "Extracted coefficient estimates, standard errors, t-values, p-values, 95% CIs, and approximate percent-change for NameFem and GenderBinary from the provided models. "
                       "Also provides a short interpretation stating whether the estimates support the hypothesis (direction and statistical significance). "
                       "See 'object' for numeric results.",
        "interpretation_lines": interp_lines
    }