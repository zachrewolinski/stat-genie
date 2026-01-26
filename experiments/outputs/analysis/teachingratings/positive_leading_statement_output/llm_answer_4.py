def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, and 95% CI for the 'beauty_z'
    variable from the provided model_output dict (expects keys 'clustered' and 'fe').

    Returns:
      {
        "object": {
          "clustered": { "coef": ..., "se": ..., "t": ..., "p": ..., "ci_lower": ..., "ci_upper": ..., "significant": True/False },
          "fe":        { ... same fields ... }
        },
        "description": "<brief plain-language interpretation>"
      }
    """
    import math

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing 'clustered' and 'fe' results.")

    required_keys = ['clustered', 'fe']
    for k in required_keys:
        if k not in model_output:
            raise ValueError(f"model_output is missing required key: '{k}'")

    results = {}
    for key in required_keys:
        res = model_output[key]
        # Ensure the result object looks like a statsmodels RegressionResultsWrapper
        if not hasattr(res, 'params'):
            raise ValueError(f"The object under '{key}' does not look like a fitted statsmodels result (missing .params)")

        param_index = list(res.params.index) if hasattr(res.params, 'index') else list(res.params.keys())
        if 'beauty_z' not in param_index:
            raise ValueError(f"'beauty_z' not found in model parameters for '{key}'")

        # extract basic stats
        coef = float(res.params['beauty_z'])
        se = float(res.bse['beauty_z']) if hasattr(res, 'bse') else float('nan')
        tstat = float(res.tvalues['beauty_z']) if hasattr(res, 'tvalues') else float('nan')
        pval = float(res.pvalues['beauty_z']) if hasattr(res, 'pvalues') else float('nan')

        # confidence interval extraction robust to DataFrame/ndarray return types
        try:
            ci = res.conf_int()
            # if ci is a DataFrame (pandas)
            if hasattr(ci, 'loc'):
                ci_lower, ci_upper = float(ci.loc['beauty_z', 0]), float(ci.loc['beauty_z', 1])
            else:
                # assume numpy array in same order as params
                idx = param_index.index('beauty_z')
                ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
        except Exception:
            # fallback to using coef +/- 1.96*se if CI extraction fails
            if not math.isnan(se):
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                ci_lower = float('nan')
                ci_upper = float('nan')

        significant = (pval < 0.05)

        # Round numbers for readability
        results[key] = {
            "coef": round(coef, 4),
            "se": round(se, 4) if not math.isnan(se) else se,
            "t": round(tstat, 3) if not math.isnan(tstat) else tstat,
            "p": round(pval, 4) if not math.isnan(pval) else pval,
            "ci_lower": round(ci_lower, 4) if not math.isnan(ci_lower) else ci_lower,
            "ci_upper": round(ci_upper, 4) if not math.isnan(ci_upper) else ci_upper,
            "significant": bool(significant)
        }

    # Build a concise plain-language description
    desc_lines = []
    desc_lines.append("Effect of instructor beauty (beauty_z, standardized) on student evaluations (eval):")
    for key in required_keys:
        r = results[key]
        sig_text = "statistically significant (p < 0.05)" if r["significant"] else "not statistically significant (p >= 0.05)"
        desc_lines.append(
            f"- {key}: coefficient = {r['coef']}, SE = {r['se']}, 95% CI = [{r['ci_lower']}, {r['ci_upper']}], p = {r['p']} -> {sig_text}."
        )
        desc_lines.append(
            f"  Interpretation: a 1 SD increase in rated beauty is associated with a change of {r['coef']} points in the evaluation score (holding controls constant)."
        )

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }