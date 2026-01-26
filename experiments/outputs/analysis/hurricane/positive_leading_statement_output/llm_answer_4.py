def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and an interpretable
    effect size for the two focal predictors (masfem_z and IsFemaleName) from the
    model_output dictionary produced by the modeling function.

    Returns:
      {
        "object": {
          "primary": { "masfem_z": {coef, se, p, ci_lower, ci_upper, exp_coef, pct_change, significant}, ... },
          "poisson": { ... } or None,
          "negbin":  { ... } or None
        },
        "description": "<short interpretation across models and final yes/no statement>"
      }
    """
    import numpy as np

    def summarize_result(res, varlist):
        """
        Summarize a statsmodels results object for variables in varlist.
        Returns dict mapping variable -> stats dict. If a variable is missing,
        it records None for that variable.
        """
        if res is None:
            return None

        out = {}
        # Obtain arrays or Series for params, bse, pvalues, conf_int
        try:
            params = res.params
        except Exception:
            params = None
        try:
            bse = res.bse
        except Exception:
            bse = None
        try:
            pvalues = res.pvalues
        except Exception:
            pvalues = None
        try:
            ci = res.conf_int()
        except Exception:
            ci = None

        # If ci is ndarray, try to map rows to index names
        # We'll rely on matching variable names from params.index if possible
        index = None
        try:
            index = params.index.tolist()
        except Exception:
            index = None

        for var in varlist:
            if params is None or var not in params.index:
                out[var] = None
                continue
            coef = float(params[var])
            se = float(bse[var]) if (bse is not None and var in bse.index) else None
            p = float(pvalues[var]) if (pvalues is not None and var in pvalues.index) else None

            # Confidence interval
            ci_lower = ci_upper = None
            try:
                # If conf_int returns DataFrame-like with index
                if hasattr(ci, 'loc'):
                    ci_lower = float(ci.loc[var, 0])
                    ci_upper = float(ci.loc[var, 1])
                else:
                    # ci may be ndarray; find row by matching order in params.index
                    if index is not None and var in index:
                        pos = index.index(var)
                        ci_lower = float(ci[pos, 0])
                        ci_upper = float(ci[pos, 1])
            except Exception:
                ci_lower = ci_upper = None

            # For log-scale models (OLS on log outcome, Poisson/NB on counts),
            # exponentiate coefficient to get multiplicative effect on outcome:
            try:
                exp_coef = float(np.exp(coef))
                # approximate percent change in outcome or 1+outcome for log model
                pct_change = float((np.exp(coef) - 1.0) * 100.0)
            except Exception:
                exp_coef = None
                pct_change = None

            significant = None
            try:
                significant = (p is not None) and (p < 0.05)
            except Exception:
                significant = None

            out[var] = {
                "coef": coef,
                "se": se,
                "pvalue": p,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "exp_coef": exp_coef,
                "pct_change": pct_change,
                "significant_at_0.05": bool(significant)
            }
        return out

    # Prepare
    masfem = "masfem_z"
    isf = "IsFemaleName"
    focal_vars = [masfem, isf]

    results_summary = {}
    # Primary OLS on log_alldeaths
    ols_res = model_output.get('ols_log_deaths')
    results_summary['primary_ols_log_deaths'] = summarize_result(ols_res, focal_vars)

    # Robustness: Poisson on raw counts
    pois_res = model_output.get('poisson_deaths')
    results_summary['poisson_deaths'] = summarize_result(pois_res, focal_vars)

    # Robustness: Negative binomial
    nb_res = model_output.get('negbin_deaths')
    results_summary['negbin_deaths'] = summarize_result(nb_res, focal_vars)

    # Optionally include damage model if present
    dmg_res = model_output.get('ols_log_damage')
    if dmg_res is not None:
        results_summary['ols_log_damage'] = summarize_result(dmg_res, focal_vars)
    else:
        results_summary['ols_log_damage'] = None

    # Short textual interpretation focusing on masfem_z (primary test of hypothesis)
    interpretation_lines = []
    try:
        primary = results_summary.get('primary_ols_log_deaths')
        if primary and primary.get(masfem) is not None:
            s = primary[masfem]
            line = (
                f"Primary (OLS on log(1+fatalities)): masfem_z coef = {s['coef']:.4f}, "
                f"SE = {s['se']:.4f}, p = {s['pvalue']:.4g}, 95% CI = [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]. "
            )
            if s['pct_change'] is not None:
                line += f"Interpretation: a 1 SD increase in perceived femininity is associated with ≈{s['pct_change']:.1f}% change in (1+fatalities). "
            if s['significant_at_0.05']:
                line += "This effect is statistically significant (p < 0.05)."
            else:
                line += "This effect is NOT statistically significant (p ≥ 0.05)."
            interpretation_lines.append(line)
        else:
            interpretation_lines.append("Primary OLS result for masfem_z not available.")
    except Exception as e:
        interpretation_lines.append(f"Could not interpret primary OLS result due to error: {e}")

    # Summarize robustness consistency
    def concise_summary(role, sumdict):
        if sumdict is None:
            return f"{role}: model missing."
        v = sumdict.get(masfem)
        if v is None:
            return f"{role}: masfem_z not estimated."
        sign = "positive" if v['coef'] > 0 else "negative" if v['coef'] < 0 else "zero"
        sig = "significant" if v['significant_at_0.05'] else "not significant"
        return f"{role}: masfem_z coef={v['coef']:.4f} ({sign}), p={v['pvalue']:.4g} ({sig})."

    interpretation_lines.append(concise_summary("Poisson", results_summary.get('poisson_deaths')))
    interpretation_lines.append(concise_summary("NegBin", results_summary.get('negbin_deaths')))
    if results_summary.get('ols_log_damage') is not None:
        interpretation_lines.append(concise_summary("OLS on log damage", results_summary.get('ols_log_damage')))

    # Final yes/no judgment about the hypothesis:
    # We treat the primary OLS as the main test: require positive coef and p<0.05 to say "Yes (supported)".
    final_judgment = "inconclusive"
    try:
        p = primary[masfem]['pvalue']
        coef = primary[masfem]['coef']
        if (p is not None) and (coef is not None):
            if (coef > 0) and (p < 0.05):
                final_judgment = "yes"
            elif (coef <= 0) and (p < 0.05):
                final_judgment = "no"
            else:
                final_judgment = "no (not statistically significant)"
        else:
            final_judgment = "inconclusive (missing stats)"
    except Exception:
        final_judgment = "inconclusive (error inspecting primary result)"

    interpretation_lines.append(f"Final judgment: {final_judgment}. (Hypothesis: more feminine names -> fewer precautions -> more fatalities)")

    description = " ".join(interpretation_lines)

    return {
        "object": results_summary,
        "description": description
    }