def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of hurricane name femininity on fatalities
    from the fitted statsmodels RegressionResultsWrapper objects contained in model_output.

    Expects model_output to be a dict with keys (some or all):
      - 'alldeaths_masfem_scaled' : main model (masfem_scaled)
      - 'alldeaths_gender_female'  : alternative binary gender indicator model (gender_female)
      - 'ndam15_masfem_scaled'    : robustness model using property damage (masfem_scaled)

    Returns a dict with:
      - "object": a dict of extracted numeric results for each available model
      - "description": a short interpretation describing direction, significance,
                       and whether the results support the hypothesis.
    """
    import numpy as np

    def summarize_result(res, term):
        if res is None:
            return None
        # Ensure term exists
        if term not in res.params.index:
            return None
        coef = float(res.params[term])
        se = float(res.bse[term]) if term in res.bse.index else None
        t = float(res.tvalues[term]) if term in res.tvalues.index else None
        p = float(res.pvalues[term]) if term in res.pvalues.index else None
        try:
            ci = res.conf_int(alpha=0.05).loc[term].values.astype(float)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower, ci_upper = None, None
        # For log(1 + deaths) DV, convert coef to approximate percent change in (1 + deaths)
        try:
            pct_change = (np.exp(coef) - 1.0) * 100.0
            pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0 if ci_lower is not None else None
            pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0 if ci_upper is not None else None
        except Exception:
            pct_change = pct_ci_lower = pct_ci_upper = None

        return {
            "term": term,
            "coef": coef,
            "std_err": se,
            "t": t,
            "p_value": p,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "approx_pct_change_in_1_plus_outcome": pct_change,
            "pct_change_ci_lower": pct_ci_lower,
            "pct_change_ci_upper": pct_ci_upper,
            "nobs": int(res.nobs) if hasattr(res, "nobs") else None
        }

    results_summary = {}

    # Main model: masfem_scaled
    main_res = model_output.get('alldeaths_masfem_scaled')
    results_summary['main_masfem_scaled'] = summarize_result(main_res, 'masfem_scaled')

    # Alternative: gender_female
    alt_res = model_output.get('alldeaths_gender_female')
    results_summary['alt_gender_female'] = summarize_result(alt_res, 'gender_female')

    # Robustness: property damage (masfem_scaled)
    rob_res = model_output.get('ndam15_masfem_scaled')
    results_summary['robustness_masfem_scaled'] = summarize_result(rob_res, 'masfem_scaled')

    # Build a brief interpretation based on available summaries
    def interpret_entry(entry):
        if entry is None:
            return "not available"
        coef = entry["coef"]
        p = entry["p_value"]
        sign = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
        signif = "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p >= 0.05)"
        pct = entry["approx_pct_change_in_1_plus_outcome"]
        pct_text = f"≈ {pct:.1f}% change in (1 + outcome)" if pct is not None else "percent change N/A"
        return f"{sign} coefficient ({coef:.4f}), {signif}; {pct_text}."

    interpretations = {
        "main": interpret_entry(results_summary['main_masfem_scaled']),
        "alternative": interpret_entry(results_summary['alt_gender_female']),
        "robustness": interpret_entry(results_summary['robustness_masfem_scaled'])
    }

    # Overall conclusion logic
    def support_conclusion(entry):
        if entry is None:
            return None
        coef = entry["coef"]
        p = entry["p_value"]
        # Hypothesis: more feminine names -> more fatalities (positive coef supports)
        if p is not None and p < 0.05:
            return coef > 0  # True if supports hypothesis
        else:
            return None  # inconclusive

    support_main = support_conclusion(results_summary['main_masfem_scaled'])
    support_alt = support_conclusion(results_summary['alt_gender_female'])
    support_rob = support_conclusion(results_summary['robustness_masfem_scaled'])

    # Count supporting models among those with clear significance
    supports = sum(1 for x in [support_main, support_alt, support_rob] if x is True)
    contradicts = sum(1 for x in [support_main, support_alt, support_rob] if x is False)
    significant_count = sum(1 for x in [support_main, support_alt, support_rob] if x is not None)

    if significant_count == 0:
        overall = ("No clear evidence either way: none of the available estimates for the "
                   "name-femininity effect are statistically significant at p < 0.05.")
    else:
        if supports > contradicts:
            overall = (f"Overall, among models with statistically significant estimates ({significant_count}), "
                       f"{supports} support the hypothesis (more feminine names -> higher fatalities) "
                       f"and {contradicts} contradict it. This leans toward supporting the hypothesis.")
        elif contradicts > supports:
            overall = (f"Overall, among models with statistically significant estimates ({significant_count}), "
                       f"{contradicts} contradict the hypothesis and {supports} support it. This leans against the hypothesis.")
        else:
            overall = (f"Mixed evidence among the {significant_count} statistically significant models: "
                       "some support and some contradict the hypothesis.")

    description = (
        "Extracted coefficients, standard errors, 95% CIs, p-values, and approximate percent-change "
        "interpretations for the key predictors. Interpretation by model:\n"
        f"- Main (masfem_scaled): {interpretations['main']}\n"
        f"- Alternative (gender_female): {interpretations['alternative']}\n"
        f"- Robustness (masfem_scaled on property damage): {interpretations['robustness']}\n\n"
        f"Overall conclusion: {overall}"
    )

    return {"object": results_summary, "description": description}