def extract_final_answer(model_output):
    """
    Extract statistics for the masculinity-femininity predictor from the provided model output.
    Returns a dictionary with keys:
      - "object": dict with extracted numeric results for the continuous (masfem_z) and
                  binary (female_name) specifications plus a brief conclusion.
      - "description": short plain-language interpretation of those results in the context
                       of the hypothesis.

    The function accepts either:
      - a dict-like object containing keys 'model_masfem_continuous' and 'model_gender_binary'
        mapping to statsmodels RegressionResultsWrapper objects, OR
      - a single RegressionResultsWrapper (in which case it will try to extract masfem_z).
    """
    import numpy as np
    from types import SimpleNamespace

    def _extract_from_result(res, var):
        """Safely extract coef, se, pvalue, 95% CI for var from a statsmodels result object.
           Returns None if var not in the model."""
        try:
            params = res.params
        except Exception:
            return None
        if var not in params.index:
            return None
        coef = float(params[var])
        se = float(res.bse[var])
        p = float(res.pvalues[var])
        try:
            ci = res.conf_int(alpha=0.05).loc[var].tolist()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            # fallback: approximate CI using coef +/- 1.96*se
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        # For interpretation on the log scale, compute percent change approx: 100*(exp(coef)-1)
        pct_change = (np.exp(coef) - 1.0) * 100.0
        pct_change_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0
        pct_change_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0
        return {
            'coef': coef,
            'se': se,
            'pvalue': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'approx_percent_change_in_deaths': pct_change,
            'pct_change_ci_lower': pct_change_ci_lower,
            'pct_change_ci_upper': pct_change_ci_upper
        }

    # Normalize input to a dict-like container with expected names
    models = {}
    if isinstance(model_output, dict):
        models = model_output
    else:
        # single result: wrap it
        models = {'single_model': model_output}

    # Try to get the two expected models (continuous and binary). Accept some name variants.
    model_a = models.get('model_masfem_continuous') or models.get('model_masfem') or models.get('single_model')
    model_b = models.get('model_gender_binary') or models.get('model_female_binary') or None

    res_a = None
    res_b = None
    if model_a is not None:
        res_a = _extract_from_result(model_a, 'masfem_z')
    if model_b is None and isinstance(models, dict):
        # maybe the single model contains female_name instead; try extracting from model_a if present
        if model_a is not None:
            res_b = _extract_from_result(model_a, 'female_name')
    else:
        if model_b is not None:
            res_b = _extract_from_result(model_b, 'female_name')

    conclusion_lines = []
    # Evaluate evidence for hypothesis: "more feminine names -> fewer precautions -> more deaths"
    # That predicts a positive coefficient on masfem (masfem_z or female_name).
    def interpret(res_entry, var_label):
        if res_entry is None:
            return f"No estimate for {var_label} was found in the provided models."
        coef = res_entry['coef']
        p = res_entry['pvalue']
        direction = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
        sig = p < 0.05
        if sig and coef > 0:
            return (f"{var_label}: estimated coefficient = {coef:.4f} (95% CI [{res_entry['ci_lower']:.4f}, {res_entry['ci_upper']:.4f}]), "
                    f"p = {p:.3f}. Significant positive effect — consistent with the hypothesis: "
                    f"~{res_entry['approx_percent_change_in_deaths']:.1f}% change in deaths per 1-unit increase (log scale).")
        elif sig and coef <= 0:
            return (f"{var_label}: estimated coefficient = {coef:.4f} (95% CI [{res_entry['ci_lower']:.4f}, {res_entry['ci_upper']:.4f}]), "
                    f"p = {p:.3f}. Significant effect but in the opposite direction to the hypothesis.")
        else:
            return (f"{var_label}: estimated coefficient = {coef:.4f} (95% CI [{res_entry['ci_lower']:.4f}, {res_entry['ci_upper']:.4f}]), "
                    f"p = {p:.3f}. Not statistically significant at alpha=0.05 — insufficient evidence to support the hypothesis.")

    conclusion_lines.append(interpret(res_a, 'masfem_z (continuous)') if res_a is not None else "masfem_z (continuous): not available.")
    conclusion_lines.append(interpret(res_b, 'female_name (binary)') if res_b is not None else "female_name (binary): not available.")

    # Final synthesis: if either specification shows a significant positive effect, we say the model provides evidence.
    evidence = False
    evidence_reasons = []
    for r, name in ((res_a, 'masfem_z'), (res_b, 'female_name')):
        if r is None:
            continue
        if (r['pvalue'] < 0.05) and (r['coef'] > 0):
            evidence = True
            evidence_reasons.append(name)
    if evidence:
        final_statement = ("There is evidence consistent with the hypothesis in the model(s): " +
                           ", ".join(evidence_reasons) + ".")
    else:
        final_statement = "No robust evidence supporting the hypothesis was found in the provided model(s)."

    result_object = {
        'masfem_z': res_a,
        'female_name': res_b,
        'final_conclusion': final_statement
    }

    description = ("Extracted coefficients, standard errors, p-values, and 95% CIs for the name-gender predictors. "
                   "Positive and statistically significant coefficients support the hypothesis that more-feminine names "
                   "are associated with more fatalities (interpreted as fewer precautions). The 'final_conclusion' field "
                   "summarizes whether any model specification provided such evidence at alpha=0.05.")

    return {'object': result_object, 'description': "\n".join(conclusion_lines) + "\n\n" + description}