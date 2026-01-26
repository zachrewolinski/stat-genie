def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and odds-ratios from the
    provided statsmodels BinaryResultsWrapper objects and produces a concise
    interpretation about how reliance on the majority develops with age and
    whether that developmental trajectory differs across cultures.

    Input:
      model_output: dict-like with keys (expected) 'majority_model', 'social_model', 'minority_model'
                    each value a statsmodels.discrete.discrete_model.BinaryResultsWrapper

    Returns:
      dict with keys:
        - "object": dict with per-model coefficient tables (as nested dicts), and a small
                    summary object for the majority_model highlighting age effects and interactions
        - "description": a short human-readable interpretation about development of majority reliance
    """
    import pandas as pd
    import numpy as np

    out = {}
    summary_tables = {}

    # Helper to build table from a statsmodels result
    def build_table(res):
        # params, se, pvalues, conf_int
        params = res.params
        se = res.bse
        pvals = res.pvalues
        ci = res.conf_int()
        # Some versions return DataFrame for conf_int; ensure columns named
        try:
            ci_low = ci.iloc[:, 0]
            ci_high = ci.iloc[:, 1]
        except Exception:
            # Fallback if conf_int returns ndarray
            ci_low = pd.Series(ci[:, 0], index=params.index)
            ci_high = pd.Series(ci[:, 1], index=params.index)

        zvals = params / se
        or_vals = np.exp(params)
        or_ci_low = np.exp(ci_low)
        or_ci_high = np.exp(ci_high)

        df = pd.DataFrame({
            'coef': params,
            'se': se,
            'z': zvals,
            'pvalue': pvals,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'odds_ratio': or_vals,
            'or_ci_low': or_ci_low,
            'or_ci_high': or_ci_high
        })
        # Round for readability
        df_rounded = df.round(4)
        return df_rounded

    # Process expected models
    for key in ['majority_model', 'social_model', 'minority_model']:
        if key in model_output and model_output[key] is not None:
            res = model_output[key]
            try:
                table = build_table(res)
                # Convert to dict for safe JSON-like return; keep index (parameter names)
                summary_tables[key] = table.to_dict(orient='index')
            except Exception as e:
                summary_tables[key] = {"error": f"Failed to extract table: {str(e)}"}
        else:
            summary_tables[key] = {"error": "Model not provided in model_output"}

    out['coef_tables'] = summary_tables

    # Focused interpretation for the majority_model (development of majority preference)
    interpretation = {}
    if 'majority_model' in model_output and model_output['majority_model'] is not None:
        res = model_output['majority_model']
        params = res.params
        pvals = res.pvalues

        # Check for linear and quadratic age terms
        age_terms = {}
        for term in ['age_centered', 'age_centered2']:
            if term in params.index:
                age_terms[term] = {
                    'coef': float(params[term]),
                    'pvalue': float(pvals[term]),
                    'odds_ratio': float(np.exp(params[term])),
                }
            else:
                age_terms[term] = None

        # Find interaction terms age_x_culture_*
        interaction_mask = [name for name in params.index if name.startswith('age_x_culture_')]
        interactions = {}
        significant_interactions = []
        for name in interaction_mask:
            coef = float(params[name])
            p = float(pvals[name])
            interactions[name] = {'coef': coef, 'pvalue': p, 'odds_ratio': float(np.exp(coef))}
            if p < 0.05:
                significant_interactions.append((name, interactions[name]))

        # Determine overall evidence
        baseline_effect_sig = False
        baseline_desc = ""
        if age_terms['age_centered'] is not None:
            p_lin = age_terms['age_centered']['pvalue']
            coef_lin = age_terms['age_centered']['coef']
            if p_lin < 0.05:
                baseline_effect_sig = True
                direction = 'increase' if coef_lin > 0 else 'decrease'
                baseline_desc = f"In the baseline culture (culture 1), there is a statistically significant linear effect of age (coef={coef_lin:.4f}, p={p_lin:.3f}) indicating a {direction} in the log-odds of choosing the majority with increasing age (OR={age_terms['age_centered']['odds_ratio']:.3f} per unit of age_centered)."
            else:
                baseline_desc = f"In the baseline culture (culture 1), the linear age effect is not statistically significant (coef={coef_lin:.4f}, p={p_lin:.3f})."

        # Quadratic term comment
        quad_desc = ""
        if age_terms['age_centered2'] is not None:
            p_quad = age_terms['age_centered2']['pvalue']
            coef_quad = age_terms['age_centered2']['coef']
            if p_quad < 0.05:
                quad_desc = f" The quadratic age term is significant (coef={coef_quad:.4f}, p={p_quad:.3f}), indicating a nonlinear (curved) developmental trajectory."
            else:
                quad_desc = f" The quadratic age term is not significant (coef={coef_quad:.4f}, p={p_quad:.3f})."

        # Interactions
        if len(interaction_mask) == 0:
            interaction_desc = "No culture-by-age interaction terms are present in the model."
        else:
            if len(significant_interactions) > 0:
                names_list = ", ".join([f"{n} (p={v['pvalue']:.3f})" for n, v in significant_interactions])
                interaction_desc = f"Some age-by-culture interaction terms are statistically significant: {names_list}. This indicates that the developmental trajectory of majority preference differs between those cultures and the baseline culture."
            else:
                interaction_desc = "No age-by-culture interaction terms reach conventional significance (p < .05), so there is no strong evidence that developmental trajectories differ across cultures."

        interpretation['age_terms'] = age_terms
        interpretation['interactions'] = interactions
        interpretation['significant_interactions'] = {n: v for n, v in significant_interactions}
        interpretation['summary_text'] = baseline_desc + quad_desc + " " + interaction_desc

    else:
        interpretation['error'] = "majority_model not available for interpretation."

    out['majority_interpretation'] = interpretation

    # Compose a short human-readable description
    if 'majority_model' in interpretation and 'summary_text' in interpretation:
        desc = (
            "Summary interpretation (majority choice):\n"
            f"{interpretation['summary_text']}\n\n"
            "Notes: Coefficients are on the log-odds scale. Odds ratios (OR) shown are exp(coef); "
            "OR > 1 indicates higher odds of choosing the majority for a one-unit increase in the predictor. "
            "Interaction coefficients describe how the linear age slope differs in that culture compared to the baseline (culture 1). "
            "If you need model-based predicted probabilities (e.g., plotted developmental curves per culture), "
            "I can calculate them using the original data or by simulating across age values."
        )
    else:
        desc = "Could not produce an interpretation because majority_model was not provided."

    return {"object": out, "description": desc}