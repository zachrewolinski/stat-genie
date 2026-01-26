def extract_final_answer(model_output):
    """
    Extract statistics for the genus indicator coefficients from a fitted statsmodels GLM result
    (preferring cluster-robust results if available). Returns a dictionary with keys:
      - "object": a dict with extracted numeric results (coef, se, p, 95% CI, odds ratio and OR CI)
                  for genus_Homo, genus_Pongo, genus_Papio, and a short boolean/summary conclusion
                  about whether Homo sapiens show higher AMTL than Pan after controls.
      - "description": a human-readable explanation of the extracted statistics and the conclusion.
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary as returned by the modeling function.")

    # Prefer clustered robust results if provided
    res = model_output.get('glm_result_clustered') or model_output.get('glm_result')
    if res is None:
        raise ValueError("No GLM result found in model_output under keys 'glm_result' or 'glm_result_clustered'.")

    # Extract parameter table components
    try:
        params = res.params            # pandas Series
        bse = res.bse                  # pandas Series or array-like
        pvalues = res.pvalues
        ci = res.conf_int()            # DataFrame with two columns (lower, upper)
    except Exception as e:
        raise RuntimeError(f"Unable to extract statistics from provided result object: {e}")

    def extract_for(name):
        if name not in params.index:
            return None
        coef = float(params[name])
        # bse and pvalues may be Series or array; access by name if possible
        try:
            se = float(bse[name])
        except Exception:
            # fallback: if bse is array-like aligned by position
            se = float(bse[list(params.index).index(name)])
        try:
            p = float(pvalues[name])
        except Exception:
            p = float(pvalues[list(params.index).index(name)])
        # confidence interval handling
        try:
            ci_row = ci.loc[name]
            ci_lower = float(ci_row.iloc[0])
            ci_upper = float(ci_row.iloc[1])
        except Exception:
            # fallback in case conf_int returned array-like
            idx = list(params.index).index(name)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
        odds_ratio = float(np.exp(coef))
        or_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
        return {
            'coef_logodds': coef,
            'se': se,
            'pvalue': p,
            'ci_95_logodds': (ci_lower, ci_upper),
            'odds_ratio': odds_ratio,
            'odds_ratio_95_ci': or_ci
        }

    gens = ['genus_Homo', 'genus_Pongo', 'genus_Papio']
    results = {g: extract_for(g) for g in gens}

    # Formulate conclusion for Homo vs Pan (Pan is reference)
    homo_stats = results.get('genus_Homo')
    if homo_stats is None:
        conclusion_bool = None
        conclusion_text = "The model does not contain a coefficient named 'genus_Homo'. Cannot conclude."
    else:
        coef = homo_stats['coef_logodds']
        p = homo_stats['pvalue']
        or_val = homo_stats['odds_ratio']
        # Decide significance at alpha = 0.05
        if p < 0.05:
            if coef > 0:
                conclusion_bool = True
                conclusion_text = (f"Yes — modern humans (Homo) have a statistically significantly higher frequency "
                                   f"of AMTL than Pan after controlling for age, sex-probability, and tooth class "
                                   f"(log-odds = {coef:.3f}, SE = {homo_stats['se']:.3f}, p = {p:.3f}; "
                                   f"OR = {or_val:.3f}, 95% CI for OR = "
                                   f"({homo_stats['odds_ratio_95_ci'][0]:.3f}, {homo_stats['odds_ratio_95_ci'][1]:.3f})).")
            else:
                conclusion_bool = False
                conclusion_text = (f"No — modern humans (Homo) have a statistically significantly lower frequency "
                                   f"of AMTL than Pan (log-odds = {coef:.3f}, p = {p:.3f}; OR = {or_val:.3f}).")
        else:
            conclusion_bool = False
            conclusion_text = (f"No statistically significant difference detected between Homo and Pan in AMTL "
                               f"after controls (log-odds = {coef:.3f}, SE = {homo_stats['se']:.3f}, p = {p:.3f}; "
                               f"OR = {or_val:.3f}, 95% CI for OR = "
                               f"({homo_stats['odds_ratio_95_ci'][0]:.3f}, {homo_stats['odds_ratio_95_ci'][1]:.3f})).")

    output_object = {
        'genus_stats': results,
        'conclusion_boolean_homo_higher_than_pan': conclusion_bool,
        'conclusion_text': conclusion_text,
        'notes': "Coefficients are log-odds relative to reference genus Pan; positive coef => higher AMTL odds."
    }

    description = ("Extracted coefficient, standard error, p-value, 95% CI on log-odds, and their exponentiated "
                   "odds-ratio equivalents for the genus indicator variables from the fitted GLM. The conclusion_text "
                   "reports whether Homo shows a significantly higher AMTL frequency than Pan after controlling "
                   "for age, sex probability, and tooth class using p<0.05 as the threshold.")

    return {"object": output_object, "description": description}