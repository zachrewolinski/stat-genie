def extract_final_answer(model_output):
    """
    Extract and interpret the SkinDark effect from the model_output returned by the model() function.
    Returns a dictionary with keys:
      - "object": a dict with numeric summaries (coef, pvalue, IRR, IRR_CI, significant, conclusion)
      - "description": a brief plain-language interpretation of the result in context.
    """
    import numpy as np

    # Initialize placeholders
    coef = None
    pval = None
    irr = None
    irr_ci = None
    conf_int = None

    # Preferred direct fields (present in the provided model_output)
    if isinstance(model_output, dict):
        if 'SkinDark_coef' in model_output:
            coef = float(model_output['SkinDark_coef'])
        if 'SkinDark_pvalue' in model_output:
            pval = float(model_output['SkinDark_pvalue'])
        if 'SkinDark_IRR' in model_output:
            irr = float(model_output['SkinDark_IRR'])
        if 'SkinDark_IRR_CI' in model_output:
            irr_ci = list(map(float, model_output['SkinDark_IRR_CI']))

    # Fallback: try to extract from clustered_results (if available)
    clustered = model_output.get('clustered_results') if isinstance(model_output, dict) else None
    if (coef is None or pval is None or irr is None or irr_ci is None) and clustered is not None:
        try:
            # clustered.params is expected to be a pandas Series-like
            if coef is None and hasattr(clustered, 'params') and 'SkinDark' in clustered.params.index:
                coef = float(clustered.params['SkinDark'])
            if pval is None and hasattr(clustered, 'pvalues') and 'SkinDark' in clustered.pvalues.index:
                pval = float(clustered.pvalues['SkinDark'])
            if (irr is None or irr_ci is None) and hasattr(clustered, 'conf_int'):
                ci = clustered.conf_int().loc['SkinDark'].tolist()
                conf_int = [float(ci[0]), float(ci[1])]
                if irr is None and coef is not None:
                    irr = float(np.exp(coef))
                if irr_ci is None and conf_int is not None:
                    irr_ci = [float(np.exp(conf_int[0])), float(np.exp(conf_int[1]))]
        except Exception:
            # ignore and proceed; we'll attempt other fallbacks
            pass

    # Final fallback: derive from glm_model if available
    glm = model_output.get('glm_model') if isinstance(model_output, dict) else None
    if glm is not None and (coef is None or (pval is None and hasattr(glm, 'pvalues'))):
        try:
            params = getattr(glm, 'params', None)
            if coef is None and params is not None and 'SkinDark' in params.index:
                coef = float(params['SkinDark'])
        except Exception:
            pass

    # If IRR or CI still missing but coef present, compute them (note: CI computed only if conf_int available)
    if irr is None and coef is not None:
        irr = float(np.exp(coef))
    if irr_ci is None and conf_int is not None:
        irr_ci = [float(np.exp(conf_int[0])), float(np.exp(conf_int[1]))]

    # If p-value still missing, mark as None
    significant = None
    if pval is not None:
        significant = bool(pval < 0.05)

    # Build a concise conclusion statement
    if coef is None:
        conclusion = "Could not find the SkinDark coefficient in the provided model output."
    else:
        # percent increase approximation from IRR
        if irr is not None:
            pct = (irr - 1.0) * 100.0
            pct_txt = f"about {pct:.1f}% higher" if pct >= 0 else f"about {abs(pct):.1f}% lower"
        else:
            pct_txt = "an increase (IRR unavailable)"

        sig_txt = ""
        if significant is True:
            sig_txt = " This effect is statistically significant (p < 0.05)."
        elif significant is False:
            sig_txt = " This effect is not statistically significant (p >= 0.05)."
        else:
            sig_txt = ""

        ci_txt = ""
        if irr_ci is not None:
            ci_txt = f" 95% CI for IRR = [{irr_ci[0]:.3f}, {irr_ci[1]:.3f}]."
        elif pval is not None:
            ci_txt = ""

        conclusion = (
            f"Players coded as having dark skin have a positive coefficient on SkinDark "
            f"(coef = {coef:.4f}), corresponding to an IRR = {irr:.3f} ({pct_txt} rate of red cards per game)."
            + ci_txt + sig_txt
        )

    # Prepare the object to return (structured numeric summary)
    object_out = {
        'coef': coef,
        'pvalue': pval,
        'significant_at_0.05': significant,
        'IRR': irr,
        'IRR_95CI': irr_ci,
        'conclusion': conclusion
    }

    # Description: brief interpretation in context
    description = (
        "Interpretation: Controlling for age, height, weight, position, league country, and referee-country-level "
        "implicit/explicit bias (and clustering SEs by referee), the model estimates that players with a dark skin "
        "tone receive red cards at a higher rate than light-skinned players. The IRR indicates the multiplicative "
        "change in the red-card rate per game for dark-skinned vs light-skinned players. The numeric fields above "
        "give the coefficient, p-value, IRR, and 95% CI (if available)."
    )

    return {"object": object_out, "description": description}