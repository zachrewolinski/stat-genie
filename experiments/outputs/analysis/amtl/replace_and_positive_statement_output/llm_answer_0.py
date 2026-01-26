def extract_final_answer(model_output):
    """
    Extracts statistics about the IsHuman effect from the model_output dict
    and returns a structured conclusion and explanatory text.

    Returns:
      {
        "object": { ... extracted numeric values and a flagged conclusion ... },
        "description": "Plain-language interpretation and cautions"
      }
    """
    import math
    import numpy as np

    out = {
        "object": None,
        "description": None
    }

    # Basic extraction with fallbacks
    coef = model_output.get('IsHuman_coef', None)
    se = model_output.get('IsHuman_se', None)
    pval = model_output.get('IsHuman_pvalue', None)
    or_est = model_output.get('IsHuman_OR', None)
    or_ci = model_output.get('IsHuman_OR_CI', (None, None))
    dispersion = model_output.get('dispersion', None)
    pearson_chi2 = model_output.get('pearson_chi2', None)
    df_resid = model_output.get('df_resid', None)
    summary_text = model_output.get('summary_text', None)

    # If keys missing, try to pull from model_result (statsmodels result wrapper)
    res = model_output.get('model_result', None)
    if res is not None:
        try:
            if coef is None and 'IsHuman' in res.params.index:
                coef = float(res.params['IsHuman'])
            if se is None and 'IsHuman' in res.bse.index:
                se = float(res.bse['IsHuman'])
            if pval is None and 'IsHuman' in res.pvalues.index:
                pval = float(res.pvalues['IsHuman'])
            if (or_est is None or (isinstance(or_est, float) and (math.isnan(or_est) or math.isinf(or_est)))) and coef is not None:
                # compute OR if coef is numeric and reasonable
                try:
                    or_est = float(np.exp(coef))
                except Exception:
                    or_est = or_est
            if (or_ci is None or or_ci == (None, None)) and res is not None:
                try:
                    ci = res.conf_int().loc['IsHuman'].values
                    or_ci = (float(np.exp(ci[0])), float(np.exp(ci[1])))
                except Exception:
                    pass
        except Exception:
            pass

    # Build verdict flags
    evidence_flag = None
    notes = []

    # Check for extreme/invalid estimates often indicating separation or convergence failure
    extreme_coef = False
    if coef is None:
        notes.append("IsHuman coefficient not available.")
    else:
        if not np.isfinite(coef):
            extreme_coef = True
            notes.append("IsHuman coefficient is not finite (NaN/Inf).")
        elif abs(coef) > 1e6:
            extreme_coef = True
            notes.append("IsHuman coefficient is extremely large (suggests separation or numerical failure).")

    if or_est is None:
        notes.append("Odds ratio for IsHuman not available.")
    else:
        if (isinstance(or_est, float) and (math.isinf(or_est) or math.isnan(or_est))):
            notes.append("Odds ratio is infinite or NaN (suggests perfect or quasi-complete separation).")

    if summary_text:
        # look for signs of non-convergence / iteration cap / NaN log-likelihood
        if "No. Iterations:" in summary_text and "100" in summary_text:
            notes.append("Model reached iteration limit (possible non-convergence).")
        if "Log-Likelihood:                    nan" in summary_text or "nan" in summary_text.splitlines()[1]:
            notes.append("Log-likelihood is NaN in model summary (diagnostic of fitting problems).")

    # Overdispersion check
    if dispersion is not None:
        try:
            if np.isfinite(dispersion) and dispersion > 2:
                notes.append(f"Dispersion statistic is large ({dispersion:.2g}), indicating overdispersion relative to binomial.")
        except Exception:
            pass

    # Make a conclusion about direction and statistical significance,
    # but downgrade to "inconclusive" if diagnostics indicate separation/convergence problems.
    direction = None
    significant = None
    if coef is not None and np.isfinite(coef):
        direction = "higher" if coef > 0 else "lower"
    if pval is not None and np.isfinite(pval):
        significant = (pval < 0.05)

    if extreme_coef or ('Odds ratio is infinite' in " ".join(notes) or any("separation" in n.lower() for n in notes)) :
        # Strong association signal but unreliable numeric estimate
        evidence_flag = "suggestive_but_unreliable"
        conclusion_text = (
            "The model indicates a very large, positive IsHuman effect (modern humans associated with higher AMTL), "
            "and the reported p-value is extremely small. HOWEVER the coefficient/OR estimates are numerically "
            "unstable or infinite and model diagnostics (large dispersion, NaN log-likelihood, or iteration limit) "
            "suggest complete or quasi-complete separation or other convergence problems. Therefore the direction "
            "of higher AMTL in modern humans is supported by the fitted model, but the numeric effect size is not "
            "trustworthy from this fit."
        )
    else:
        # If estimates appear reasonable, draw a standard conclusion
        if direction is not None and significant is not None:
            if significant:
                evidence_flag = "supported"
                conclusion_text = (
                    f"After adjustment for covariates, IsHuman has a {direction} AMTL (coef = {coef:.4g}, "
                    f"OR = {or_est if (or_est is None or not np.isfinite(or_est)) else round(or_est,3)}, p = {pval:.3g}). "
                    "This provides statistical evidence that modern humans have higher AMTL than the non-human primates in this dataset."
                )
            else:
                evidence_flag = "not_supported"
                conclusion_text = (
                    f"After adjustment, IsHuman coefficient has direction {direction} but is not statistically significant "
                    f"(coef = {coef:.4g}, p = {pval:.3g}). The model does not provide evidence that modern humans have different AMTL."
                )
        else:
            evidence_flag = "inconclusive"
            conclusion_text = "Insufficient numerical information to draw a clear conclusion."

    # Assemble object to return
    result_obj = {
        "conclusion_flag": evidence_flag,      # 'supported', 'not_supported', 'suggestive_but_unreliable', or 'inconclusive'
        "conclusion_direction": direction,     # 'higher' / 'lower' / None
        "IsHuman_coef": coef,
        "IsHuman_se": se,
        "IsHuman_pvalue": pval,
        "IsHuman_OR": or_est,
        "IsHuman_OR_CI": or_ci,
        "dispersion": dispersion,
        "pearson_chi2": pearson_chi2,
        "df_resid": df_resid,
        "notes": notes
    }

    out['object'] = result_obj
    # Compose description: short, plain-language with caution
    out['description'] = conclusion_text

    return out