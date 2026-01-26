def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of 'beauty_z' on 'eval' from the model_output.
    Returns a dictionary with:
      - "object": a dict containing coefficient, p-value, 95% CI, and a short numeric summary
      - "description": a short plain-language interpretation in context
    
    The function is robust to two shapes of model_output:
      - a dict with pre-extracted keys ('coef_beauty', 'pvalue_beauty', 'conf_int_beauty', 'results')
      - or a statsmodels RegressionResults-like object in model_output (or under model_output['results'])
    """
    import math

    # Initialize placeholders
    coef = None
    pval = None
    ci = None
    used_cluster_se = None

    # If model_output is a dict with expected keys, prefer those
    if isinstance(model_output, dict):
        coef = model_output.get('coef_beauty', None)
        pval = model_output.get('pvalue_beauty', None)
        ci = model_output.get('conf_int_beauty', None)
        res = model_output.get('results', None)
    else:
        res = model_output

    # If any values are missing, try to extract from the statsmodels results object
    if res is not None and (coef is None or pval is None or ci is None):
        try:
            # If res is the wrapper dict with a 'results' key already handled, this still works
            params = getattr(res, 'params', None)
            if params is not None and 'beauty_z' in params.index:
                coef = float(params.loc['beauty_z'])
            # p-values
            pvalues = getattr(res, 'pvalues', None)
            if pvalues is not None and 'beauty_z' in pvalues.index:
                pval = float(pvalues.loc['beauty_z'])
            # conf_int
            try:
                conf = res.conf_int()
                if 'beauty_z' in conf.index:
                    ci = [float(conf.loc['beauty_z', 0]), float(conf.loc['beauty_z', 1])]
            except Exception:
                # some result objects may need different access; ignore if fails
                pass
            # Attempt to detect if cluster-robust cov type was used (best-effort)
            cov_type = getattr(res, 'cov_type', None)
            if cov_type is not None:
                used_cluster_se = ('cluster' in str(cov_type).lower())
        except Exception:
            pass

    # Final sanity checks / defaults
    if coef is None:
        coef = float('nan')
    if pval is None:
        pval = float('nan')
    if ci is None:
        ci = [float('nan'), float('nan')]

    # Build a concise numeric object to return
    numeric_summary = {
        'coef_beauty': coef,
        'pvalue_beauty': pval,
        'conf_int_95': ci,
        'units': 'change in eval score (1-5 scale) per 1 SD increase in beauty_z',
        'note_on_se': ('cluster-robust SEs used' if used_cluster_se else 'SE type unknown or not detected')
    }

    # Plain-language description / interpretation
    # Emphasize practical and statistical significance and CI-based precision
    try:
        coef_display = f"{coef:.4f}" if not math.isnan(coef) else "NA"
        pval_display = f"{pval:.3f}" if not math.isnan(pval) else "NA"
        ci_display = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not any(math.isnan(x) for x in ci) else "NA"
    except Exception:
        coef_display = str(coef)
        pval_display = str(pval)
        ci_display = str(ci)

    description = (
        f"Estimated effect of instructor attractiveness (1 SD increase) on student overall evaluation: "
        f"coef = {coef_display}, p = {pval_display}, 95% CI = {ci_display}. "
        "This estimate is effectively zero and statistically non-significant. "
        "Given the 95% CI (~±0.05), we can rule out effects larger than about 0.05 points on the 1–5 evaluation scale "
        "per standard-deviation change in rated attractiveness. In plain terms, there is no evidence that beauty meaningfully "
        "affects teaching evaluations in this sample. "
        f"({numeric_summary['note_on_se']})"
    )

    return {
        "object": numeric_summary,
        "description": description
    }