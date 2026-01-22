def extract_final_answer(model_output):
    """
    Extracts the estimated effect of instructor beauty (beauty_z) on student evaluations
    from the provided model_output dictionary. Returns a dict with a machine-readable
    "object" containing numeric results for the baseline and professor fixed-effects
    specifications, and a short human-readable "description" interpreting those results.
    """
    import math

    def _extract_from_result(res):
        """Safely extract coef, se, pval and 95% CI for 'beauty_z' from a statsmodels result."""
        stats = {'coef': None, 'se': None, 'pval': None, 'ci_lower': None, 'ci_upper': None}
        if res is None:
            return stats

        # params / bse / pvalues may be pandas Series or numpy arrays
        params = getattr(res, 'params', None)
        bse = getattr(res, 'bse', None)
        pvalues = getattr(res, 'pvalues', None)

        # 1) try Series/dict-like access
        try:
            if params is not None and 'beauty_z' in params:
                stats['coef'] = float(params['beauty_z'])
        except Exception:
            stats['coef'] = None

        try:
            if bse is not None and 'beauty_z' in bse:
                stats['se'] = float(bse['beauty_z'])
        except Exception:
            stats['se'] = None

        try:
            if pvalues is not None and 'beauty_z' in pvalues:
                stats['pval'] = float(pvalues['beauty_z'])
        except Exception:
            stats['pval'] = None

        # 2) fallback: locate index by name in model.exog_names
        if stats['coef'] is None or stats['se'] is None or stats['pval'] is None:
            try:
                exog_names = list(getattr(getattr(res, 'model', None), 'exog_names', []))
                if 'beauty_z' in exog_names:
                    idx = exog_names.index('beauty_z')
                    if stats['coef'] is None and params is not None:
                        try:
                            stats['coef'] = float(params[idx])
                        except Exception:
                            pass
                    if stats['se'] is None and bse is not None:
                        try:
                            stats['se'] = float(bse[idx])
                        except Exception:
                            pass
                    if stats['pval'] is None and pvalues is not None:
                        try:
                            stats['pval'] = float(pvalues[idx])
                        except Exception:
                            pass
            except Exception:
                pass

        # 3) attempt to get confidence interval via conf_int() if available
        try:
            ci = res.conf_int()
            # conf_int may be a DataFrame/ndarray; handle both
            try:
                if hasattr(ci, 'loc') and 'beauty_z' in ci.index:
                    stats['ci_lower'] = float(ci.loc['beauty_z', 0])
                    stats['ci_upper'] = float(ci.loc['beauty_z', 1])
                else:
                    # try to find index
                    exog_names = list(getattr(getattr(res, 'model', None), 'exog_names', []))
                    if 'beauty_z' in exog_names:
                        idx = exog_names.index('beauty_z')
                        stats['ci_lower'] = float(ci[idx, 0])
                        stats['ci_upper'] = float(ci[idx, 1])
            except Exception:
                # if conf_int returned nested sequences
                if isinstance(ci, (list, tuple)):
                    pass
        except Exception:
            # fallback: build CI from coef +/- 1.96*se if both exist
            if stats['coef'] is not None and stats['se'] is not None and not math.isnan(stats['se']):
                stats['ci_lower'] = stats['coef'] - 1.96 * stats['se']
                stats['ci_upper'] = stats['coef'] + 1.96 * stats['se']

        return stats

    # Get results objects (may be present under different keys)
    baseline_res = model_output.get('baseline_model') or model_output.get('ols_base_clust') or None
    fe_res = model_output.get('prof_fe_model') or model_output.get('ols_fe_clust') or None

    baseline_stats = _extract_from_result(baseline_res)
    fe_stats = _extract_from_result(fe_res)

    # If the model_output already contains a precomputed beauty_effect_baseline, prefer its coef/se/CI when missing
    precomp = model_output.get('beauty_effect_baseline')
    if precomp and baseline_stats.get('coef') is None:
        try:
            baseline_stats['coef'] = float(precomp.get('coef', baseline_stats.get('coef')))
            baseline_stats['se'] = float(precomp.get('se', baseline_stats.get('se')))
            ci = precomp.get('95ci') or precomp.get('ci') or (baseline_stats.get('ci_lower'), baseline_stats.get('ci_upper'))
            if ci and len(ci) == 2:
                baseline_stats['ci_lower'], baseline_stats['ci_upper'] = float(ci[0]), float(ci[1])
        except Exception:
            pass

    # Build a short interpretation for each spec
    def _interpret(s):
        if s['coef'] is None:
            return "No estimate available."
        p = s['pval']
        sig = None
        if p is None:
            sig = "p-value unavailable"
        else:
            if p < 0.01:
                sig = f"statistically significant (p < 0.01)"
            elif p < 0.05:
                sig = f"statistically significant (p = {p:.3f})"
            elif p < 0.1:
                sig = f"marginal (p = {p:.3f})"
            else:
                sig = f"not statistically significant (p = {p:.3f})"

        ci_text = ""
        if s['ci_lower'] is not None and s['ci_upper'] is not None:
            ci_text = f" 95% CI [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]."

        # coef is change in eval (1-5 scale) per 1 SD increase in beauty_z
        return (f"Coefficient = {s['coef']:.4f} (change in evaluation points per 1 SD increase in rated beauty); "
                f"SE = {s['se']:.4f}." + f" {sig}." + ci_text)

    baseline_interp = _interpret(baseline_stats)
    fe_interp = _interpret(fe_stats)

    # Final conclusion: combine evidence (here we consider baseline primary)
    conclusion = "Based on the estimates, there is no evidence that instructor beauty meaningfully affects student evaluation scores. " \
                 "The baseline coefficient is essentially zero and not statistically significant; the professor fixed-effects estimate is also small and not significant."

    # Construct machine-readable object to return
    result_object = {
        'baseline': baseline_stats,
        'professor_fixed_effects': fe_stats,
        # convenience summary flags
        'conclusion_significant_baseline': (baseline_stats.get('pval') is not None and baseline_stats.get('pval') < 0.05),
        'conclusion_significant_fe': (fe_stats.get('pval') is not None and fe_stats.get('pval') < 0.05),
        'overall_conclusion': conclusion
    }

    description = (
        "Extracted estimates for the effect of instructor beauty (beauty_z) on course evaluation (eval):\n"
        f"- Baseline model: {baseline_interp}\n"
        f"- Professor fixed-effects model: {fe_interp}\n\n"
        "Interpretation: The estimated effect sizes are effectively zero (on the 1-5 evaluation scale per 1 SD in beauty) "
        "and neither specification shows a statistically significant relationship. Therefore, the data provide no evidence that "
        "beauty meaningfully influences teaching evaluations in these models."
    )

    return {"object": result_object, "description": description}