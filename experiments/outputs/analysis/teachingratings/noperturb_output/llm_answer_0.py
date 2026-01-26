def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, and 95% CI for 'beauty_z'
    from the provided model_output dict. Picks a primary model for interpretation
    (prefers clustered results) and also returns statistics for all available models.

    Returns:
      {
        "object": {
           "primary_model": <model_name>,
           "primary_stats": {coef, std_err, p_value, ci_lower, ci_upper, nobs, significant},
           "all_models": { model_name: {coef, std_err, p_value, ci_lower, ci_upper, nobs}, ... }
        },
        "description": "<text interpretation>"
      }
    """
    import numpy as np

    preferred_order = ['ols_cluster', 'fe_cluster', 'wls_cluster', 'ols', 'fe', 'wls']

    def _get_stats(res, param='beauty_z'):
        # Safely extract numeric stats for a single parameter from a statsmodels result object
        try:
            coef = float(res.params[param])
        except Exception:
            return None
        try:
            se = float(res.bse[param])
        except Exception:
            # fallback: compute se from tvalue if available and coef != 0 (not ideal)
            try:
                tval = float(res.tvalues[param])
                se = abs(coef / tval) if tval != 0 else np.nan
            except Exception:
                se = np.nan
        try:
            pval = float(res.pvalues[param])
        except Exception:
            pval = np.nan
        # confidence interval extraction robust to different return types
        try:
            ci_table = res.conf_int()
            try:
                # try pandas-like access
                ci_lower, ci_upper = list(ci_table.loc[param])
            except Exception:
                # assume numpy array with same order as params
                params_index = list(res.params.index)
                idx = params_index.index(param)
                ci_lower, ci_upper = float(ci_table[idx, 0]), float(ci_table[idx, 1])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        # attempt to get nobs
        nobs = None
        for attr in ('nobs', 'model', 'df_resid'):
            try:
                if attr == 'nobs' and hasattr(res, 'nobs'):
                    nobs = int(res.nobs)
                    break
                if attr == 'df_resid' and hasattr(res, 'df_resid') and hasattr(res, 'df_model'):
                    # approximate nobs = df_model + df_resid + 1 (depends on model)
                    try:
                        nobs = int(res.df_model + res.df_resid + 1)
                        break
                    except Exception:
                        pass
            except Exception:
                pass

        return {
            'coef': coef,
            'std_err': se,
            'p_value': pval,
            'ci_lower': float(ci_lower) if ci_lower is not None else np.nan,
            'ci_upper': float(ci_upper) if ci_upper is not None else np.nan,
            'nobs': nobs
        }

    all_stats = {}
    for name, res in (model_output or {}).items():
        try:
            stats = _get_stats(res, 'beauty_z')
            if stats is not None:
                all_stats[name] = stats
        except Exception:
            # skip models we can't parse
            continue

    # choose primary model according to preferred order
    primary_model = None
    for name in preferred_order:
        if name in all_stats:
            primary_model = name
            break
    if primary_model is None and len(all_stats) > 0:
        # fallback to first available
        primary_model = next(iter(all_stats.keys()))

    primary_stats = None
    description = "No model statistics for 'beauty_z' could be extracted."
    if primary_model is not None:
        primary_stats = all_stats[primary_model]
        signif = None
        try:
            signif = bool(primary_stats['p_value'] < 0.05)
        except Exception:
            signif = None

        coef = primary_stats['coef']
        pval = primary_stats['p_value']
        ci_l = primary_stats['ci_lower']
        ci_u = primary_stats['ci_upper']

        # Interpretation: coefficient units are evaluation points (1-5) per 1 SD in beauty
        direction = 'positive' if coef > 0 else ('negative' if coef < 0 else 'zero')
        significance_text = ("statistically significant at the 5% level" if signif is True
                             else "not statistically significant at the 5% level" if signif is False
                             else "significance could not be determined")

        description = (
            f"Primary model used for interpretation: '{primary_model}'.\n"
            f"Estimated effect of one standard deviation increase in instructor beauty (beauty_z) on student evaluation (eval):\n"
            f"  Coefficient = {coef:.4f} (this is on the 1-5 evaluation scale),\n"
            f"  SE = {primary_stats['std_err']:.4f}, p-value = {pval:.4g}, 95% CI = [{ci_l:.4f}, {ci_u:.4f}].\n"
            f"Interpretation: A one-SD higher beauty score is associated with a {coef:.4f}-point change in the evaluation score ({direction}).\n"
            f"This effect is {significance_text}.\n"
            "Note: Results for other available models are provided in 'object' -> 'all_models'."
        )

    return {
        "object": {
            "primary_model": primary_model,
            "primary_stats": primary_stats,
            "all_models": all_stats
        },
        "description": description
    }