def extract_final_answer(model_output):
    """
    Extracts effect estimates for 'SkinDark' from the supplied model_output dict.
    Returns a dict with:
      - "object": dict containing coefficients, cluster-robust SEs, IRR/OR, 95% CIs, p-values, and sample size
      - "description": short interpretation answering the task question
    """
    import numpy as np
    from math import exp, sqrt
    try:
        from scipy import stats as _stats
        norm_sf = _stats.norm.sf
    except Exception:
        # fallback to approximate p using math.erfc if scipy not available
        import math as _math
        def norm_sf(z):
            return 0.5 * _math.erfc(z / _math.sqrt(2))

    out = {}
    try:
        # Negative binomial (rate) model
        nb_res = model_output.get('nb_model_clustered', None)
        nb_cov = model_output.get('nb_cluster_cov', None)
        nb_info = None
        if nb_res is not None and nb_cov is not None:
            params = nb_res.params  # pandas Series
            if 'SkinDark' in params.index:
                idx = params.index.get_loc('SkinDark')
                coef = float(params['SkinDark'])
                se = float(np.sqrt(np.diag(nb_cov))[idx])
                z = coef / se if se != 0 else np.nan
                p = 2.0 * float(norm_sf(abs(z)))
                irr = float(np.exp(coef))
                ci_lower = float(np.exp(coef - 1.96 * se))
                ci_upper = float(np.exp(coef + 1.96 * se))
                nb_info = {
                    'model': 'negative_binomial_rate',
                    'coef_log': coef,
                    'se_cluster': se,
                    'z': z,
                    'p_value': p,
                    'IRR': irr,
                    'IRR_95CI': (ci_lower, ci_upper)
                }
            else:
                nb_info = {'error': "'SkinDark' not in nb model params"}
        else:
            nb_info = {'error': 'nb model or nb covariance missing'}

        # Logistic (AnyRed) robustness model
        bin_res = model_output.get('binomial_model_clustered', None)
        bin_cov = model_output.get('bin_cluster_cov', None)
        bin_info = None
        if bin_res is not None and bin_cov is not None:
            params_b = bin_res.params
            if 'SkinDark' in params_b.index:
                idx_b = params_b.index.get_loc('SkinDark')
                coef_b = float(params_b['SkinDark'])
                se_b = float(np.sqrt(np.diag(bin_cov))[idx_b])
                z_b = coef_b / se_b if se_b != 0 else np.nan
                p_b = 2.0 * float(norm_sf(abs(z_b)))
                orr = float(np.exp(coef_b))
                ci_lower_b = float(np.exp(coef_b - 1.96 * se_b))
                ci_upper_b = float(np.exp(coef_b + 1.96 * se_b))
                bin_info = {
                    'model': 'logistic_anyred',
                    'coef_logodds': coef_b,
                    'se_cluster': se_b,
                    'z': z_b,
                    'p_value': p_b,
                    'OR': orr,
                    'OR_95CI': (ci_lower_b, ci_upper_b)
                }
            else:
                bin_info = {'error': "'SkinDark' not in binomial model params"}
        else:
            bin_info = {'error': 'binomial model or covariance missing'}

        # sample size
        n_rows = model_output.get('model_df_rows', None)

        # Build final object with both results
        result_obj = {
            'negative_binomial': nb_info,
            'binomial_robustness': bin_info,
            'n_rows_used': n_rows
        }

        # Short description / interpretation
        # Determine whether NB effect is statistically significant at alpha=0.05 and direction
        conclusion = "inconclusive"
        if isinstance(nb_info, dict) and 'p_value' in nb_info:
            if nb_info['p_value'] < 0.05:
                if nb_info['IRR'] > 1.0:
                    conclusion = "yes (NB rate model): dark-skinned players receive more red cards per game"
                elif nb_info['IRR'] < 1.0:
                    conclusion = "yes (NB rate model): dark-skinned players receive fewer red cards per game"
            else:
                conclusion = "no strong evidence (NB rate model): difference not statistically significant"

        description = (
            f"Negative binomial (rate per game) result: IRR = {nb_info.get('IRR') if isinstance(nb_info, dict) else 'NA'} "
            f"95% CI = {nb_info.get('IRR_95CI') if isinstance(nb_info, dict) else 'NA'} "
            f"p = {nb_info.get('p_value') if isinstance(nb_info, dict) else 'NA'}. "
            f"Logistic robustness (ever received a red card) result: OR = {bin_info.get('OR') if isinstance(bin_info, dict) else 'NA'} "
            f"95% CI = {bin_info.get('OR_95CI') if isinstance(bin_info, dict) else 'NA'} "
            f"p = {bin_info.get('p_value') if isinstance(bin_info, dict) else 'NA'}. "
            f"Based on the NB model (cluster-robust SEs), the conclusion is: {conclusion}. "
            f"n (dyads) = {n_rows}."
        )

        return {'object': result_obj, 'description': description}

    except Exception as e:
        return {
            'object': None,
            'description': f'Failed to extract results due to error: {e}'
        }