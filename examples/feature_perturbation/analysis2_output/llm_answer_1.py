def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of the standardized name femininity index (MasFem_z)
    from the modeling output produced by the modeling function.

    Returns a dictionary with keys:
      - "object": a JSON-serializable dict with extracted numeric results or a status dict
      - "description": a human-readable explanation of what was extracted and its meaning

    Behavior:
      - If a Negative Binomial (nb_results) is available, this is treated as the primary result.
      - If nb_results is missing but an OLS (ols_results) is available, OLS is used as a sensitivity result.
      - If neither model is present, returns a status indicating models could not be fitted and the
        number of observations available (if any).
    """
    import numpy as np

    nb = model_output.get('nb_results')
    ols = model_output.get('ols_results')
    df = model_output.get('model_dataframe')

    # Helper to safely extract stats from a statsmodels result-like object
    def _extract_from_result(res, param_name):
        """Return dict with coef, pvalue, conf_lower, conf_upper if available, else None."""
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            conf_int = None
            try:
                conf_int = res.conf_int()
            except Exception:
                conf_int = None

            if params is None or param_name not in params.index:
                return None

            coef = float(params[param_name])
            pval = float(pvalues[param_name]) if (pvalues is not None and param_name in pvalues.index) else None

            if conf_int is not None:
                # conf_int may be a DataFrame (with loc) or numpy array
                try:
                    # DataFrame-like
                    lower, upper = conf_int.loc[param_name].tolist()
                except Exception:
                    # array-like; find index
                    try:
                        idx = list(params.index).index(param_name)
                        lower, upper = float(conf_int[idx, 0]), float(conf_int[idx, 1])
                    except Exception:
                        lower, upper = None, None
            else:
                lower, upper = None, None

            return {
                'coef': coef,
                'pvalue': pval,
                'conf_lower': float(lower) if lower is not None else None,
                'conf_upper': float(upper) if upper is not None else None
            }
        except Exception:
            return None

    # Determine number of observations if possible
    n_obs = None
    try:
        if df is not None:
            n_obs = int(df.shape[0])
    except Exception:
        n_obs = None

    # Prefer NB results if available
    if nb is not None:
        stats = _extract_from_result(nb, 'MasFem_z')
        if stats is None:
            return {
                'object': {'status': 'no_MasFem_z_in_nb', 'n_obs': n_obs},
                'description': "Negative Binomial model was fitted but does not contain a parameter named 'MasFem_z'."
            }
        # Interpret for count model: exp(coef) is multiplicative change in expected counts per 1 SD increase
        try:
            mult = float(np.exp(stats['coef']))
        except Exception:
            mult = None

        signif = None
        try:
            if stats['pvalue'] is None:
                signif = 'p-value unavailable'
            else:
                signif = 'statistically significant (p < 0.05)' if stats['pvalue'] < 0.05 else 'not statistically significant (p >= 0.05)'
        except Exception:
            signif = 'significance unknown'

        obj = {
            'model_type': 'NegativeBinomial (GLM)',
            'n_obs': n_obs,
            'parameter': 'MasFem_z',
            'coef': stats['coef'],
            'pvalue': stats['pvalue'],
            'conf_lower': stats['conf_lower'],
            'conf_upper': stats['conf_upper'],
            'multiplicative_effect_on_counts': mult,
            'interpretation': (
                "Multiplicative effect on expected deaths per 1 SD increase in name femininity. "
                "E.g., value 0.90 means expected deaths are 0.90x (10% lower)."
            )
        }
        description = (
            "Primary model (Negative Binomial GLM) results for MasFem_z. "
            f"Estimated coefficient = {obj['coef']:.4f}, multiplicative effect = "
            f"{obj['multiplicative_effect_on_counts']:.4f} (if available). "
            f"p-value = {obj['pvalue']}. This effect is {signif}. "
            "If multiplicative_effect_on_counts < 1 the result indicates fewer deaths (fewer fatalities) "
            "associated with more feminine names; >1 indicates more deaths associated with more feminine names."
        )
        return {'object': obj, 'description': description}

    # If NB not present but OLS present, use OLS sensitivity
    if ols is not None:
        stats = _extract_from_result(ols, 'MasFem_z')
        if stats is None:
            return {
                'object': {'status': 'no_MasFem_z_in_ols', 'n_obs': n_obs},
                'description': "OLS model was fitted but does not contain a parameter named 'MasFem_z'."
            }
        # Interpret for OLS on log(deaths + 1): approximate percent change = 100*(exp(coef)-1)
        try:
            pct_change = float((np.exp(stats['coef']) - 1.0) * 100.0)
        except Exception:
            pct_change = None

        signif = None
        try:
            if stats['pvalue'] is None:
                signif = 'p-value unavailable'
            else:
                signif = 'statistically significant (p < 0.05)' if stats['pvalue'] < 0.05 else 'not statistically significant (p >= 0.05)'
        except Exception:
            signif = 'significance unknown'

        obj = {
            'model_type': 'OLS on LogDeaths (sensitivity)',
            'n_obs': n_obs,
            'parameter': 'MasFem_z',
            'coef': stats['coef'],
            'pvalue': stats['pvalue'],
            'conf_lower': stats['conf_lower'],
            'conf_upper': stats['conf_upper'],
            'approx_percent_change_in_(deaths+1)': pct_change,
            'interpretation': (
                "Approximate percent change in (deaths + 1) per 1 SD increase in name femininity. "
                "E.g., -10% means (deaths+1) is ~10% lower."
            )
        }
        description = (
            "Sensitivity model (OLS on log(deaths+1)) results for MasFem_z. "
            f"Estimated coefficient = {obj['coef']:.4f}. Approximate percent change = "
            f"{obj['approx_percent_change_in_(deaths+1)']:.2f}% (if available). "
            f"p-value = {obj['pvalue']}. This effect is {signif}. "
            "Negative percent change indicates fewer fatalities associated with more feminine names."
        )
        return {'object': obj, 'description': description}

    # Neither model is present
    return {
        'object': {'status': 'no_model_fitted', 'n_obs': n_obs},
        'description': (
            "No fitted models were returned (both 'nb_results' and 'ols_results' are None). "
            f"Number of observations in the modeling dataframe: {n_obs}. Cannot estimate or test the effect of MasFem_z."
        )
    }