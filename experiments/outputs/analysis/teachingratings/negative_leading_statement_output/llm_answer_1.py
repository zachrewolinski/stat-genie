def extract_final_answer(model_output):
    """
    Extracts the effect of instructor beauty (beauty_z) on eval from the provided
    modeling output dictionary.

    Returns a dict with:
      - "object": a dict containing numeric estimates (coef, se, p, 95% CI, n_obs)
                  for both the OLS (cluster-robust) and the MixedLM (if available).
      - "description": a short, plain-language interpretation of the results.

    The function is defensive: it handles missing keys and cases where the MixedLM
    failed (in which case mixedlm_result may be an Exception).
    """
    import numpy as np

    def _safe_get_conf_int(res, name):
        """Return (lower, upper) 95% CI for parameter name from a statsmodels result."""
        try:
            ci = res.conf_int()
            # conf_int may be a DataFrame or ndarray
            if hasattr(ci, "loc"):
                # DataFrame-like
                row = ci.loc[name]
                return float(row[0]), float(row[1])
            else:
                # ndarray-like; need to find index of name
                try:
                    idx = list(res.params.index).index(name)
                except Exception:
                    # fallback: if params isn't indexable, return NaNs
                    return (np.nan, np.nan)
                return float(ci[idx, 0]), float(ci[idx, 1])
        except Exception:
            return (np.nan, np.nan)

    def _extract_from_result(res, name='beauty_z'):
        """
        Extract params, se, pvalue, conf_int, nobs from a statsmodels result-like object.
        Returns dict or None if extraction fails.
        """
        if res is None:
            return None
        try:
            # params: try typical attributes
            if hasattr(res, 'params') and name in getattr(res, 'params').index:
                coef = float(res.params[name])
            elif hasattr(res, 'fe_params') and name in getattr(res, 'fe_params').index:
                coef = float(res.fe_params[name])
            else:
                return None

            # standard error
            if hasattr(res, 'bse') and name in getattr(res, 'bse').index:
                se = float(res.bse[name])
            elif hasattr(res, 'bse_fe') and name in getattr(res, 'bse_fe').index:
                se = float(res.bse_fe[name])
            else:
                # Some wrappers store bse as attribute with same index as params
                try:
                    se = float(res.bse.loc[name])
                except Exception:
                    se = float(np.nan)

            # p-value
            if hasattr(res, 'pvalues') and name in getattr(res, 'pvalues').index:
                p = float(res.pvalues[name])
            else:
                # For MixedLMResults, pvalues may be accessible via pvalues or computed
                try:
                    p = float(res.pvalues[name])
                except Exception:
                    p = float(np.nan)

            # conf int
            ci_lower, ci_upper = _safe_get_conf_int(res, name)

            # nobs
            try:
                nobs = int(res.nobs)
            except Exception:
                # fallback: try to find model_df in supplied model_output
                nobs = None

            return {
                'coef': coef,
                'se': se,
                'pvalue': p,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'nobs': nobs
            }
        except Exception:
            return None

    # Start building output
    out_obj = {}
    descriptions = []

    # OLS clustered
    ols_res = model_output.get('ols_clustered')
    ols_stats = None
    try:
        ols_stats = _extract_from_result(ols_res, 'beauty_z')
    except Exception:
        ols_stats = None

    if ols_stats is None:
        out_obj['ols_clustered'] = None
        descriptions.append("Could not extract OLS (cluster-robust) estimates for 'beauty_z'.")
    else:
        # Interpret effect: beauty_z is standardized so coef = change in eval per 1 SD beauty
        signif = (not np.isnan(ols_stats['pvalue'])) and (ols_stats['pvalue'] < 0.05)
        direction = 'positive' if ols_stats['coef'] > 0 else ('negative' if ols_stats['coef'] < 0 else 'no')
        out_obj['ols_clustered'] = ols_stats
        descriptions.append(
            "OLS (cluster-robust SE): a one-standard-deviation increase in instructor "
            f"attractiveness is associated with a {ols_stats['coef']:.3f} point change in "
            f"course evaluation (SE={ols_stats['se']:.3f}, 95% CI [{ols_stats['ci_95_lower']:.3f}, "
            f"{ols_stats['ci_95_upper']:.3f}], p={ols_stats['pvalue']:.3g}). This effect is "
            f"{'statistically significant' if signif else 'not statistically significant'} and "
            f"{direction}."
        )

    # MixedLM
    mixed_res = model_output.get('mixedlm_result')
    mixed_stats = None
    if isinstance(mixed_res, Exception):
        out_obj['mixedlm'] = None
        descriptions.append(f"MixedLM failed with exception: {repr(mixed_res)}")
    else:
        try:
            mixed_stats = _extract_from_result(mixed_res, 'beauty_z')
        except Exception:
            mixed_stats = None

        if mixed_stats is None:
            out_obj['mixedlm'] = None
            descriptions.append("Could not extract MixedLM estimates for 'beauty_z'.")
        else:
            signif_m = (not np.isnan(mixed_stats['pvalue'])) and (mixed_stats['pvalue'] < 0.05)
            direction_m = 'positive' if mixed_stats['coef'] > 0 else ('negative' if mixed_stats['coef'] < 0 else 'no')
            out_obj['mixedlm'] = mixed_stats
            descriptions.append(
                "Mixed-effects model (random intercept for professor): a one-standard-deviation increase in "
                f"instructor attractiveness is associated with a {mixed_stats['coef']:.3f} point change in "
                f"course evaluation (SE={mixed_stats['se']:.3f}, 95% CI [{mixed_stats['ci_95_lower']:.3f}, "
                f"{mixed_stats['ci_95_upper']:.3f}], p={mixed_stats['pvalue']:.3g}). This effect is "
                f"{'statistically significant' if signif_m else 'not statistically significant'} and "
                f"{direction_m}."
            )

    # Combine descriptions into a short summary
    final_description = " ".join(descriptions)

    return {
        "object": out_obj,
        "description": final_description
    }