def extract_final_answer(model_output):
    """
    Extract key statistics for the predictors of interest from a fitted statsmodels binary
    logistic regression result (BinaryResultsWrapper or similar).

    Returns a dictionary with:
      - "object": dict with entries for 'RelSize_z', 'DistAdv_z', 'RelSize_z:DistAdv_z', and 'MaleDiff_z'.
                  For each entry we return coef (log-odds), se (standard error used), pvalue,
                  95% CI (lower, upper) and odds_ratio with its 95% CI.
      - "description": brief interpretation of these quantities in the context of the task.

    The function attempts to use cluster-robust standard errors if it can compute them from the
    original DataFrame (looks for a column named 'dyad' in model_output.model.data.frame).
    If cluster-robust cov cannot be computed, it falls back to the model-provided standard errors
    and p-values.
    """
    import numpy as np
    import pandas as pd
    from math import isfinite, erf, sqrt

    # Provide a minimal stats-like object with norm.cdf available if scipy is not installed
    try:
        from scipy import stats  # noqa: F401
        # stats.norm.cdf will be available from scipy
    except Exception:
        class _Norm:
            @staticmethod
            def cdf(x):
                # Normal cdf via error function
                return 0.5 * (1.0 + erf(x / sqrt(2.0)))
        class _Stats:
            norm = _Norm()
        stats = _Stats()

    # Terms of interest
    terms = ['RelSize_z', 'DistAdv_z', 'RelSize_z:DistAdv_z', 'MaleDiff_z']

    # Get parameter estimates
    try:
        params = model_output.params.copy()
    except Exception as e:
        raise ValueError(f"Cannot access model parameters from provided model_output: {e}")

    # Try to obtain a covariance matrix with cluster-robust adjustment if possible.
    cov = None
    se_used = None
    pvalues = None
    conf_int = None
    used_cluster = False

    # First, check if the fitted result already has cluster cov or cluster bse attached
    try:
        # Some results may expose a cov_cluster attribute (callable or array)
        cov_candidate = getattr(model_output, 'cov_cluster', None)
        if cov_candidate is not None and not callable(cov_candidate):
            cov = cov_candidate
            se_used = np.sqrt(np.diag(cov))
            used_cluster = True
    except Exception:
        cov = None

    # Next, if cov not available, try to compute cluster cov using the original data frame if present
    if cov is None:
        df = None
        try:
            # statsmodels usually stores the original DataFrame in model.data.frame
            df = model_output.model.data.frame
        except Exception:
            df = None

        if isinstance(df, pd.DataFrame) and 'dyad' in df.columns:
            groups = df['dyad']
            try:
                from statsmodels.stats.sandwich_covariance import cov_cluster
                cov = cov_cluster(model_output, groups)
                se_used = np.sqrt(np.diag(cov))
                used_cluster = True
            except Exception:
                cov = None

    # If still no covariance, fall back to model-provided covariance / bse
    if cov is None:
        try:
            cov = model_output.cov_params()
            # model_output.bse might be a Series or ndarray
            se_used = getattr(model_output, 'bse', None)
            if se_used is None:
                se_used = np.sqrt(np.diag(cov))
            used_cluster = False
        except Exception:
            # final fallback: compute se from params and z-stat if available
            try:
                if hasattr(model_output, 'bse'):
                    se_used = model_output.bse
                    cov = np.diag(np.asarray(se_used) ** 2)
                else:
                    raise RuntimeError("No standard error information available.")
            except Exception as e:
                raise RuntimeError(f"Unable to obtain any standard error or covariance info: {e}")

    # Ensure se_used is a numpy array aligned with params order
    try:
        se_used = np.asarray(se_used)
        if se_used.ndim == 0:
            # scalar
            se_used = np.full(len(params), float(se_used))
        elif se_used.shape[0] != len(params):
            # try to align if it's a Series
            if hasattr(model_output, 'bse') and isinstance(model_output.bse, pd.Series):
                se_used = model_output.bse.reindex(params.index).to_numpy()
            else:
                # fallback: attempt to broadcast or raise
                se_used = np.asarray(se_used).flatten()
                if se_used.shape[0] != len(params):
                    se_used = np.full(len(params), np.nan)
    except Exception:
        se_used = np.full(len(params), np.nan)

    # Compute p-values and confidence intervals using the se_used (normal approximation)
    zval = 1.959963984540054  # 97.5th percentile for two-sided 95% CI

    # If model_output provides pvalues and we are not using cluster-robust, reuse them
    if not used_cluster and hasattr(model_output, 'pvalues'):
        try:
            pvalues = model_output.pvalues.copy()
        except Exception:
            try:
                pvalues = pd.Series(model_output.pvalues)
            except Exception:
                pvalues = None
    else:
        pvalues = None

    # If pvalues still None, compute using normal approx and the chosen standard errors
    if pvalues is None:
        try:
            zstats = params / se_used
            # zstats is a Series when params is Series and se_used is array -> results in Series
            # Ensure we turn it into a numpy array or Series consistently
            if isinstance(zstats, pd.Series):
                pvalues = 2 * (1.0 - stats.norm.cdf(np.abs(zstats)))
                pvalues = pd.Series(pvalues, index=params.index)
            else:
                pvalues = 2 * (1.0 - stats.norm.cdf(np.abs(np.asarray(zstats))))
                pvalues = pd.Series(pvalues, index=params.index)
        except Exception:
            # fallback: use model pvalues if available
            if hasattr(model_output, 'pvalues'):
                try:
                    pvalues = pd.Series(model_output.pvalues, index=params.index)
                except Exception:
                    pvalues = pd.Series(model_output.pvalues)
            else:
                # as a last resort set NaNs
                pvalues = pd.Series([np.nan] * len(params), index=params.index)

    # Ensure pvalues is a Series indexed by params.index
    if not isinstance(pvalues, pd.Series):
        try:
            pvalues = pd.Series(pvalues, index=params.index)
        except Exception:
            # generic fallback
            pvalues = pd.Series(list(pvalues)).reindex(range(len(params)))
            pvalues.index = params.index

    # Confidence intervals
    try:
        # params is a Series, se_used is array -> broadcasting yields Series
        ci_lower = params - zval * se_used
        ci_upper = params + zval * se_used
        conf_int = pd.DataFrame({'2.5%': ci_lower, '97.5%': ci_upper})
    except Exception:
        # fallback to NaNs if anything goes wrong
        conf_int = pd.DataFrame({
            '2.5%': pd.Series([np.nan] * len(params), index=params.index),
            '97.5%': pd.Series([np.nan] * len(params), index=params.index)
        })

    # Assemble results for terms of interest
    results = {}
    for term in terms:
        # interaction term in statsmodels is named 'RelSize_z:DistAdv_z' (with colon). If the exact name
        # is not present, try the alternative ordering 'DistAdv_z:RelSize_z'.
        if term not in params.index:
            if ':' in term:
                a, b = term.split(':')
                alt = f"{b}:{a}"
                if alt in params.index:
                    key = alt
                else:
                    key = None
            else:
                key = None
        else:
            key = term

        if key is None or key not in params.index:
            results[term] = {
                'found': False,
                'note': f"Term '{term}' not found in model parameters."
            }
            continue

        coef = float(params.loc[key])
        # Find the position for this key to get se_used
        try:
            pos = params.index.get_loc(key)
            se = float(se_used[pos]) if hasattr(se_used, '__len__') else float(se_used)
        except Exception:
            # fallback: try to get from model_output.bse if Series
            try:
                if hasattr(model_output, 'bse') and isinstance(model_output.bse, pd.Series):
                    se = float(model_output.bse.reindex(params.index).loc[key])
                else:
                    se = float(np.nan)
            except Exception:
                se = float(np.nan)

        # p-value retrieval, prefer labeled index
        try:
            if key in pvalues.index:
                pval = float(pvalues.loc[key])
            else:
                pval = float(pvalues.iloc[pos])
        except Exception:
            pval = float(np.nan)

        try:
            ci_l = float(conf_int.loc[key, '2.5%'])
            ci_u = float(conf_int.loc[key, '97.5%'])
        except Exception:
            ci_l = ci_u = float('nan')

        # odds ratio and its CI
        try:
            orr = float(np.exp(coef))
            or_ci_l = float(np.exp(ci_l))
            or_ci_u = float(np.exp(ci_u))
        except Exception:
            orr = or_ci_l = or_ci_u = float('nan')

        results[term] = {
            'found': True,
            'coef_log_odds': coef,
            'se': se,
            'p_value': pval,
            '95CI_log_odds': (ci_l, ci_u),
            'odds_ratio': orr,
            '95CI_odds_ratio': (or_ci_l, or_ci_u),
            'notes': ("SE/p-values are cluster-robust" if used_cluster else "SE/p-values are model-provided (not cluster-robust)")
        }

    # Prepare a brief description interpreting the signs:
    description_lines = [
        "Extracted coefficients are on the log-odds scale. Positive coefficients mean that an increase",
        "in the predictor is associated with a higher probability that the focal group wins.",
        "- 'RelSize_z' positive: focal group's larger relative size increases win probability.",
        "- 'DistAdv_z' positive: focal group's location advantage (closer to home center) increases win probability.",
        "- 'RelSize_z:DistAdv_z' interaction positive: the effect of relative size on winning increases when the focal group has a stronger location advantage (and vice versa).",
        "- 'MaleDiff_z' controls for differences in adult male numbers."
    ]
    if any(
        v.get('found') and v.get('p_value') is not None and isfinite(v.get('p_value')) and v.get('p_value') < 0.05
        for v in results.values() if isinstance(v, dict) and v.get('found')
    ):
        description_lines.append("Statistical significance (p < 0.05) is reported per-term below.")
    else:
        description_lines.append("No term shows p < 0.05 based on the extracted p-values (if any).")

    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }