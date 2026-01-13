def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'HasChildren' from a two-part model output.
    Expects model_output to be a dict with keys 'logit_model' and 'ols_positive_model'.
    Returns a dict with keys:
      - "object": dict with extracted numeric results (or None if not available)
      - "description": textual interpretation of those results in context
    """
    import numpy as np

    out = {
        'logit': None,
        'ols_positive': None
    }

    descriptions = []

    def _extract_from_result(res, varname='HasChildren'):
        """Helper to extract coef, pval, conf int from a statsmodels result-like object."""
        if res is None:
            return None
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            conf = None
            try:
                conf = res.conf_int()
            except Exception:
                conf = None

            if params is None or pvalues is None:
                # Not a recognized statsmodels result object
                return None

            if varname in params.index:
                coef = float(params.loc[varname])
                pval = float(pvalues.loc[varname])
                if conf is not None:
                    try:
                        ci_low, ci_high = conf.loc[varname].astype(float).tolist()
                    except Exception:
                        # fallback to positional
                        idx = list(params.index).index(varname)
                        ci_low = float(conf.iloc[idx, 0])
                        ci_high = float(conf.iloc[idx, 1])
                else:
                    ci_low, ci_high = (None, None)

                # number of observations if available
                nobs = None
                if hasattr(res, 'nobs'):
                    try:
                        nobs = int(res.nobs)
                    except Exception:
                        nobs = None

                return {
                    'coef': coef,
                    'pvalue': pval,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    'nobs': nobs
                }
            else:
                # variable not present in model
                return None
        except Exception:
            return None

    # Extract for logit (probability of any affair)
    logit_res = model_output.get('logit_model')
    logit_stats = _extract_from_result(logit_res, 'HasChildren')
    if logit_stats is not None:
        # compute odds ratio and its CI if possible
        coef = logit_stats['coef']
        ci_low = logit_stats['ci_lower']
        ci_high = logit_stats['ci_upper']
        odds_ratio = float(np.exp(coef))
        or_ci = (None, None)
        if ci_low is not None and ci_high is not None:
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        logit_stats.update({
            'odds_ratio': odds_ratio,
            'odds_ratio_ci_lower': or_ci[0],
            'odds_ratio_ci_upper': or_ci[1]
        })
        out['logit'] = logit_stats

        # Interpret sign and significance
        if logit_stats['pvalue'] < 0.05:
            if odds_ratio < 1:
                descriptions.append(
                    f"Logistic model: Having children is associated with LOWER odds of any extramarital affair "
                    f"(odds ratio = {odds_ratio:.3f}, 95% CI [{or_ci[0]:.3f}, {or_ci[1]:.3f}], p = {logit_stats['pvalue']:.3g})."
                )
            else:
                descriptions.append(
                    f"Logistic model: Having children is associated with HIGHER odds of any extramarital affair "
                    f"(odds ratio = {odds_ratio:.3f}, 95% CI [{or_ci[0]:.3f}, {or_ci[1]:.3f}], p = {logit_stats['pvalue']:.3g})."
                )
        else:
            # not statistically significant
            if odds_ratio < 1:
                descriptions.append(
                    f"Logistic model: Point estimate suggests LOWER odds of any affair with children (odds ratio = {odds_ratio:.3f}), "
                    f"but this is not statistically significant (p = {logit_stats['pvalue']:.3g})."
                )
            else:
                descriptions.append(
                    f"Logistic model: Point estimate suggests HIGHER odds of any affair with children (odds ratio = {odds_ratio:.3f}), "
                    f"but this is not statistically significant (p = {logit_stats['pvalue']:.3g})."
                )
    else:
        descriptions.append("Logistic model: No usable estimate for 'HasChildren' (model missing or variable not present).")

    # Extract for OLS on log(count) among positives
    ols_res = model_output.get('ols_positive_model')
    ols_stats = _extract_from_result(ols_res, 'HasChildren')
    if ols_stats is not None:
        coef = ols_stats['coef']
        ci_low = ols_stats['ci_lower']
        ci_high = ols_stats['ci_upper']
        # Interpret in percent change terms: (exp(beta)-1)*100
        pct_change = (np.exp(coef) - 1) * 100.0
        pct_ci_lower = None
        pct_ci_upper = None
        if ci_low is not None and ci_high is not None:
            pct_ci_lower = (np.exp(ci_low) - 1) * 100.0
            pct_ci_upper = (np.exp(ci_high) - 1) * 100.0

        ols_stats.update({
            'percent_change': float(pct_change),
            'percent_change_ci_lower': float(pct_ci_lower) if pct_ci_lower is not None else None,
            'percent_change_ci_upper': float(pct_ci_upper) if pct_ci_upper is not None else None
        })
        out['ols_positive'] = ols_stats

        # Interpretation
        if ols_stats['pvalue'] < 0.05:
            if pct_change < 0:
                descriptions.append(
                    f"Positive-count OLS: Among those reporting an affair, having children is associated with a LOWER reported frequency "
                    f"({pct_change:.1f}% change; 95% CI [{pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%], p = {ols_stats['pvalue']:.3g})."
                )
            else:
                descriptions.append(
                    f"Positive-count OLS: Among those reporting an affair, having children is associated with a HIGHER reported frequency "
                    f"({pct_change:.1f}% change; 95% CI [{pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%], p = {ols_stats['pvalue']:.3g})."
                )
        else:
            if pct_change < 0:
                descriptions.append(
                    f"Positive-count OLS: Point estimate suggests a lower frequency among those with children ({pct_change:.1f}%), "
                    f"but not statistically significant (p = {ols_stats['pvalue']:.3g})."
                )
            else:
                descriptions.append(
                    f"Positive-count OLS: Point estimate suggests a higher frequency among those with children ({pct_change:.1f}%), "
                    f"but not statistically significant (p = {ols_stats['pvalue']:.3g})."
                )
    else:
        descriptions.append("Positive-count OLS: No usable estimate for 'HasChildren' (model missing, too few positives, or variable not present).")

    # If both models missing
    if out['logit'] is None and out['ols_positive'] is None:
        final_description = (
            "Neither the logistic model for any affair nor the OLS model for positive affair counts produced usable estimates "
            "for the effect of 'HasChildren'. Cannot determine from these models whether having children decreases engagement in extramarital affairs."
        )
        return {
            "object": None,
            "description": final_description
        }

    # Consolidate description text
    final_description = " ".join(descriptions)

    return {
        "object": out,
        "description": final_description
    }