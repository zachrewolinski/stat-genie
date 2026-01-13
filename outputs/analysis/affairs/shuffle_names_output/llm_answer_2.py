def extract_final_answer(model_output):
    """
    Extract statistics about the effect of HasChildren from model_output.
    Expects model_output to be a dict with keys 'logit' and/or 'ols' whose
    values are fitted statsmodels result objects (or None).
    Returns a dict with:
      - "object": dict with extracted stats for 'logit' and 'ols' (or None)
      - "description": brief interpretation in context ("does having children decrease affairs?")
    """
    import math

    def _extract_from_result(res, target_substr='HasChildren'):
        """
        Extract coef, se, pvalue, conf_int for parameter whose name contains target_substr.
        Returns None if res is None or parameter not found.
        """
        if res is None:
            return None

        # Ensure params-like attributes exist
        try:
            params = getattr(res, 'params', None)
            pvalues = getattr(res, 'pvalues', None)
            bse = getattr(res, 'bse', None)
            # conf_int may be a method
            try:
                conf = res.conf_int()
            except Exception:
                conf = None
            if params is None:
                return {'error': 'Result object has no params attribute'}
        except Exception as e:
            return {'error': f'Error accessing result attributes: {e}'}

        # params may be a Series with index; coerce to dict-like
        try:
            param_index = list(params.index)
        except Exception:
            # if params has no index, can't find parameter by name
            return {'error': 'params exists but has no index (cannot find parameter name)'}

        # find a parameter name that contains target_substr
        matches = [name for name in param_index if target_substr in str(name)]
        if not matches:
            return {'error': f"No parameter containing '{target_substr}' found",
                    'available_params': param_index}

        name = matches[0]

        # extract numeric values, guard against missing pvalues/bse/conf
        try:
            coef = float(params[name])
        except Exception:
            return {'error': f'Could not convert coefficient for {name} to float'}

        pval = None
        if pvalues is not None and name in pvalues.index:
            try:
                pval = float(pvalues[name])
            except Exception:
                pval = None

        se = None
        if bse is not None and name in bse.index:
            try:
                se = float(bse[name])
            except Exception:
                se = None

        ci = None
        if conf is not None:
            try:
                # conf may be DataFrame-like with rows indexed by param names and two columns
                if name in conf.index:
                    lower = float(conf.loc[name].iloc[0])
                    upper = float(conf.loc[name].iloc[1])
                    ci = (lower, upper)
            except Exception:
                ci = None

        return {
            'param_name': name,
            'coef': coef,
            'se': se,
            'pvalue': pval,
            'conf_int': ci
        }

    result_summary = {'logit': None, 'ols': None}

    # Extract for logit
    logit_res = model_output.get('logit') if isinstance(model_output, dict) else None
    logit_stats = _extract_from_result(logit_res, 'HasChildren')
    if logit_stats and 'error' not in (logit_stats or {}):
        # add odds ratio and OR CI if possible
        coef = logit_stats['coef']
        logit_stats['odds_ratio'] = math.exp(coef)
        if logit_stats.get('conf_int') is not None:
            ci = logit_stats['conf_int']
            logit_stats['odds_ratio_conf_int'] = (math.exp(ci[0]), math.exp(ci[1]))
    result_summary['logit'] = logit_stats

    # Extract for ols
    ols_res = model_output.get('ols') if isinstance(model_output, dict) else None
    ols_stats = _extract_from_result(ols_res, 'HasChildren')
    result_summary['ols'] = ols_stats

    # Build human-readable description
    def interpret_logit(s):
        if s is None:
            return "No logistic model available."
        if 'error' in s:
            return "Logistic extraction error: " + s['error']
        coef = s['coef']
        p = s['pvalue']
        orr = s['odds_ratio']
        orc = s.get('odds_ratio_conf_int')
        # significance threshold: 0.05 (conventional)
        if p is None:
            sig_text = "p-value not available"
        else:
            sig_text = ("statistically significant (p = {:.3g})".format(p)
                        if p < 0.05 else "not statistically significant (p = {:.3g})".format(p))
        # direction
        if coef < 0:
            direction = "Having children is associated with LOWER odds of reporting an affair."
            pct = (1 - orr) * 100
            change_text = "Odds ratio = {:.3f} ({:+.1f}% change in odds)".format(orr, -pct)
        else:
            direction = "Having children is associated with HIGHER odds of reporting an affair."
            pct = (orr - 1) * 100
            change_text = "Odds ratio = {:.3f} ({:+.1f}% change in odds)".format(orr, pct)
        ci_text = ("; OR 95% CI = [{:.3f}, {:.3f}]".format(orc[0], orc[1]) if orc is not None else "")
        return "Logistic: {} {}. {}{}".format(direction, sig_text, change_text, ci_text)

    def interpret_ols(s):
        if s is None:
            return "No OLS robustness model available."
        if 'error' in s:
            return "OLS extraction error: " + s['error']
        coef = s['coef']
        p = s['pvalue']
        if p is None:
            sig_text = "p-value not available"
        else:
            sig_text = ("statistically significant (p = {:.3g})".format(p)
                        if p < 0.05 else "not statistically significant (p = {:.3g})".format(p))
        if coef < 0:
            direction = "Having children is associated with a decrease in the coded affair frequency."
        else:
            direction = "Having children is associated with an increase in the coded affair frequency."
        # coef units: coded frequency (0,1,2,3,7,12)
        return "OLS (robust SE): {} {}. Coefficient = {:.3g}.".format(direction, sig_text, coef)

    # Compose final description combining available information
    if (result_summary['logit'] is None or ('error' in result_summary['logit'] and result_summary['logit']['error'].startswith("No parameter"))) \
       and (result_summary['ols'] is None or ('error' in result_summary['ols'] and result_summary['ols']['error'].startswith("No parameter"))):
        description = ("No fitted model parameters for 'HasChildren' were found in the provided model_output. "
                       "This usually means the model did not fit or the HasChildren variable was not included/was dropped.")
    else:
        parts = []
        parts.append(interpret_logit(result_summary['logit']))
        parts.append(interpret_ols(result_summary['ols']))
        description = " ".join(parts)

    return {
        "object": result_summary,
        "description": description
    }