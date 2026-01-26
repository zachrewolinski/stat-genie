def extract_final_answer(model_output):
    """
    Extract key statistics about the 'is_human' effect from the model_output dict
    and produce a concise conclusion about whether modern humans have higher
    AMTL rates after controlling for age, sex, and tooth class.

    Returns a dict with:
      - "object": a dict of extracted numeric results and a boolean/statements
      - "description": a short human-readable interpretation
    """
    import numpy as np

    # Prepare outputs with defaults
    result_obj = {
        'is_human_param_name': None,
        'coef': np.nan,
        'odds_ratio': np.nan,
        'ci_or': [np.nan, np.nan],
        'p_value': np.nan,
        'pred_prob_nonhuman_at_mean': np.nan,
        'pred_prob_human_at_mean': np.nan,
        'dispersion': np.nan,
        'significant': None,
        'conclusion': None
    }

    # Try to get cluster-robust results first, then plain results
    res = model_output.get('glm_result_cluster') or model_output.get('glm_result')

    # Helper to find parameter name referencing is_human
    def _find_param_name(res_obj):
        if res_obj is None:
            return None
        try:
            idx = list(res_obj.params.index)
        except Exception:
            try:
                # if params not available, try model_output provided fields
                return None
            except Exception:
                return None
        # look for exact match or containing 'is_human'
        for name in idx:
            if name == 'is_human' or (isinstance(name, str) and 'is_human' in name):
                return name
        return None

    param_name = _find_param_name(res)
    result_obj['is_human_param_name'] = param_name

    # Extract from statsmodels result object if available
    if param_name is not None and res is not None:
        try:
            coef = float(res.params[param_name])
            result_obj['coef'] = coef
            result_obj['odds_ratio'] = float(np.exp(coef))
            # conf_int on log-odds, then exponentiate
            try:
                ci = res.conf_int().loc[param_name].astype(float)
                ci_or = list(np.exp(ci.values))
                result_obj['ci_or'] = ci_or
            except Exception:
                # leave ci_or as NaNs if conf_int not available
                pass
            # p-value from (robust) results object
            try:
                pval = float(res.pvalues[param_name])
                result_obj['p_value'] = pval
            except Exception:
                pass
        except Exception:
            # fallback: do nothing, will try model_output fields next
            pass

    # Fallback: extract precomputed values from model_output if present
    if np.isnan(result_obj['coef']) and 'coef_is_human' in model_output:
        try:
            result_obj['coef'] = float(model_output.get('coef_is_human'))
        except Exception:
            pass
    if (np.isnan(result_obj['odds_ratio']) or result_obj['odds_ratio'] is None) and 'odds_ratio_is_human' in model_output:
        try:
            result_obj['odds_ratio'] = float(model_output.get('odds_ratio_is_human'))
        except Exception:
            pass
    if (result_obj['ci_or'] == [np.nan, np.nan] or result_obj['ci_or'] is None) and 'ci_or_is_human' in model_output:
        try:
            ci_in = model_output.get('ci_or_is_human')
            # ensure list of two floats
            result_obj['ci_or'] = [float(ci_in[0]), float(ci_in[1])]
        except Exception:
            pass

    # Predicted probabilities and dispersion if provided
    if 'pred_prob_nonhuman_at_mean' in model_output:
        try:
            result_obj['pred_prob_nonhuman_at_mean'] = float(model_output.get('pred_prob_nonhuman_at_mean'))
        except Exception:
            pass
    if 'pred_prob_human_at_mean' in model_output:
        try:
            result_obj['pred_prob_human_at_mean'] = float(model_output.get('pred_prob_human_at_mean'))
        except Exception:
            pass
    if 'dispersion' in model_output:
        try:
            result_obj['dispersion'] = float(model_output.get('dispersion'))
        except Exception:
            pass

    # Decide significance: require a numeric p-value
    p = result_obj['p_value']
    coef = result_obj['coef']
    if not np.isnan(p):
        result_obj['significant'] = bool(p < 0.05)
    else:
        result_obj['significant'] = None  # unknown

    # Formulate a concise conclusion string
    if result_obj['significant'] is True:
        if not np.isnan(coef) and coef > 0:
            concl = ("Yes — after controlling for age, sex, and tooth class, modern humans "
                     "have statistically significantly higher AMTL (coef={:.3f}, OR={:.3f}, p={:.3g})."
                     ).format(coef, result_obj['odds_ratio'], result_obj['p_value'])
        elif not np.isnan(coef) and coef < 0:
            concl = ("No — modern humans have statistically significantly LOWER AMTL after controls "
                     "(coef={:.3f}, OR={:.3f}, p={:.3g})."
                     ).format(coef, result_obj['odds_ratio'], result_obj['p_value'])
        else:
            concl = ("There is a statistically significant difference associated with is_human "
                     "(p={:.3g}), but coefficient direction is unclear.").format(result_obj['p_value'])
    elif result_obj['significant'] is False:
        concl = ("No — there is no statistically significant difference in AMTL between modern humans "
                 "and the non-human primates after controlling for covariates (coef={:.3f}, OR={:.3f}, p={:.3g})."
                 ).format(coef if not np.isnan(coef) else np.nan,
                          result_obj['odds_ratio'] if not np.isnan(result_obj['odds_ratio']) else np.nan,
                          p if not np.isnan(p) else np.nan)
    else:
        # p-value unknown: base conclusion on effect estimate if available
        if not np.isnan(coef):
            concl = ("Unable to determine statistical significance (p-value not available). "
                     "Estimated effect: coef={:.3f}, OR={:.3f}; interpretation should be cautious."
                     ).format(coef, result_obj['odds_ratio'] if not np.isnan(result_obj['odds_ratio']) else np.nan)
        else:
            concl = "Unable to extract necessary statistics to answer the question."

    result_obj['conclusion'] = concl

    # Build description summarizing the key numbers
    desc_parts = []
    desc_parts.append("Effect (log-odds) for is_human: {}".format(
        "NA" if np.isnan(result_obj['coef']) else "{:.4f}".format(result_obj['coef'])))
    desc_parts.append("Odds ratio: {}".format(
        "NA" if np.isnan(result_obj['odds_ratio']) else "{:.3f}".format(result_obj['odds_ratio'])))
    if result_obj['ci_or'] and not all(np.isnan(x) for x in result_obj['ci_or']):
        desc_parts.append("95% CI for OR: [{:.3f}, {:.3f}]".format(result_obj['ci_or'][0], result_obj['ci_or'][1]))
    desc_parts.append("p-value: {}".format("NA" if np.isnan(result_obj['p_value']) else "{:.3g}".format(result_obj['p_value'])))
    if not np.isnan(result_obj['pred_prob_nonhuman_at_mean']) and not np.isnan(result_obj['pred_prob_human_at_mean']):
        desc_parts.append("Predicted AMTL probability at mean covariates — non-human: {:.3%}, human: {:.3%}".format(
            result_obj['pred_prob_nonhuman_at_mean'], result_obj['pred_prob_human_at_mean']))
    if not np.isnan(result_obj['dispersion']):
        desc_parts.append("Dispersion (deviance/df_resid): {:.3f}".format(result_obj['dispersion']))

    description = " | ".join(desc_parts) + " || Conclusion: " + result_obj['conclusion']

    return {'object': result_obj, 'description': description}