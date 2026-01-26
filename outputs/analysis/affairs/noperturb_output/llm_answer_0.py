def extract_final_answer(model_output):
    """
    Extracts the effect of 'has_children' from fitted model objects returned by the model() function.
    Expects model_output to be a dict-like object with keys: 'ols', 'neg_bin', 'logit' whose values are
    statsmodels fitted result wrappers.

    Returns:
      {
        "object": {
           "ols": {coef, se, pvalue, ci_low, ci_high, interpretation_str, significance_bool, direction_str, transformed (None)},
           "neg_bin": {coef, se, pvalue, ci_low, ci_high, irr, irr_ci, interpretation_str, significance_bool, direction_str},
           "logit": {coef, se, pvalue, ci_low, ci_high, odds_ratio, or_ci, interpretation_str, significance_bool, direction_str}
        },
        "description": "Brief overall interpretation and how to read the returned 'object'."
      }
    """
    import numpy as np

    results_summary = {}

    def _extract(res, param_name='has_children'):
        """
        Safely extract coef, se, pvalue, conf int for param_name from a statsmodels results object.
        Returns tuple (coef, se, pvalue, ci_low, ci_high)
        """
        # params, bse, pvalues are usually pandas Series with index names
        params = getattr(res, "params")
        bse = getattr(res, "bse", None)
        pvalues = getattr(res, "pvalues", None)

        # find index of parameter
        try:
            param_index = list(params.index).index(param_name)
        except Exception:
            raise KeyError(f"Parameter '{param_name}' not found in model params: {list(params.index)}")

        coef = float(params[param_name]) if param_name in params.index else float(params.iloc[param_index])
        se = float(bse[param_name]) if (bse is not None and param_name in bse.index) else float(bse.iloc[param_index]) if bse is not None else None
        pval = float(pvalues[param_name]) if (pvalues is not None and param_name in pvalues.index) else float(pvalues.iloc[param_index]) if pvalues is not None else None

        # conf_int may return array-like; convert to numpy and index by param_index
        ci_arr = np.asarray(res.conf_int())
        ci_low = float(ci_arr[param_index, 0])
        ci_high = float(ci_arr[param_index, 1])

        return coef, se, pval, ci_low, ci_high

    # Process OLS
    if 'ols' in model_output and model_output['ols'] is not None:
        res = model_output['ols']
        try:
            coef, se, pval, ci_low, ci_high = _extract(res, 'has_children')
            direction = 'decrease' if coef < 0 else ('increase' if coef > 0 else 'no effect')
            signif = (pval is not None) and (pval < 0.05)
            interp = (f"OLS: coef={coef:.4f} (SE={se:.4f}), p={pval:.4g}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
                      f"Interpretation: having children is associated with an average {abs(coef):.4f} "
                      f"{'fewer' if coef < 0 else 'more' if coef > 0 else 'no change in'} reported affairs "
                      f"in the past year, controlling for covariates. {'Statistically significant.' if signif else 'Not statistically significant.'}")
            results_summary['ols'] = {
                'coef': coef, 'se': se, 'pvalue': pval, 'ci_low': ci_low, 'ci_high': ci_high,
                'significant_at_0.05': signif, 'direction': direction, 'interpretation': interp,
                'transformed': None
            }
        except Exception as e:
            results_summary['ols'] = {'error': str(e)}

    # Process Negative Binomial
    if 'neg_bin' in model_output and model_output['neg_bin'] is not None:
        res = model_output['neg_bin']
        try:
            coef, se, pval, ci_low, ci_high = _extract(res, 'has_children')
            irr = float(np.exp(coef))
            irr_ci_low = float(np.exp(ci_low))
            irr_ci_high = float(np.exp(ci_high))
            direction = 'decrease' if coef < 0 else ('increase' if coef > 0 else 'no effect')
            signif = (pval is not None) and (pval < 0.05)
            percent_change = (1 - irr) * 100  # if irr < 1 this is percent decrease
            interp = (f"NegBin: coef={coef:.4f} (SE={se:.4f}), p={pval:.4g}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
                      f"Exponentiated (IRR)={irr:.4f}, 95% CI for IRR [{irr_ci_low:.4f}, {irr_ci_high:.4f}]. "
                      f"Interpretation: having children is associated with a multiplicative change of {irr:.4f} in the expected count of affairs "
                      f"(i.e. about {abs(percent_change):.1f}% {'decrease' if irr < 1 else 'increase' if irr > 1 else 'no change'}) controlling for covariates. "
                      f"{'Statistically significant.' if signif else 'Not statistically significant.'}")
            results_summary['neg_bin'] = {
                'coef': coef, 'se': se, 'pvalue': pval, 'ci_low': ci_low, 'ci_high': ci_high,
                'irr': irr, 'irr_ci': (irr_ci_low, irr_ci_high),
                'significant_at_0.05': signif, 'direction': direction, 'interpretation': interp
            }
        except Exception as e:
            results_summary['neg_bin'] = {'error': str(e)}

    # Process Logit
    if 'logit' in model_output and model_output['logit'] is not None:
        res = model_output['logit']
        try:
            coef, se, pval, ci_low, ci_high = _extract(res, 'has_children')
            odds_ratio = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
            direction = 'decrease' if coef < 0 else ('increase' if coef > 0 else 'no effect')
            signif = (pval is not None) and (pval < 0.05)
            interp = (f"Logit: coef={coef:.4f} (SE={se:.4f}), p={pval:.4g}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
                      f"Odds ratio={odds_ratio:.4f}, 95% CI [{or_ci_low:.4f}, {or_ci_high:.4f}]. "
                      f"Interpretation: having children is associated with {'lower' if odds_ratio < 1 else 'higher' if odds_ratio > 1 else 'no change in'} odds of reporting any affair; "
                      f"{'Statistically significant.' if signif else 'Not statistically significant.'}")
            results_summary['logit'] = {
                'coef': coef, 'se': se, 'pvalue': pval, 'ci_low': ci_low, 'ci_high': ci_high,
                'odds_ratio': odds_ratio, 'or_ci': (or_ci_low, or_ci_high),
                'significant_at_0.05': signif, 'direction': direction, 'interpretation': interp
            }
        except Exception as e:
            results_summary['logit'] = {'error': str(e)}

    # Build an overall concise conclusion based on direction and significance across models
    concl_parts = []
    neg_significant_count = 0
    total_models_considered = 0
    for key in ['ols', 'neg_bin', 'logit']:
        entry = results_summary.get(key)
        if entry is None or 'error' in entry:
            continue
        total_models_considered += 1
        if entry.get('significant_at_0.05') and entry.get('direction') == 'decrease':
            neg_significant_count += 1
        # add brief per-model summary
        concl_parts.append(f"{key.upper()}: coef={entry.get('coef'):.4f}, p={entry.get('pvalue'):.4g}, direction={entry.get('direction')}, significant={entry.get('significant_at_0.05')}")

    if total_models_considered == 0:
        overall = "No model results available to draw a conclusion."
    else:
        if neg_significant_count >= 2:
            overall = (f"Consistent evidence across models: {neg_significant_count} of {total_models_considered} models "
                       "show a statistically significant negative association (having children associated with fewer reported affairs).")
        elif neg_significant_count == 1:
            overall = (f"Mixed evidence: {neg_significant_count} of {total_models_considered} model(s) show a statistically significant negative association; "
                       "other models are not statistically significant or show weaker evidence. Interpret with caution.")
        else:
            # check if coefficients are generally negative though not significant
            neg_count = sum(1 for k in ['ols','neg_bin','logit'] if k in results_summary and results_summary[k].get('direction')=='decrease')
            if neg_count == total_models_considered:
                overall = ("All models estimate negative coefficients (having children associated with fewer affairs) but none are statistically significant at 0.05. "
                           "No strong evidence of an effect.")
            else:
                overall = ("No consistent evidence that having children decreases engagement in extramarital affairs (coefficients either not negative or not statistically significant).")

    description = (
        "Returned 'object' is a dict with one entry per model ('ols','neg_bin','logit'). Each entry contains the raw coefficient for 'has_children', "
        "its standard error, p-value, 95% confidence interval, model-specific transformed effect (IRR for NegBin, odds ratio for Logit), "
        "a short interpretation string, and a boolean 'significant_at_0.05'.\n\n"
        "Overall conclusion (based on number of models with negative, statistically significant estimates):\n" + overall + 
        "\n\nPer-model brief summaries:\n" + ("\n".join(concl_parts) if concl_parts else "none")
    )

    return {"object": results_summary, "description": description}