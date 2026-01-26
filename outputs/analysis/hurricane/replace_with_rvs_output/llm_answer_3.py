def extract_final_answer(model_output):
    """
    Extracts statistics for the 'masfem_z' coefficient from the provided model output dict.
    Expects model_output to be the dict returned by the modeling function, e.g.:
      {'neg_binom': <GLMResultsWrapper>, 'ols_log': <RegressionResultsWrapper>}
    Returns a dict with:
      - "object": dict with extracted numeric results for each model (or error message)
      - "description": readable summary interpreting the coefficient(s) relative to the hypothesis
    """
    import numpy as np

    def _extract_from_result(res, name):
        # res: statsmodels results wrapper
        idx = 'masfem_z'
        if not hasattr(res, 'params'):
            return {'error': f'model object for {name} has no .params attribute'}
        params = res.params
        if idx not in params.index:
            return {'error': f"'{idx}' not found in model parameters for {name}"}
        coef = float(params.loc[idx])
        # Some results may not have robust bse/pvalues, so guard accesses
        bse = float(res.bse.loc[idx]) if hasattr(res, 'bse') and idx in res.bse.index else np.nan
        pval = float(res.pvalues.loc[idx]) if hasattr(res, 'pvalues') and idx in res.pvalues.index else np.nan
        # confidence interval (two-column DataFrame)
        try:
            ci_row = res.conf_int().loc[idx].values
            ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan

        out = {
            'coef': coef,
            'se': bse,
            'p_value': pval,
            'ci_95': [ci_lower, ci_upper],
        }

        if name == 'neg_binom':
            # For count model with log link, exponentiate coefficient to get incidence rate ratio (IRR)
            out['irr'] = float(np.exp(coef))
            out['irr_ci_95'] = [float(np.exp(ci_lower)) if not np.isnan(ci_lower) else np.nan,
                                float(np.exp(ci_upper)) if not np.isnan(ci_upper) else np.nan]
            # Interpret direction: positive coef -> IRR > 1 -> higher expected counts
        elif name == 'ols_log':
            # For OLS on log(alldeaths + 1): exponentiated coef is multiplicative effect on (alldeaths+1)
            out['multiplicative_effect_on_y_plus1'] = float(np.exp(coef))
            out['pct_change_on_y_plus1'] = float((np.exp(coef) - 1) * 100.0)
            out['pct_change_ci_95'] = [
                float((np.exp(ci_lower) - 1) * 100.0) if not np.isnan(ci_lower) else np.nan,
                float((np.exp(ci_upper) - 1) * 100.0) if not np.isnan(ci_upper) else np.nan,
            ]
        return out

    results_obj = {}
    summary_parts = []

    # Negative Binomial
    if 'neg_binom' in model_output and model_output.get('neg_binom') is not None:
        try:
            results_obj['neg_binom'] = _extract_from_result(model_output['neg_binom'], 'neg_binom')
        except Exception as e:
            results_obj['neg_binom'] = {'error': f'error extracting neg_binom: {e}'}
    elif 'neg_binom_error' in model_output:
        results_obj['neg_binom'] = {'error': model_output['neg_binom_error']}
    else:
        results_obj['neg_binom'] = {'error': 'no neg_binom result found'}

    # OLS on log outcome
    if 'ols_log' in model_output and model_output.get('ols_log') is not None:
        try:
            results_obj['ols_log'] = _extract_from_result(model_output['ols_log'], 'ols_log')
        except Exception as e:
            results_obj['ols_log'] = {'error': f'error extracting ols_log: {e}'}
    elif 'ols_error' in model_output:
        results_obj['ols_log'] = {'error': model_output['ols_error']}
    else:
        results_obj['ols_log'] = {'error': 'no ols_log result found'}

    # Build an English summary interpreting results (if numeric info available)
    def _interpret_piece(name, piece):
        if 'error' in piece:
            return f"{name}: {piece['error']}"
        coef = piece.get('coef', np.nan)
        p = piece.get('p_value', np.nan)
        ci = piece.get('ci_95', [np.nan, np.nan])
        sign = 'positive' if coef > 0 else ('negative' if coef < 0 else 'null')
        significance = 'statistically significant' if (not np.isnan(p) and p < 0.05) else 'not statistically significant'
        if name == 'neg_binom':
            irr = piece.get('irr', np.nan)
            irr_ci = piece.get('irr_ci_95', [np.nan, np.nan])
            return (f"Negative Binomial: masfem_z coef = {coef:.4g}, SE = {piece.get('se', np.nan):.4g}, "
                    f"95% CI = [{ci[0]:.4g}, {ci[1]:.4g}], p = {p:.4g}. "
                    f"Direction: {sign}. Exponentiated IRR = {irr:.4g} (95% CI [{irr_ci[0]:.4g}, {irr_ci[1]:.4g}]). "
                    f"Interpretation: a 1 SD increase in name femininity is associated with a multiplicative change of {irr:.4g} "
                    f"in expected fatalities. Evidence: {significance}.")
        else:
            pct = piece.get('pct_change_on_y_plus1', np.nan)
            pct_ci = piece.get('pct_change_ci_95', [np.nan, np.nan])
            return (f"OLS on log(y+1): masfem_z coef = {coef:.4g}, SE = {piece.get('se', np.nan):.4g}, "
                    f"95% CI = [{ci[0]:.4g}, {ci[1]:.4g}], p = {p:.4g}. "
                    f"Direction: {sign}. This corresponds to an estimated {pct:.3g}% change in (alldeaths + 1) per 1 SD increase "
                    f"in femininity (95% CI [{pct_ci[0]:.3g}%, {pct_ci[1]:.3g}%]). Evidence: {significance}.")

    summary_parts.append(_interpret_piece('Negative Binomial', results_obj['neg_binom']))
    summary_parts.append(_interpret_piece('OLS', results_obj['ols_log']))

    # Overall verdict relative to the hypothesis:
    verdict = "Insufficient model results to form a verdict."
    try:
        nb = results_obj.get('neg_binom')
        if nb and 'error' not in nb:
            p = nb.get('p_value')
            coef = nb.get('coef')
            if (not np.isnan(p)) and (p < 0.05):
                if coef > 0:
                    verdict = ("Negative Binomial model shows a statistically significant positive association "
                               "— results are consistent with the hypothesis that more feminine names are associated "
                               "with higher fatalities (consistent with fewer precautions).")
                else:
                    verdict = ("Negative Binomial model shows a statistically significant negative association "
                               "— results are inconsistent with the hypothesis.")
            else:
                # fall back to OLS if NB not significant
                ols = results_obj.get('ols_log')
                if ols and 'error' not in ols:
                    p2 = ols.get('p_value')
                    coef2 = ols.get('coef')
                    if (not np.isnan(p2)) and (p2 < 0.05):
                        if coef2 > 0:
                            verdict = ("OLS (log outcome) shows a statistically significant positive association "
                                       "— results are consistent with the hypothesis.")
                        else:
                            verdict = ("OLS (log outcome) shows a statistically significant negative association "
                                       "— results are inconsistent with the hypothesis.")
                    else:
                        verdict = ("Neither model provides statistically significant evidence linking name femininity "
                                   "to fatalities at the conventional p < 0.05 level.")
        else:
            # If NB missing, try OLS
            ols = results_obj.get('ols_log')
            if ols and 'error' not in ols:
                p2 = ols.get('p_value')
                coef2 = ols.get('coef')
                if (not np.isnan(p2)) and (p2 < 0.05):
                    if coef2 > 0:
                        verdict = ("OLS (log outcome) shows a statistically significant positive association "
                                   "— results are consistent with the hypothesis.")
                    else:
                        verdict = ("OLS (log outcome) shows a statistically significant negative association "
                                   "— results are inconsistent with the hypothesis.")
                else:
                    verdict = ("OLS model does not show statistically significant evidence linking name femininity "
                               "to fatalities at the conventional p < 0.05 level.")
    except Exception:
        # keep default insufficient
        pass

    description = "\n".join(summary_parts) + "\n\nOverall verdict: " + verdict

    return {'object': results_obj, 'description': description}