def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and incidence-rate/percent-change
    interpretations for the primary predictors related to femininity of hurricane names from the
    provided model_output dict.

    Expects model_output to be a dict with keys:
      - 'nb_model_masfem_z' : statsmodels GLMResultsWrapper (NegativeBinomial) for alldeaths ~ masfem_z + controls
      - 'ols_damage_masfem_z' : statsmodels RegressionResultsWrapper (OLS) for log_ndam15 ~ masfem_z + controls
      - 'nb_model_female_binary' : statsmodels GLMResultsWrapper (NegativeBinomial) for alldeaths ~ FemaleName + controls

    Returns a dict with:
      - "object": dict of extracted numeric results
      - "description": human-readable summary/interpretation of those results in the context of the hypothesis
    """
    import numpy as np

    def _get_conf_int(res, name):
        # statsmodels results.conf_int() may return DataFrame-like or ndarray;
        # handle both robustly.
        try:
            ci = res.conf_int().loc[name].values
        except Exception:
            # fallback: conf_int returns ndarray with rows in same order as params
            try:
                params_index = list(res.params.index)
                i = params_index.index(name)
                ci = res.conf_int()[i]
            except Exception:
                raise KeyError(f"Could not extract confidence interval for parameter '{name}'.")
        return float(ci[0]), float(ci[1])

    def _summarize_coef(res, name, model_type):
        if name not in res.params.index:
            raise KeyError(f"Parameter '{name}' not found in model results.")
        coef = float(res.params[name])
        pval = float(res.pvalues[name])
        ci_low, ci_high = _get_conf_int(res, name)

        summary = {
            'coef': coef,
            'pvalue': pval,
            'ci_95': [ci_low, ci_high]
        }

        if model_type == 'nb_count':
            # For negative binomial (count) model, interpret via incidence rate ratio (IRR)
            irr = float(np.exp(coef))
            irr_ci = [float(np.exp(ci_low)), float(np.exp(ci_high))]
            summary.update({
                'interpretation': 'Incidence Rate Ratio (IRR) for a one-unit change in predictor',
                'IRR': irr,
                'IRR_95_CI': irr_ci,
                'interpretation_text': (
                    "IRR > 1 means higher expected count (deaths); IRR < 1 means lower expected count."
                )
            })
        elif model_type == 'ols_log':
            # For OLS on log outcome, interpret as percent change: (exp(coef)-1)*100
            pct_change = float(np.exp(coef) - 1) * 100.0
            pct_ci = [float(np.exp(ci_low) - 1) * 100.0, float(np.exp(ci_high) - 1) * 100.0]
            summary.update({
                'interpretation': 'Approximate percent change in outcome for a one-unit change in predictor',
                'percent_change': pct_change,
                'percent_change_95_CI': pct_ci,
                'interpretation_text': (
                    "Values > 0 indicate an increase in logged damages (i.e., higher damages); values < 0 indicate a decrease."
                )
            })
        else:
            # Generic
            summary.update({'interpretation': 'raw coefficient (interpretation depends on link/scale)'})

        # Significance label
        summary['significant_0.05'] = pval < 0.05

        return summary

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    results_obj = {}
    desc_lines = []

    # Extract from negative binomial model with continuous masfem_z
    if 'nb_model_masfem_z' in model_output:
        nb_m = model_output['nb_model_masfem_z']
        try:
            nb_summary = _summarize_coef(nb_m, 'masfem_z', model_type='nb_count')
            results_obj['nb_masfem_z'] = nb_summary
            # interpret direction w.r.t hypothesis
            direction = 'positive' if nb_summary['coef'] > 0 else 'negative'
            sig_text = 'statistically significant' if nb_summary['significant_0.05'] else 'not statistically significant'
            desc_lines.append(
                f"Negative binomial (deaths): masfem_z coef = {nb_summary['coef']:.4f} (p = {nb_summary['pvalue']:.3g}), "
                f"IRR = {nb_summary['IRR']:.3f} [{nb_summary['IRR_95_CI'][0]:.3f}, {nb_summary['IRR_95_CI'][1]:.3f}]. "
                f"Coefficient is {direction} and {sig_text}."
            )
        except Exception as e:
            desc_lines.append(f"Could not extract masfem_z from nb_model_masfem_z: {e}")
    else:
        desc_lines.append("nb_model_masfem_z not found in model_output.")

    # Extract from OLS on logged damages
    if 'ols_damage_masfem_z' in model_output:
        ols_m = model_output['ols_damage_masfem_z']
        try:
            ols_summary = _summarize_coef(ols_m, 'masfem_z', model_type='ols_log')
            results_obj['ols_masfem_z'] = ols_summary
            direction = 'positive' if ols_summary['coef'] > 0 else 'negative'
            sig_text = 'statistically significant' if ols_summary['significant_0.05'] else 'not statistically significant'
            desc_lines.append(
                f"OLS (log damages): masfem_z coef = {ols_summary['coef']:.4f} (p = {ols_summary['pvalue']:.3g}), "
                f"approx. percent change = {ols_summary['percent_change']:.2f}% "
                f"[{ols_summary['percent_change_95_CI'][0]:.2f}%, {ols_summary['percent_change_95_CI'][1]:.2f}%]. "
                f"Coefficient is {direction} and {sig_text}."
            )
        except Exception as e:
            desc_lines.append(f"Could not extract masfem_z from ols_damage_masfem_z: {e}")
    else:
        desc_lines.append("ols_damage_masfem_z not found in model_output.")

    # Extract from negative binomial model with binary FemaleName
    if 'nb_model_female_binary' in model_output:
        nb_bin = model_output['nb_model_female_binary']
        try:
            nb_bin_summary = _summarize_coef(nb_bin, 'FemaleName', model_type='nb_count')
            results_obj['nb_female_binary'] = nb_bin_summary
            direction = 'positive' if nb_bin_summary['coef'] > 0 else 'negative'
            sig_text = 'statistically significant' if nb_bin_summary['significant_0.05'] else 'not statistically significant'
            desc_lines.append(
                f"Negative binomial (deaths) - FemaleName: coef = {nb_bin_summary['coef']:.4f} (p = {nb_bin_summary['pvalue']:.3g}), "
                f"IRR = {nb_bin_summary['IRR']:.3f} [{nb_bin_summary['IRR_95_CI'][0]:.3f}, {nb_bin_summary['IRR_95_CI'][1]:.3f}]. "
                f"Coefficient is {direction} and {sig_text}."
            )
        except Exception as e:
            desc_lines.append(f"Could not extract FemaleName from nb_model_female_binary: {e}")
    else:
        desc_lines.append("nb_model_female_binary not found in model_output.")

    # High-level conclusion on hypothesis consistency
    # Hypothesis: more feminine names -> fewer precautions -> higher fatalities/damages.
    # We interpret "supports hypothesis" if coefficient for masfem_z or FemaleName is positive and statistically significant.
    support_msgs = []
    try:
        # check continuous nb model first
        nb_ok = 'nb_masfem_z' in results_obj
        if nb_ok and results_obj['nb_masfem_z']['significant_0.05'] and results_obj['nb_masfem_z']['coef'] > 0:
            support_msgs.append("Primary count model (nb_model_masfem_z) shows a positive, statistically significant association consistent with the hypothesis.")
        elif nb_ok:
            support_msgs.append("Primary count model does not provide statistically significant evidence in support of the hypothesis." if not results_obj['nb_masfem_z']['significant_0.05'] else "Primary count model shows a coefficient in the opposite direction of the hypothesis.")

        # check OLS
        ols_ok = 'ols_masfem_z' in results_obj
        if ols_ok and results_obj['ols_masfem_z']['significant_0.05'] and results_obj['ols_masfem_z']['coef'] > 0:
            support_msgs.append("Robustness OLS (log damages) shows a positive, statistically significant association consistent with the hypothesis.")
        elif ols_ok:
            support_msgs.append("Robustness OLS does not provide statistically significant evidence in support of the hypothesis." if not results_obj['ols_masfem_z']['significant_0.05'] else "Robustness OLS shows a coefficient in the opposite direction of the hypothesis.")

        # check binary female
        bin_ok = 'nb_female_binary' in results_obj
        if bin_ok and results_obj['nb_female_binary']['significant_0.05'] and results_obj['nb_female_binary']['coef'] > 0:
            support_msgs.append("Binary female-name model shows a positive, statistically significant association consistent with the hypothesis.")
        elif bin_ok:
            support_msgs.append("Binary female-name model does not provide statistically significant evidence in support of the hypothesis." if not results_obj['nb_female_binary']['significant_0.05'] else "Binary female-name model shows a coefficient in the opposite direction of the hypothesis.")
    except Exception:
        support_msgs.append("Could not form a clear conclusion programmatically; inspect the extracted statistics in 'object' for manual interpretation.")

    description = "\n".join(desc_lines) + "\n\nConclusion summary:\n" + "\n".join(support_msgs)

    return {
        "object": results_obj,
        "description": description
    }