def extract_final_answer(model_output):
    """
    Extracts key statistics about the femininity variables from fitted models.

    Returns a dictionary with:
      - "object": dict with extracted numeric statistics for each model/variable
      - "description": short interpretation of those statistics in the task context
    """
    import numpy as np

    def _get_param_stats(mod, param):
        """Return (coef, pvalue, ci_low, ci_high) for parameter name `param` from statsmodels result."""
        # params and pvalues should support index access by parameter name
        coef = float(mod.params[param])
        pval = float(mod.pvalues[param])
        ci_mat = mod.conf_int()
        # conf_int may be a DataFrame or ndarray; handle both
        if hasattr(ci_mat, "loc"):
            ci_low, ci_high = map(float, ci_mat.loc[param].values)
        else:
            # find index position of parameter in params
            idx = list(mod.params.index).index(param)
            ci_low, ci_high = map(float, ci_mat[idx])
        return coef, pval, ci_low, ci_high

    def _safe_round(x, nd=4):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [round(float(v), nd) for v in x]
        else:
            return round(float(x), nd)

    results_output = {}

    # 1) Negative Binomial / GLM: binary FemaleName
    try:
        m_bin = model_output.get('nb_model_female_binary')
        if m_bin is None:
            raise KeyError("nb_model_female_binary not found in model_output")
        coef_b, p_b, ci_low_b, ci_high_b = _get_param_stats(m_bin, 'FemaleName')
        irr_b = np.exp(coef_b)
        irr_ci_b = [np.exp(ci_low_b), np.exp(ci_high_b)]
        results_output['female_binary'] = {
            'coef_log_count_FemaleName': _safe_round(coef_b, 5),
            'pvalue': _safe_round(p_b, 5),
            'ci_95_log_count': _safe_round([ci_low_b, ci_high_b], 5),
            'IRR (exp(coef))': _safe_round(irr_b, 5),
            'IRR_95_CI': _safe_round(irr_ci_b, 5),
            'interpretation_brief': (
                "IRR < 1 means female-named storms are associated with fewer expected deaths; "
                "IRR > 1 means the opposite."
            )
        }
    except Exception as e:
        results_output['female_binary'] = {'error': str(e)}

    # 2) Negative Binomial / GLM: continuous NameFem_z
    try:
        m_cont = model_output.get('nb_model_femscore')
        if m_cont is None:
            raise KeyError("nb_model_femscore not found in model_output")
        coef_c, p_c, ci_low_c, ci_high_c = _get_param_stats(m_cont, 'NameFem_z')
        irr_c = np.exp(coef_c)
        irr_ci_c = [np.exp(ci_low_c), np.exp(ci_high_c)]
        results_output['femscore_count'] = {
            'coef_log_count_NameFem_z': _safe_round(coef_c, 5),
            'pvalue': _safe_round(p_c, 5),
            'ci_95_log_count': _safe_round([ci_low_c, ci_high_c], 5),
            'IRR_per_SD_increase': _safe_round(irr_c, 5),
            'IRR_95_CI': _safe_round(irr_ci_c, 5),
            'interpretation_brief': (
                "IRR < 1 for NameFem_z means a one-standard-deviation increase in perceived femininity "
                "is associated with a lower expected death count; IRR > 1 means higher expected deaths."
            )
        }
    except Exception as e:
        results_output['femscore_count'] = {'error': str(e)}

    # 3) OLS on log damage with NameFem_z (robust SEs)
    try:
        m_ols = model_output.get('ols_logdamage')
        if m_ols is None:
            raise KeyError("ols_logdamage not found in model_output")
        coef_o, p_o, ci_low_o, ci_high_o = _get_param_stats(m_ols, 'NameFem_z')
        # For log damage: percent change approximation = (exp(coef)-1)*100
        pct_change = (np.exp(coef_o) - 1.0) * 100.0
        pct_ci = [(np.exp(ci_low_o) - 1.0) * 100.0, (np.exp(ci_high_o) - 1.0) * 100.0]
        results_output['femscore_logdamage'] = {
            'coef_log_damage_NameFem_z': _safe_round(coef_o, 5),
            'pvalue': _safe_round(p_o, 5),
            'ci_95_log_damage': _safe_round([ci_low_o, ci_high_o], 5),
            'percent_change_per_SD': _safe_round(pct_change, 3),
            'percent_change_95_CI': _safe_round(pct_ci, 3),
            'interpretation_brief': (
                "Negative percent_change means higher femininity predicts lower damage (in percent); "
                "positive means higher damage."
            )
        }
    except Exception as e:
        results_output['femscore_logdamage'] = {'error': str(e)}

    # Summary interpretation: decide whether results support the hypothesis
    summary_lines = []
    try:
        # Female binary: support if coef < 0 and p < 0.05
        fb = results_output.get('female_binary')
        if 'error' not in fb:
            supports_fb = (fb['coef_log_count_FemaleName'] < 0) and (fb['pvalue'] < 0.05)
            summary_lines.append(
                f"Binary FemaleName: coef={fb['coef_log_count_FemaleName']}, p={fb['pvalue']}. "
                f"IRR={fb['IRR (exp(coef))']} (95% CI {fb['IRR_95_CI']}). "
                + ("Supports hypothesis." if supports_fb else "Does not support hypothesis.")
            )
    except Exception:
        pass

    try:
        fc = results_output.get('femscore_count')
        if 'error' not in fc:
            supports_fc = (fc['coef_log_count_NameFem_z'] < 0) and (fc['pvalue'] < 0.05)
            summary_lines.append(
                f"Continuous NameFem_z (deaths): coef={fc['coef_log_count_NameFem_z']}, p={fc['pvalue']}. "
                f"IRR per SD={fc['IRR_per_SD_increase']} (95% CI {fc['IRR_95_CI']}). "
                + ("Supports hypothesis." if supports_fc else "Does not support hypothesis.")
            )
    except Exception:
        pass

    try:
        fo = results_output.get('femscore_logdamage')
        if 'error' not in fo:
            supports_fo = (fo['coef_log_damage_NameFem_z'] < 0) and (fo['pvalue'] < 0.05)
            summary_lines.append(
                f"Continuous NameFem_z (log damage): coef={fo['coef_log_damage_NameFem_z']}, p={fo['pvalue']}. "
                f"Estimated % change per SD = {fo['percent_change_per_SD']}% "
                f"(95% CI {fo['percent_change_95_CI']}). "
                + ("Supports hypothesis." if supports_fo else "Does not support hypothesis.")
            )
    except Exception:
        pass

    description = "Extracted parameter estimates for the femininity variables. " + (" ".join(summary_lines) if summary_lines else "No valid summary available (errors encountered).")

    return {'object': results_output, 'description': description}