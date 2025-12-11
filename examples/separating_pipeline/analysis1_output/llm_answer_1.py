def extract_final_answer(model_output):
    """
    Extracts statistics for the effect of 'masfem_z' from the model_output dictionary.
    Returns a dict with keys 'object' (detailed extracted stats) and 'description' (brief interpretation).
    """
    import numpy as np
    from math import exp
    from scipy.stats import norm, t as student_t

    def _get_result_and_params(res_obj):
        # res_obj may be a wrapper or raw statsmodels result
        if res_obj is None:
            return None, None
        # params usually available as a pandas Series or numpy array-like with index
        params = getattr(res_obj, 'params', None)
        return res_obj, params

    def _get_param_index(params, name):
        # params may be a pandas Series (has index) or numpy array; try to find index position
        if params is None:
            return None
        try:
            # pandas Series
            return list(params.index).index(name)
        except Exception:
            try:
                # params as dict-like
                return list(params.keys()).index(name)
            except Exception:
                return None

    def _get_se(res_obj, params, name):
        # Try multiple ways to obtain a (robust) standard error for parameter `name`
        # 1) If res_obj has .bse attribute and it's indexable by name or position
        bse = getattr(res_obj, 'bse', None)
        if bse is not None:
            try:
                # bse as pandas Series
                return float(bse[name])
            except Exception:
                try:
                    # bse as numpy array, align by position with params
                    idx = _get_param_index(params, name)
                    if idx is not None:
                        return float(np.asarray(bse)[idx])
                except Exception:
                    pass
        # 2) Try cov_params (may be robust covariance if wrapper stored it)
        cov = None
        try:
            cov = res_obj.cov_params()
        except Exception:
            # maybe the underlying object has cov_params
            try:
                cov = getattr(res_obj, '_res', res_obj).cov_params()
            except Exception:
                cov = None
        if cov is not None:
            try:
                # cov might be a DataFrame or ndarray
                try:
                    return float(np.sqrt(cov.loc[name, name]))
                except Exception:
                    idx = _get_param_index(params, name)
                    if idx is not None:
                        return float(np.sqrt(np.asarray(cov)[idx, idx]))
            except Exception:
                pass
        # 3) As a last resort, try to use the underlying result's bse if wrapper delegates
        try:
            underlying = getattr(res_obj, '_res', None)
            if underlying is not None:
                ubse = getattr(underlying, 'bse', None)
                if ubse is not None:
                    try:
                        return float(ubse[name])
                    except Exception:
                        idx = _get_param_index(params, name)
                        if idx is not None:
                            return float(np.asarray(ubse)[idx])
        except Exception:
            pass
        return None

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # Main extraction for 'masfem_z' on Deaths (negative-binomial result)
    nb_res_obj = model_output.get('nb_results')
    nb_res, nb_params = _get_result_and_params(nb_res_obj)
    nb_stats = None
    nb_interpretation = None

    if nb_res is not None and nb_params is not None:
        name = 'masfem_z'
        # coefficient
        try:
            coef = _safe_float(nb_params[name])
        except Exception:
            # if params is array-like without names, try positional lookup
            coef = None
            try:
                idx = _get_param_index(nb_params, name)
                if idx is not None:
                    coef = _safe_float(np.asarray(nb_params)[idx])
            except Exception:
                coef = None

        se = _get_se(nb_res, nb_params, name)
        # If no robust se available but result has z/pvalues, try to extract pvalue directly
        pval = None
        stat = None
        ci_lower = ci_upper = None
        irr = irr_ci_low = irr_ci_high = None

        if coef is not None and se is not None:
            stat = coef / se
            pval = float(2.0 * norm.sf(abs(stat)))
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            irr = float(exp(coef))
            irr_ci_low = float(exp(ci_lower))
            irr_ci_high = float(exp(ci_upper))
        else:
            # Try to get pvalue from result object if available
            try:
                # statsmodels result may provide pvalues Series or table
                pvals = getattr(nb_res, 'pvalues', None)
                if pvals is not None:
                    pval = float(pvals[name])
                # confidence intervals
                try:
                    ci = nb_res.conf_int()
                    if name in list(ci.index):
                        ci_lower = float(ci.loc[name, 0])
                        ci_upper = float(ci.loc[name, 1])
                        if coef is None:
                            coef = _safe_float(nb_params[name]) if nb_params is not None else None
                        if coef is not None:
                            irr = float(exp(coef))
                            irr_ci_low = float(exp(ci_lower))
                            irr_ci_high = float(exp(ci_upper))
                except Exception:
                    pass
            except Exception:
                pass

        nb_stats = {
            'coef_log_deaths': coef,
            'se_log_deaths': se,
            'z_or_t': stat,
            'p_value': pval,
            '95ci_log_deaths': (ci_lower, ci_upper),
            'incidence_rate_ratio': irr,
            'irr_95ci': (irr_ci_low, irr_ci_high)
        }

        # Interpretation: positive coef => higher expected counts (more deaths) for more feminine names.
        if coef is not None and pval is not None:
            if pval < 0.05:
                if coef > 0:
                    nb_interpretation = "Statistically significant positive association: more feminine names -> higher deaths (supports hypothesis)."
                elif coef < 0:
                    nb_interpretation = "Statistically significant negative association: more feminine names -> fewer deaths (contradicts hypothesis)."
                else:
                    nb_interpretation = "No effect (coef = 0)."
            else:
                nb_interpretation = "No statistically significant association detected (p >= 0.05); evidence is inconclusive."
        else:
            nb_interpretation = "Could not compute a complete inference (missing coefficient or standard error/p-value)."

    else:
        nb_stats = None
        nb_interpretation = "Negative-binomial result not present."

    # Secondary extraction for OLS on log_damage (if present)
    ols_res_obj = model_output.get('ols_damage_results')
    ols_res, ols_params = _get_result_and_params(ols_res_obj)
    ols_stats = None
    ols_interpretation = None

    if ols_res is not None and ols_params is not None:
        name = 'masfem_z'
        try:
            coef = _safe_float(ols_params[name])
        except Exception:
            coef = None
            try:
                idx = _get_param_index(ols_params, name)
                if idx is not None:
                    coef = _safe_float(np.asarray(ols_params)[idx])
            except Exception:
                coef = None

        se = _get_se(ols_res, ols_params, name)
        tstat = None
        pval = None
        ci_lower = ci_upper = None
        if coef is not None and se is not None:
            # use t-distribution with df_resid if available
            df_resid = getattr(ols_res, 'df_resid', None)
            try:
                df = int(df_resid) if df_resid is not None else None
            except Exception:
                df = None
            tstat = coef / se
            if df is not None:
                pval = float(2.0 * student_t.sf(abs(tstat), df))
                crit = float(student_t.ppf(0.975, df))
            else:
                pval = float(2.0 * norm.sf(abs(tstat)))
                crit = 1.96
            ci_lower = coef - crit * se
            ci_upper = coef + crit * se
        else:
            # try to pull p-values/conf-int from the result if present
            try:
                pvals = getattr(ols_res, 'pvalues', None)
                if pvals is not None and name in list(pvals.index):
                    pval = float(pvals[name])
                ci = None
                try:
                    ci = ols_res.conf_int()
                except Exception:
                    ci = None
                if ci is not None and name in list(ci.index):
                    ci_lower = float(ci.loc[name, 0])
                    ci_upper = float(ci.loc[name, 1])
            except Exception:
                pass

        ols_stats = {
            'coef_log_damage': coef,
            'se_log_damage': se,
            't_stat': tstat,
            'p_value': pval,
            '95ci_log_damage': (ci_lower, ci_upper)
        }

        if coef is not None and pval is not None:
            if pval < 0.05:
                if coef > 0:
                    ols_interpretation = "Statistically significant positive association: more feminine names -> higher logged damage (supports hypothesis)."
                elif coef < 0:
                    ols_interpretation = "Statistically significant negative association: more feminine names -> lower logged damage (contradicts hypothesis)."
                else:
                    ols_interpretation = "No effect (coef = 0)."
            else:
                ols_interpretation = "No statistically significant association detected (p >= 0.05); evidence is inconclusive."
        else:
            ols_interpretation = "Could not compute a complete inference for OLS (missing coef or se/p-value)."
    else:
        ols_stats = None
        ols_interpretation = "OLS result on log_damage not present."

    # Final overall conclusion based on primary model (nb)
    if nb_stats is not None and nb_stats.get('coef_log_deaths') is not None and nb_stats.get('p_value') is not None:
        coef = nb_stats['coef_log_deaths']
        pval = nb_stats['p_value']
        if pval < 0.05 and coef > 0:
            final_text = "Yes — primary (negative-binomial) model shows a statistically significant positive association between femininity of name and deaths (supports hypothesis)."
            final_bool = True
        elif pval < 0.05 and coef < 0:
            final_text = "No — primary model shows a statistically significant negative association (contradicts hypothesis)."
            final_bool = False
        else:
            final_text = "Inconclusive — primary model does not provide statistically significant evidence for the hypothesis (p >= 0.05)."
            final_bool = None
    else:
        final_text = "Could not determine final conclusion from primary model (missing stats)."
        final_bool = None

    output_object = {
        'nb_stats': nb_stats,
        'nb_interpretation': nb_interpretation,
        'ols_stats': ols_stats,
        'ols_interpretation': ols_interpretation,
        'final_conclusion_text': final_text,
        'final_conclusion_boolean_supports_hypothesis': final_bool
    }

    description = (
        "Extracted coefficient, (robust) standard error, test statistic, p-value, and 95% CI "
        "for the predictor 'masfem_z' from the negative-binomial model predicting Deaths (primary), "
        "and from the OLS on log_damage (secondary) when available. Positive coefficient on the NB model "
        "means more feminine names are associated with higher expected death counts (which would support the "
        "hypothesis that more feminine names lead to fewer precautions / greater harm). The final_conclusion_boolean_supports_hypothesis "
        "field is True if the NB result shows coef>0 with p<0.05, False if coef<0 with p<0.05, and None if inconclusive or missing."
    )

    return {"object": output_object, "description": description}