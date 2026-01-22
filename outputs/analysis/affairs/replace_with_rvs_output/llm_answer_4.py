def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'HasChildren' on 'affairs_count'
    from the provided model_output dictionary (must contain keys
    'negative_binomial', 'ols', 'overdispersion_stat', 'n_obs').

    Returns a dict with:
      - "object": a dict of numeric results (coefficients, SEs, p-values,
                  95% CIs, IRR and IRR CI, overdispersion, n_obs)
      - "description": a short text interpretation answering whether having
                       children decreases engagement in extramarital affairs.
    """
    import numpy as np

    # Names
    param = 'HasChildren'

    nb_res = model_output.get('negative_binomial', None)
    ols_res = model_output.get('ols', None)
    overdisp = model_output.get('overdispersion_stat', None)
    n_obs = model_output.get('n_obs', None)

    if nb_res is None or ols_res is None:
        raise ValueError("model_output must contain 'negative_binomial' and 'ols' results.")

    # Helper to safely get param, se, pvalue, ci
    def get_param_info(res, name):
        # coefficient
        try:
            coef = float(res.params[name])
        except Exception:
            raise KeyError(f"Parameter '{name}' not found in model results.")
        # standard error
        try:
            se = float(res.bse[name])
        except Exception:
            se = np.nan
        # p-value
        try:
            pval = float(res.pvalues[name])
        except Exception:
            pval = np.nan
        # 95% CI: try conf_int with label; fallback to coef +/- 1.96*se
        try:
            ci_table = res.conf_int()
            # conf_int() may return DataFrame (with .loc) or ndarray
            if hasattr(ci_table, 'loc'):
                ci_vals = ci_table.loc[name].values.astype(float)
            else:
                # assume ordering of params aligns with params index
                # find index of the parameter
                idx = list(res.params.index).index(name)
                ci_vals = np.asarray(ci_table[idx], dtype=float)
        except Exception:
            if not np.isnan(se):
                ci_vals = np.array([coef - 1.96 * se, coef + 1.96 * se], dtype=float)
            else:
                ci_vals = np.array([np.nan, np.nan], dtype=float)

        return coef, se, pval, float(ci_vals[0]), float(ci_vals[1])

    # Extract NB statistics (primary)
    nb_coef, nb_se, nb_p, nb_ci_low, nb_ci_high = get_param_info(nb_res, param)
    # IRR and its CI (exponentiate NB log-coef and CI)
    try:
        nb_irr = float(np.exp(nb_coef))
        nb_irr_ci_low = float(np.exp(nb_ci_low))
        nb_irr_ci_high = float(np.exp(nb_ci_high))
    except Exception:
        nb_irr = nb_irr_ci_low = nb_irr_ci_high = np.nan

    # Extract OLS statistics (robustness check)
    ols_coef, ols_se, ols_p, ols_ci_low, ols_ci_high = get_param_info(ols_res, param)

    # Build numeric result object
    result_object = {
        'n_obs': int(n_obs) if n_obs is not None else None,
        'overdispersion_stat': float(overdisp) if overdisp is not None else None,
        'negative_binomial': {
            'coef_log_count': round(nb_coef, 4),
            'se': round(nb_se, 4) if not np.isnan(nb_se) else None,
            'p_value': round(nb_p, 4) if not np.isnan(nb_p) else None,
            'ci_95_log_count': (round(nb_ci_low, 4), round(nb_ci_high, 4)),
            'incidence_rate_ratio_IRR': round(nb_irr, 4) if not np.isnan(nb_irr) else None,
            'ci_95_IRR': (round(nb_irr_ci_low, 4), round(nb_irr_ci_high, 4)) if not np.isnan(nb_irr) else (None, None)
        },
        'ols_robustness': {
            'coef_count_difference': round(ols_coef, 4),
            'se': round(ols_se, 4) if not np.isnan(ols_se) else None,
            'p_value': round(ols_p, 4) if not np.isnan(ols_p) else None,
            'ci_95': (round(ols_ci_low, 4), round(ols_ci_high, 4))
        }
    }

    # Interpretation logic for final descriptive sentence
    # Use NB as primary.
    direction = 'decrease' if nb_coef < 0 else 'increase' if nb_coef > 0 else 'no change'
    # significance levels
    if not np.isnan(nb_p):
        if nb_p < 0.01:
            sig_text = 'highly statistically significant (p < 0.01)'
        elif nb_p < 0.05:
            sig_text = 'statistically significant (p < 0.05)'
        elif nb_p < 0.1:
            sig_text = 'marginally statistically significant (0.05 <= p < 0.1)'
        else:
            sig_text = 'not statistically significant (p >= 0.1)'
    else:
        sig_text = 'p-value not available'

    # Build descriptive interpretation
    descr_lines = []
    descr_lines.append(
        f"Primary (Negative Binomial) estimate for 'HasChildren': log-count coef = {round(nb_coef,4)}, "
        f"SE = {round(nb_se,4) if not np.isnan(nb_se) else 'NA'}, p = {round(nb_p,4) if not np.isnan(nb_p) else 'NA'}; "
        f"95% CI (log scale) = [{round(nb_ci_low,4)}, {round(nb_ci_high,4)}]."
    )
    descr_lines.append(
        f"Exponentiated (IRR): {round(nb_irr,4) if not np.isnan(nb_irr) else 'NA'} "
        f"with 95% CI = [{round(nb_irr_ci_low,4) if not np.isnan(nb_irr) else 'NA'}, "
        f"{round(nb_irr_ci_high,4) if not np.isnan(nb_irr) else 'NA'}]."
    )
    descr_lines.append(
        f"Interpretation: an IRR < 1 means having children is associated with fewer expected affairs; "
        f"IRR > 1 means more expected affairs. Here the IRR = {round(nb_irr,4) if not np.isnan(nb_irr) else 'NA'} "
        f"which corresponds to a {round((nb_irr-1)*100,2) if not np.isnan(nb_irr) else 'NA'}% "
        f"change in expected number of affairs when comparing those with children to those without."
    )
    descr_lines.append(
        f"Statistical evidence: the NB effect is {sig_text}. Based on this, "
        + (
            f"there is evidence that having children is associated with a {direction} in engagement in extramarital affairs."
            if ('not' not in sig_text)
            else f"there is no statistically significant evidence that having children changes engagement in extramarital affairs."
        )
    )
    descr_lines.append(
        f"OLS robustness check: coef = {round(ols_coef,4)}, p = {round(ols_p,4) if not np.isnan(ols_p) else 'NA'} "
        f"(95% CI = [{round(ols_ci_low,4)}, {round(ols_ci_high,4)}])."
    )
    descr_lines.append(
        f"Model context: n = {result_object['n_obs']}, overdispersion (var/mean) = {round(result_object['overdispersion_stat'],4) if result_object['overdispersion_stat'] is not None else 'NA'} "
        "(value > 1 supports use of Negative Binomial over Poisson)."
    )

    description = " ".join(descr_lines)

    return {
        "object": result_object,
        "description": description
    }