def extract_final_answer(model_output):
    """
    Extracts key statistics from the model_output dictionary returned by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing numeric results for NameFemininity and FemaleName
                  from the primary (negative-binomial / Poisson fallback) model and the
                  robustness OLS model on log-deaths.
      - "description": a short interpretation of those results in the context of the hypothesis.

    The primary test for the hypothesis uses the NB/Poisson model coefficient and p-value
    for 'NameFemininity' (positive & p<0.05 supports the hypothesis). The OLS result
    on LogDeaths is provided as a robustness check.
    """
    import numpy as np
    import pandas as pd

    # Helper to safely get conf int as DataFrame indexed by parameter names
    def conf_int_df(res):
        ci = res.conf_int()
        try:
            ci_df = pd.DataFrame(ci, index=res.params.index, columns=['ci_low', 'ci_high'])
        except Exception:
            ci_arr = np.asarray(ci)
            ci_df = pd.DataFrame(ci_arr, index=res.params.index, columns=['ci_low', 'ci_high'])
        return ci_df

    # Helper to find a parameter name in a model's params index
    def find_param_name(target, params_index):
        # Try exact match first
        if target in params_index:
            return target
        # Case-insensitive exact match
        lower_index = {str(p).lower(): p for p in params_index}
        tlow = str(target).lower()
        if tlow in lower_index:
            return lower_index[tlow]
        # Substring match (case-insensitive) - prefer whole token matches
        matches = [p for p in params_index if tlow in str(p).lower()]
        if len(matches) == 1:
            return matches[0]
        # Try startswith / endswith
        for p in params_index:
            pl = str(p).lower()
            if pl.startswith(tlow) or pl.endswith(tlow):
                return p
        # No reliable match found
        return None

    # Safe formatter for possibly-missing numeric values
    def fmt(value, fmt_spec="{:.4f}"):
        if value is None:
            return "NA"
        try:
            return fmt_spec.format(value)
        except Exception:
            return str(value)

    # Validate input
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict as returned by the modeling function.")
    if 'nb_result' not in model_output or 'ols_result' not in model_output:
        raise KeyError("model_output must contain keys 'nb_result' and 'ols_result'.")

    nb_res = model_output.get('nb_result')
    ols_res = model_output.get('ols_result')

    if nb_res is None:
        raise ValueError("nb_result is None. Negative-binomial/Poisson model failed to fit.")

    results = {}

    # Variables of interest
    vars_of_interest = ['NameFemininity', 'FemaleName']

    # Extract from NB/Poisson
    try:
        nb_params = nb_res.params
        nb_bse = nb_res.bse
        nb_p = nb_res.pvalues
        nb_ci_df = conf_int_df(nb_res)
    except Exception as e:
        raise RuntimeError(f"Failed to extract statistics from nb_result: {e}")

    # Extract from OLS
    try:
        ols_params = ols_res.params
        ols_bse = ols_res.bse
        ols_p = ols_res.pvalues
        ols_ci_df = conf_int_df(ols_res)
    except Exception as e:
        raise RuntimeError(f"Failed to extract statistics from ols_result: {e}")

    for var in vars_of_interest:
        nb_param_name = find_param_name(var, nb_params.index)
        ols_param_name = find_param_name(var, ols_params.index)

        # Initialize placeholders
        nb_coef = nb_se = nb_pval = nb_ci_low = nb_ci_high = nb_irr = nb_irr_ci_low = nb_irr_ci_high = None
        ols_coef = ols_se_val = ols_pval = ols_ci_low = ols_ci_high = ols_mult = ols_pct_change = ols_pct_ci_low = ols_pct_ci_high = None

        # NB values if present
        if nb_param_name is not None:
            try:
                nb_coef = float(nb_params[nb_param_name])
            except Exception:
                nb_coef = None
            try:
                nb_se = float(nb_bse[nb_param_name]) if nb_param_name in nb_bse.index else None
            except Exception:
                nb_se = None
            try:
                nb_pval = float(nb_p[nb_param_name]) if nb_param_name in nb_p.index else None
            except Exception:
                nb_pval = None
            try:
                nb_ci_low = float(nb_ci_df.loc[nb_param_name, 'ci_low'])
                nb_ci_high = float(nb_ci_df.loc[nb_param_name, 'ci_high'])
            except Exception:
                nb_ci_low = nb_ci_high = None
            try:
                nb_irr = float(np.exp(nb_coef)) if nb_coef is not None else None
                nb_irr_ci_low = float(np.exp(nb_ci_low)) if nb_ci_low is not None else None
                nb_irr_ci_high = float(np.exp(nb_ci_high)) if nb_ci_high is not None else None
            except Exception:
                nb_irr = nb_irr_ci_low = nb_irr_ci_high = None

        # OLS values if present
        if ols_param_name is not None:
            try:
                ols_coef = float(ols_params[ols_param_name])
            except Exception:
                ols_coef = None
            try:
                ols_se_val = float(ols_bse[ols_param_name]) if ols_param_name in ols_bse.index else None
            except Exception:
                ols_se_val = None
            try:
                ols_pval = float(ols_p[ols_param_name]) if ols_param_name in ols_p.index else None
            except Exception:
                ols_pval = None
            try:
                ols_ci_low = float(ols_ci_df.loc[ols_param_name, 'ci_low'])
                ols_ci_high = float(ols_ci_df.loc[ols_param_name, 'ci_high'])
            except Exception:
                ols_ci_low = ols_ci_high = None
            try:
                ols_mult = float(np.exp(ols_coef)) if ols_coef is not None else None
                ols_pct_change = (ols_mult - 1.0) * 100.0 if ols_mult is not None else None
                ols_pct_ci_low = (float(np.exp(ols_ci_low)) - 1.0) * 100.0 if ols_ci_low is not None else None
                ols_pct_ci_high = (float(np.exp(ols_ci_high)) - 1.0) * 100.0 if ols_ci_high is not None else None
            except Exception:
                ols_mult = ols_pct_change = ols_pct_ci_low = ols_pct_ci_high = None

        results[var] = {
            'nb': {
                'param_name': str(nb_param_name) if nb_param_name is not None else None,
                'coef': nb_coef,
                'std_err': nb_se,
                'p_value': nb_pval,
                'conf_int': [nb_ci_low, nb_ci_high] if (nb_ci_low is not None and nb_ci_high is not None) else None,
                'IRR': nb_irr,
                'IRR_conf_int': [nb_irr_ci_low, nb_irr_ci_high] if (nb_irr_ci_low is not None and nb_irr_ci_high is not None) else None
            },
            'ols_log': {
                'param_name': str(ols_param_name) if ols_param_name is not None else None,
                'coef': ols_coef,
                'std_err': ols_se_val,
                'p_value': ols_pval,
                'conf_int': [ols_ci_low, ols_ci_high] if (ols_ci_low is not None and ols_ci_high is not None) else None,
                'multiplicative_effect_on_Deaths_plus1': ols_mult,
                'percent_change_on_Deaths_plus1': ols_pct_change,
                'percent_change_conf_int': [ols_pct_ci_low, ols_pct_ci_high] if (ols_pct_ci_low is not None and ols_pct_ci_high is not None) else None
            }
        }

    # Decide whether evidence supports the hypothesis
    # Hypothesis: more feminine names -> fewer precautions -> more fatalities.
    # So we expect a positive association (coef > 0) between NameFemininity and deaths.
    nf_nb = results.get('NameFemininity', {}).get('nb', {})
    nf_ols = results.get('NameFemininity', {}).get('ols_log', {})

    support = False
    support_reason = ""
    nb_pval = nf_nb.get('p_value')
    nb_coef = nf_nb.get('coef')

    # Primary decision uses NB/Poisson model
    if (nb_pval is not None) and (nb_coef is not None) and (nb_pval < 0.05) and (nb_coef > 0):
        support = True
        support_reason = ("The negative-binomial/Poisson model shows a statistically significant "
                          "positive coefficient for NameFemininity (p < 0.05), meaning more "
                          "feminine names are associated with higher fatalities (IRR > 1).")
    else:
        support = False
        if nb_pval is None or nb_coef is None:
            support_reason = "The negative-binomial/Poisson model did not provide a usable coefficient and/or p-value for NameFemininity."
        else:
            support_reason = ("The negative-binomial/Poisson model does not show a statistically significant "
                              "positive association for NameFemininity (either p >= 0.05 or coef <= 0), "
                              "so the primary model does not provide support for the hypothesis.")

    # Short human-readable description with safe formatting
    description_lines = [
        "Primary model: Negative-Binomial (GLM) on raw death counts; robustness: OLS on log(Deaths+1).",
        "",
        "Extracted statistics:",
        f"- NameFemininity (NB): param = {nf_nb.get('param_name', 'NA')}, coef = {fmt(nf_nb.get('coef'), '{:.4f}')}, SE = {fmt(nf_nb.get('std_err'), '{:.4f}')}, p = {fmt(nf_nb.get('p_value'), '{:.4g}')},",
        f"  IRR = {fmt(nf_nb.get('IRR'), '{:.4f}')} (95% CI: [{fmt((nf_nb.get('IRR_conf_int') or [None, None])[0], '{:.4f}')}, {fmt((nf_nb.get('IRR_conf_int') or [None, None])[1], '{:.4f}')}])",
        f"- NameFemininity (OLS log): param = {nf_ols.get('param_name', 'NA')}, coef = {fmt(nf_ols.get('coef'), '{:.4f}')}, SE = {fmt(nf_ols.get('std_err'), '{:.4f}')}, p = {fmt(nf_ols.get('p_value'), '{:.4g}')},",
        f"  implied % change in (Deaths+1) per unit = {fmt(nf_ols.get('percent_change_on_Deaths_plus1'), '{:.2f}')}% (95% CI: [{fmt((nf_ols.get('percent_change_conf_int') or [None, None])[0], '{:.2f}')}%, {fmt((nf_ols.get('percent_change_conf_int') or [None, None])[1], '{:.2f}')}%])",
        "",
        "Binary female name variable (FemaleName):",
        f"- FemaleName (NB): param = {results['FemaleName']['nb'].get('param_name', 'NA')}, coef = {fmt(results['FemaleName']['nb'].get('coef'), '{:.4f}')}, p = {fmt(results['FemaleName']['nb'].get('p_value'), '{:.4g}')}, IRR = {fmt(results['FemaleName']['nb'].get('IRR'), '{:.4f}')}",
        f"- FemaleName (OLS log): param = {results['FemaleName']['ols_log'].get('param_name', 'NA')}, coef = {fmt(results['FemaleName']['ols_log'].get('coef'), '{:.4f}')}, p = {fmt(results['FemaleName']['ols_log'].get('p_value'), '{:.4g}')}, implied % change = {fmt(results['FemaleName']['ols_log'].get('percent_change_on_Deaths_plus1'), '{:.2f}')}%",
        "",
        "Conclusion (primary model-driven):",
        support_reason
    ]
    description = "\n".join(description_lines)

    return {
        "object": results,
        "description": description
    }