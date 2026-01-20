def extract_final_answer(model_output):
    """
    Extract relevant statistics for the primary IV (name femininity) from the model output.
    Returns a dictionary with keys:
      - "object": a dict containing numeric results (OLS and Negative Binomial where available),
                  a boolean "supports_hypothesis" indicating whether the primary OLS result
                  supports the hypothesis (positive coef and p < 0.05), and a short textual
                  "conclusion".
      - "description": a short explanation of the returned numbers and interpretation.
    """
    import math
    import numpy as np

    res = {
        'ols': None,
        'negbin': None,
        'supports_hypothesis': None,
        'conclusion': None
    }

    ols = model_output.get('ols', None)
    negbin = model_output.get('negbin', None)

    # Helper to find parameter name matching masfem (robust to slight name differences)
    def find_param_name(params_index, keywords=('masfem', 'mas_fem', 'masfem_z')):
        if params_index is None:
            return None
        for k in keywords:
            for name in params_index:
                if k in str(name):
                    return name
        # fallback: return first param containing 'fem' or 'female'
        for name in params_index:
            if ('fem' in str(name)) or ('female' in str(name)):
                return name
        return None

    # Extract from OLS
    try:
        if ols is not None:
            params_index = getattr(ols, 'params').index
            iv_name = find_param_name(params_index)
            if iv_name is None:
                raise KeyError("Could not locate masfem-like parameter in OLS results.")
            coef = float(ols.params[iv_name])
            se = float(ols.bse[iv_name]) if hasattr(ols, 'bse') else None
            tval = float(ols.tvalues[iv_name]) if hasattr(ols, 'tvalues') else None
            pval = float(ols.pvalues[iv_name]) if hasattr(ols, 'pvalues') else None
            ci = ols.conf_int().loc[iv_name].values.tolist() if hasattr(ols, 'conf_int') else [None, None]
            ci = [float(ci[0]), float(ci[1])]

            # Interpretation for log(alldeaths + 1) outcome:
            # Approx percent change ~= 100 * coef
            approx_pct = 100.0 * coef
            # Exact multiplicative change in (alldeaths + 1): exp(coef)
            mult_change = math.exp(coef)
            exact_pct = 100.0 * (mult_change - 1.0)

            res['ols'] = {
                'param_name': str(iv_name),
                'coef': coef,
                'se': se,
                't': tval,
                'p_value': pval,
                '95ci': ci,
                'approx_pct_change': approx_pct,   # percent change approximation per 1 SD increase
                'exact_multiplicative_change': mult_change,  # factor change in (alldeaths+1)
                'exact_pct_change': exact_pct
            }
    except Exception as e:
        res['ols'] = {'error': f'Failed to extract OLS stats: {e}'}

    # Extract from Negative Binomial (if available)
    try:
        if negbin is not None:
            params_index_nb = getattr(negbin, 'params').index
            iv_name_nb = find_param_name(params_index_nb)
            if iv_name_nb is None:
                raise KeyError("Could not locate masfem-like parameter in Negative Binomial results.")
            coef_nb = float(negbin.params[iv_name_nb])
            se_nb = float(negbin.bse[iv_name_nb]) if hasattr(negbin, 'bse') else None
            pval_nb = float(negbin.pvalues[iv_name_nb]) if hasattr(negbin, 'pvalues') else None
            ci_nb = negbin.conf_int().loc[iv_name_nb].values.tolist() if hasattr(negbin, 'conf_int') else [None, None]
            ci_nb = [float(ci_nb[0]), float(ci_nb[1])]

            # For count model: exponentiated coef = incidence rate ratio (IRR)
            irr = math.exp(coef_nb)
            irr_ci = [math.exp(ci_nb[0]), math.exp(ci_nb[1])]

            res['negbin'] = {
                'param_name': str(iv_name_nb),
                'coef': coef_nb,
                'se': se_nb,
                'p_value': pval_nb,
                '95ci': ci_nb,
                'irr': irr,
                'irr_95ci': irr_ci
            }
    except Exception as e:
        res['negbin'] = {'error': f'Failed to extract Negative Binomial stats: {e}'}

    # Decide whether OLS result supports the hypothesis:
    # Hypothesis: more feminine names -> fewer precautions -> more deaths.
    # Therefore we expect a positive coef on name femininity predicting log_alldeaths.
    supports = None
    conclusion = ""
    try:
        if isinstance(res['ols'], dict) and ('coef' in res['ols']) and (res['ols'].get('p_value') is not None):
            coef = res['ols']['coef']
            p = res['ols']['p_value']
            supports = (coef > 0) and (p < 0.05)
            if supports:
                conclusion = ("The primary OLS estimate shows a positive association between name femininity "
                              "and log(deaths+1) (coef = {coef:.4f}, p = {p:.3g}), which is consistent with the "
                              "hypothesis that more feminine hurricane names lead to more fatalities (i.e., fewer "
                              "precautionary measures).").format(coef=coef, p=p)
            else:
                conclusion = ("The primary OLS estimate does not provide statistically significant evidence "
                              "that more feminine hurricane names lead to higher fatalities. "
                              "Estimate: coef = {coef:.4f}, p = {p:.3g}.").format(coef=coef, p=p)
        else:
            conclusion = "Could not evaluate support for the hypothesis because OLS statistics were not available."
    except Exception as e:
        conclusion = f"Error forming conclusion: {e}"
        supports = None

    res['supports_hypothesis'] = bool(supports) if supports is not None else None
    res['conclusion'] = conclusion

    # Final description summarizing what is returned
    description = (
        "Returns key statistics for the primary independent variable (name femininity) "
        "from the OLS on log(alldeaths+1) and, if available, the Negative Binomial count model. "
        "For OLS the function provides coefficient, standard error, t-stat, p-value, 95% CI, "
        "an approximate percent change (100 * coef) interpretation, and the exact multiplicative "
        "change exp(coef). For the negative binomial it provides coef, p-value, 95% CI and the "
        "incidence rate ratio (IRR = exp(coef)) with CI. The field 'supports_hypothesis' is True "
        "when the OLS coef is positive and p < 0.05 (evidence consistent with the hypothesis)."
    )

    return {
        "object": res,
        "description": description
    }