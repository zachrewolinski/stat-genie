def extract_final_answer(model_output):
    """
    Extracts key statistics for the masfem_z effect from the provided model_output dict.
    Expects model_output to contain:
      - 'ols': statsmodels RegressionResultsWrapper (OLS on log_deaths)
      - 'neg_binomial': statsmodels GLMResultsWrapper (NegativeBinomial on alldeaths) or None

    Returns:
      {
        "object": {
          "ols_masfem": {coef, se, pvalue, ci_lower, ci_upper, nobs},
          "ols_FemaleName": {...} or None,
          "negbin_masfem": {coef, se, pvalue, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper} or None,
          "negbin_FemaleName": {...} or None,
          "conclusion": "string concise conclusion"
        },
        "description": "Short explanation of what these numbers mean for the hypothesis"
      }
    """
    import numpy as np

    def _find_param_name(params_index, target_substr):
        # exact match preferred, otherwise find first containing substring
        if target_substr in params_index:
            return target_substr
        for p in params_index:
            if target_substr in p:
                return p
        return None

    def _extract_from_result(res, target):
        if res is None:
            return None
        # locate parameter name
        params_index = list(res.params.index)
        pname = _find_param_name(params_index, target)
        if pname is None:
            return None
        try:
            coef = float(res.params[pname])
            se = float(res.bse[pname]) if hasattr(res, 'bse') else None
            pvalue = float(res.pvalues[pname]) if hasattr(res, 'pvalues') else None
            # conf_int can return ndarray or DataFrame
            try:
                ci = res.conf_int().loc[pname]
                ci_lower, ci_upper = float(ci.iloc[0]), float(ci.iloc[1])
            except Exception:
                # fallback to array indexing
                ci_arr = res.conf_int()
                # find row index
                row_idx = params_index.index(pname)
                ci_lower, ci_upper = float(ci_arr[row_idx, 0]), float(ci_arr[row_idx, 1])
            return {
                'param_name': pname,
                'coef': coef,
                'se': se,
                'pvalue': pvalue,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        except Exception:
            return None

    results = {}
    ols_res = model_output.get('ols')
    nb_res = model_output.get('neg_binomial')

    # Extract for masfem_z and FemaleName from OLS
    ols_m = _extract_from_result(ols_res, 'masfem_z')
    ols_f = _extract_from_result(ols_res, 'FemaleName')

    # Extract for masfem_z and FemaleName from NegBinomial
    nb_m = _extract_from_result(nb_res, 'masfem_z') if nb_res is not None else None
    nb_f = _extract_from_result(nb_res, 'FemaleName') if nb_res is not None else None

    # For negbin, also calculate incidence rate ratio (IRR) and its CI by exponentiating coeff and CI
    if nb_m is not None:
        try:
            irr = float(np.exp(nb_m['coef']))
            irr_ci_lower = float(np.exp(nb_m['ci_lower']))
            irr_ci_upper = float(np.exp(nb_m['ci_upper']))
            nb_m.update({'irr': irr, 'irr_ci_lower': irr_ci_lower, 'irr_ci_upper': irr_ci_upper,
                         'irr_pct_change': (irr - 1.0) * 100.0})
        except Exception:
            pass
    if nb_f is not None:
        try:
            irr = float(np.exp(nb_f['coef']))
            irr_ci_lower = float(np.exp(nb_f['ci_lower']))
            irr_ci_upper = float(np.exp(nb_f['ci_upper']))
            nb_f.update({'irr': irr, 'irr_ci_lower': irr_ci_lower, 'irr_ci_upper': irr_ci_upper,
                         'irr_pct_change': (irr - 1.0) * 100.0})
        except Exception:
            pass

    # Add sample size for OLS if possible
    try:
        if ols_res is not None and hasattr(ols_res, 'nobs'):
            if ols_m is not None:
                ols_m['nobs'] = int(ols_res.nobs)
            if ols_f is not None:
                ols_f['nobs'] = int(ols_res.nobs)
    except Exception:
        pass

    results['ols_masfem'] = ols_m
    results['ols_FemaleName'] = ols_f
    results['negbin_masfem'] = nb_m
    results['negbin_FemaleName'] = nb_f

    # Formulate concise conclusion based primarily on OLS (primary model): masfem_z effect
    conclusion = []
    if ols_m is None:
        conclusion.append("OLS result for 'masfem_z' not found; cannot draw primary inference from OLS.")
    else:
        coef = ols_m['coef']
        p = ols_m['pvalue']
        sign = "negative" if coef < 0 else "positive" if coef > 0 else "null"
        if p is None:
            conclusion.append(f"masfem_z coefficient = {coef:.4g} (p unknown); direction = {sign}.")
        else:
            sig = (p < 0.05)
            if sig:
                direction_statement = ("more feminine names are associated with LOWER log-deaths"
                                       if coef < 0 else
                                       "more feminine names are associated with HIGHER log-deaths")
                conclusion.append(f"OLS: masfem_z coef = {coef:.4g}, p = {p:.3g}. Statistically significant (alpha=0.05). "
                                  f"Interpretation: {direction_statement}.")
            else:
                conclusion.append(f"OLS: masfem_z coef = {coef:.4g}, p = {p:.3g}. Not statistically significant at alpha=0.05; "
                                  "evidence is inconclusive for an effect of name femininity on fatalities.")

    # Cross-check with negative binomial if available
    if nb_m is None:
        conclusion.append("Negative binomial model not available or did not return masfem_z; no count-model cross-check.")
    else:
        coef = nb_m['coef']
        p = nb_m['pvalue']
        irr = nb_m.get('irr')
        if p is None:
            conclusion.append(f"NegBin: masfem_z coef = {coef:.4g} (p unknown).")
        else:
            if p < 0.05:
                if coef < 0:
                    dir_stmt = "feminine names associated with LOWER expected death counts"
                else:
                    dir_stmt = "feminine names associated with HIGHER expected death counts"
                if irr is not None:
                    conclusion.append(f"NegBin: masfem_z coef = {coef:.4g}, p = {p:.3g}; IRR = {irr:.3f} "
                                      f"(95% CI [{nb_m['irr_ci_lower']:.3f}, {nb_m['irr_ci_upper']:.3f}]). {dir_stmt}.")
                else:
                    conclusion.append(f"NegBin: masfem_z coef = {coef:.4g}, p = {p:.3g}. {dir_stmt}.")
            else:
                if irr is not None:
                    conclusion.append(f"NegBin: masfem_z coef = {coef:.4g}, p = {p:.3g}; IRR = {irr:.3f}. Not statistically significant.")
                else:
                    conclusion.append(f"NegBin: masfem_z coef = {coef:.4g}, p = {p:.3g}. Not statistically significant.")

    # Final single-sentence verdict focusing on hypothesis:
    # If OLS shows significant negative effect, that supports the hypothesis (more feminine -> fewer fatalities).
    final_verdict = "Inconclusive"
    if ols_m is not None and ols_m.get('pvalue') is not None:
        if ols_m['pvalue'] < 0.05:
            if ols_m['coef'] < 0:
                final_verdict = "Supports hypothesis: more feminine names associated with fewer fatalities (significant)."
            else:
                final_verdict = "Contradicts hypothesis: more feminine names associated with more fatalities (significant)."
        else:
            final_verdict = "No strong evidence to support the hypothesis (OLS non-significant)."

    results['conclusion_brief'] = final_verdict

    description = ("Extracted coefficients, standard errors, p-values, and 95% confidence intervals for the primary "
                   "predictor 'masfem_z' (continuous femininity rating) from the OLS on log-deaths and, where available, "
                   "from the negative-binomial GLM on raw death counts. For the negative-binomial model the exponentiated "
                   "coefficient (IRR) and its CI are also provided. The brief conclusion states whether the OLS result "
                   "provides statistically significant support for the hypothesis that more feminine hurricane names are "
                   "associated with fewer fatalities (i.e., negative coefficient on masfem_z).")

    return {"object": results, "description": description}