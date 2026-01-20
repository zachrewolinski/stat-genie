def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of 'masfem_z' from the provided model_output.
    Expects model_output to be a dict with keys:
      - 'neg_binom_alldeaths' : statsmodels GLMResultsWrapper (NegativeBinomial)
      - 'ols_log_ndam15'      : statsmodels RegressionResultsWrapper (OLS)
    
    Returns:
      {
        "object": { ... extracted numeric statistics ... },
        "description": "plain-language interpretation of those statistics"
      }
    """
    import numpy as np

    def _get_param_stats(res, varname):
        """
        Safely extract coefficient, se, pvalue, and conf_int for varname from a statsmodels result object.
        Returns a dict with keys: coef, se, pvalue, ci_lower, ci_upper
        """
        out = dict.fromkeys(['coef', 'se', 'pvalue', 'ci_lower', 'ci_upper'])
        try:
            params = res.params
            # get coefficient
            out['coef'] = float(params[varname])
        except Exception:
            raise KeyError(f"Variable '{varname}' not found in model params.")

        # standard error (bse) if present
        try:
            out['se'] = float(res.bse[varname])
        except Exception:
            # fallback: estimate se from cov_params if available
            try:
                cov = res.cov_params()
                idx = list(params.index).index(varname)
                out['se'] = float(np.sqrt(np.abs(cov.iloc[idx, idx])))
            except Exception:
                out['se'] = None

        # p-value if available
        try:
            out['pvalue'] = float(res.pvalues[varname])
        except Exception:
            out['pvalue'] = None

        # confidence interval
        try:
            ci = res.conf_int()
            # conf_int may be a DataFrame or ndarray
            if hasattr(ci, "loc"):
                ci_row = ci.loc[varname]
                out['ci_lower'] = float(ci_row.iloc[0])
                out['ci_upper'] = float(ci_row.iloc[1])
            else:
                # assume ndarray with same ordering as params
                idx = list(params.index).index(varname)
                out['ci_lower'] = float(ci[idx, 0])
                out['ci_upper'] = float(ci[idx, 1])
        except Exception:
            out['ci_lower'] = None
            out['ci_upper'] = None

        return out

    # Prepare result container
    results = {}

    # Extract from negative binomial model (alldeaths)
    if 'neg_binom_alldeaths' in model_output:
        nb = model_output['neg_binom_alldeaths']
        try:
            nb_stats = _get_param_stats(nb, 'masfem_z')
            # Interpret coefficient: for NB GLM (log link) coef is log incidence rate ratio
            coef = nb_stats['coef']
            nb_stats['IRR'] = float(np.exp(coef))
            if nb_stats['ci_lower'] is not None and nb_stats['ci_upper'] is not None:
                nb_stats['IRR_ci_lower'] = float(np.exp(nb_stats['ci_lower']))
                nb_stats['IRR_ci_upper'] = float(np.exp(nb_stats['ci_upper']))
            else:
                nb_stats['IRR_ci_lower'] = None
                nb_stats['IRR_ci_upper'] = None
        except KeyError as e:
            nb_stats = {"error": str(e)}
        results['neg_binom_alldeaths'] = nb_stats
    else:
        results['neg_binom_alldeaths'] = {"error": "neg_binom_alldeaths not found in model_output"}

    # Extract from OLS model (log_ndam15)
    if 'ols_log_ndam15' in model_output:
        ols = model_output['ols_log_ndam15']
        try:
            ols_stats = _get_param_stats(ols, 'masfem_z')
        except KeyError as e:
            ols_stats = {"error": str(e)}
        results['ols_log_ndam15'] = ols_stats
    else:
        results['ols_log_ndam15'] = {"error": "ols_log_ndam15 not found in model_output"}

    # Build plain-language description interpreting the NB result primarily (death counts)
    desc_lines = []
    # Interpret NB results if available numeric
    nb = results.get('neg_binom_alldeaths', {})
    if 'error' in nb:
        desc_lines.append("Could not extract statistics from the negative binomial model: " + nb.get('error'))
    else:
        coef = nb.get('coef')
        p = nb.get('pvalue')
        irr = nb.get('IRR')
        ci_low = nb.get('ci_lower')
        ci_high = nb.get('ci_upper')
        irr_ci_low = nb.get('IRR_ci_lower')
        irr_ci_high = nb.get('IRR_ci_upper')

        desc_lines.append("Negative binomial model (outcome = total deaths):")
        desc_lines.append(f"- Coefficient for masfem_z (standardized femininity index): {coef:.4f}")
        if p is not None:
            desc_lines.append(f"- p-value: {p:.4g}")
        if irr is not None:
            desc_lines.append(f"- Incidence Rate Ratio (IRR) = exp(coef): {irr:.4f}")
        if (ci_low is not None) and (ci_high is not None):
            desc_lines.append(f"- 95% CI for coef: [{ci_low:.4f}, {ci_high:.4f}]")
        if (irr_ci_low is not None) and (irr_ci_high is not None):
            desc_lines.append(f"- 95% CI for IRR: [{irr_ci_low:.4f}, {irr_ci_high:.4f}]")

        # Assessment vs hypothesis
        if p is None:
            desc_lines.append("Cannot assess statistical significance (p-value missing).")
        else:
            if p < 0.05:
                # direction
                if coef > 0:
                    desc_lines.append("Interpretation: The effect is positive and statistically significant at p<0.05 — more feminine names are associated with higher death counts (supports the hypothesis that feminine names lead to fewer precautions).")
                else:
                    desc_lines.append("Interpretation: The effect is negative and statistically significant at p<0.05 — more feminine names are associated with lower death counts (contradicts the hypothesis).")
            else:
                desc_lines.append("Interpretation: The effect is not statistically significant at conventional levels (p>=0.05); results do not provide strong evidence for an association between name femininity and death counts.")

    # Add OLS (damage) summary as robustness check
    ols = results.get('ols_log_ndam15', {})
    if 'error' in ols:
        desc_lines.append("Could not extract statistics from the OLS damage model: " + ols.get('error'))
    else:
        coef_o = ols.get('coef')
        p_o = ols.get('pvalue')
        ci_low_o = ols.get('ci_lower')
        ci_high_o = ols.get('ci_upper')
        desc_lines.append("")
        desc_lines.append("OLS model (outcome = log damage) — robustness check:")
        desc_lines.append(f"- Coefficient for masfem_z: {coef_o:.4f}" if coef_o is not None else "- Coefficient: missing")
        if p_o is not None:
            desc_lines.append(f"- p-value: {p_o:.4g}")
        if (ci_low_o is not None) and (ci_high_o is not None):
            desc_lines.append(f"- 95% CI for coef: [{ci_low_o:.4f}, {ci_high_o:.4f}]")
        if p_o is None:
            desc_lines.append("Cannot assess statistical significance for OLS (p-value missing).")
        else:
            if p_o < 0.05:
                if coef_o > 0:
                    desc_lines.append("Interpretation: In the OLS robustness check, the association is positive and statistically significant (supports hypothesis).")
                else:
                    desc_lines.append("Interpretation: In the OLS robustness check, the association is negative and statistically significant (contradicts hypothesis).")
            else:
                desc_lines.append("Interpretation: In the OLS robustness check, the association is not statistically significant (no strong evidence).")

    description = "\n".join(desc_lines)

    return {
        "object": results,
        "description": description
    }