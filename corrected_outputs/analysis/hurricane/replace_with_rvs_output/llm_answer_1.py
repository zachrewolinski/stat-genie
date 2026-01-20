def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of 'masfem_z' (standardized femininity rating)
    on hurricane fatalities from the provided model output dictionary.

    Expects model_output to be a dict with keys:
      - 'nb_model' : statsmodels GLMResultsWrapper (NegativeBinomial or Poisson fallback)
      - 'nb_model_gender_binary_iv' : statsmodels GLMResultsWrapper (binary gender robustness)
      - 'ols_log_damage_robustness' : statsmodels RegressionResultsWrapper (OLS on log damages)

    Returns a dictionary with:
      - "object": a dict containing extracted numeric statistics for each model (coef, se, pvalue,
                  95% CI, IRR and IRR CI for count models, etc.) and an overall boolean `supports_hypothesis`
                  based on the direction and significance of the NB model.
      - "description": a short plain-language interpretation of what the extracted statistics mean
                       with respect to the hypothesis:
                       "More feminine names -> fewer precautions -> more fatalities"
    """
    import numpy as np

    results = {}
    description_lines = []

    # Helper to safely extract stats from a statsmodels result object for a given variable name
    def _extract_from_sm_result(res, varname):
        if res is None:
            return None
        try:
            params = res.params
            pvalues = res.pvalues
            bse = res.bse
            ci = res.conf_int()
        except Exception:
            # If any of these attributes are missing/unavailable
            return None

        if varname not in params.index:
            # Try common alternative name variants
            alt_names = [varname, varname.strip(), varname.replace('-', '_')]
            found = None
            for n in alt_names:
                if n in params.index:
                    found = n
                    break
            if found is None:
                # Give up
                return None
            varname = found

        coef = float(params[varname])
        se = float(bse[varname]) if varname in bse.index else None
        pval = float(pvalues[varname]) if varname in pvalues.index else None
        try:
            ci_lower, ci_upper = float(ci.loc[varname, 0]), float(ci.loc[varname, 1])
        except Exception:
            # ci could be ndarray with same ordering as params
            try:
                idx = list(params.index).index(varname)
                ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
            except Exception:
                ci_lower, ci_upper = None, None

        return {
            'coef': coef,
            'se': se,
            'pvalue': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    # 1) Primary NB model (counts of deaths)
    nb_res = model_output.get('nb_model')
    nb_stats = _extract_from_sm_result(nb_res, 'masfem_z')
    if nb_stats is not None:
        # For count GLMs with log link, exponentiate coef to get Incidence Rate Ratio (IRR)
        try:
            irr = float(np.exp(nb_stats['coef']))
            irr_ci_lower = float(np.exp(nb_stats['ci_lower'])) if nb_stats['ci_lower'] is not None else None
            irr_ci_upper = float(np.exp(nb_stats['ci_upper'])) if nb_stats['ci_upper'] is not None else None
        except Exception:
            irr, irr_ci_lower, irr_ci_upper = None, None, None

        nb_stats.update({
            'irr': irr,
            'irr_ci_lower': irr_ci_lower,
            'irr_ci_upper': irr_ci_upper
        })

        results['nb_model_masfem'] = nb_stats

        # Interpret for hypothesis:
        # Hypothesis: more feminine names -> fewer precautions -> more fatalities
        # => expected coef on masfem_z should be positive (and significant).
        coef = nb_stats['coef']
        pval = nb_stats['pvalue']
        supports = None
        if pval is not None:
            if (coef > 0) and (pval < 0.05):
                supports = True
                description_lines.append(
                    "Primary NB model: masfem_z has a positive and statistically significant association with fatalities "
                    "(coef = {coef:.4f}, p = {pval:.3g}). This supports the hypothesis: more feminine names are associated "
                    "with MORE deaths (consistent with fewer precautions). IRR = {irr:.3f} (95% CI [{l:.3f}, {u:.3f}])."
                    .format(coef=coef, pval=pval, irr=nb_stats['irr'] if nb_stats.get('irr') is not None else float('nan'),
                            l=nb_stats.get('irr_ci_lower', float('nan')),
                            u=nb_stats.get('irr_ci_upper', float('nan')))
                )
            elif (coef < 0) and (pval < 0.05):
                supports = False
                description_lines.append(
                    "Primary NB model: masfem_z has a negative and statistically significant association with fatalities "
                    "(coef = {coef:.4f}, p = {pval:.3g}). This contradicts the hypothesis: more feminine names are associated "
                    "with FEWER deaths. IRR = {irr:.3f} (95% CI [{l:.3f}, {u:.3f}])."
                    .format(coef=coef, pval=pval, irr=nb_stats['irr'] if nb_stats.get('irr') is not None else float('nan'),
                            l=nb_stats.get('irr_ci_lower', float('nan')),
                            u=nb_stats.get('irr_ci_upper', float('nan')))
                )
            else:
                supports = None
                description_lines.append(
                    "Primary NB model: masfem_z has a coefficient of {coef:.4f} but it is NOT statistically significant (p = {pval:.3g}). "
                    "This is inconclusive with respect to the hypothesis. IRR = {irr:.3f} (95% CI [{l:.3f}, {u:.3f}])."
                    .format(coef=coef, pval=pval,
                            irr=nb_stats['irr'] if nb_stats.get('irr') is not None else float('nan'),
                            l=nb_stats.get('irr_ci_lower', float('nan')),
                            u=nb_stats.get('irr_ci_upper', float('nan')))
                )
        else:
            description_lines.append("Primary NB model: Unable to retrieve p-value; extracted coefficient = {:.4f}.".format(coef))

        results['nb_model_supports_hypothesis'] = supports
    else:
        description_lines.append("Primary NB model: masfem_z statistic could not be extracted.")
        results['nb_model_masfem'] = None
        results['nb_model_supports_hypothesis'] = None

    # 2) Robustness: binary gender IV model
    bin_res = model_output.get('nb_model_gender_binary_iv')
    bin_stats = _extract_from_sm_result(bin_res, 'masfem_z')
    # Note: the code replaced masfem_z with gender_mf numeric values in that model; name still 'masfem_z' in exog.
    if bin_stats is not None:
        try:
            irr = float(np.exp(bin_stats['coef']))
            irr_ci_lower = float(np.exp(bin_stats['ci_lower'])) if bin_stats['ci_lower'] is not None else None
            irr_ci_upper = float(np.exp(bin_stats['ci_upper'])) if bin_stats['ci_upper'] is not None else None
        except Exception:
            irr, irr_ci_lower, irr_ci_upper = None, None, None
        bin_stats.update({'irr': irr, 'irr_ci_lower': irr_ci_lower, 'irr_ci_upper': irr_ci_upper})
        results['nb_model_binary_masfem'] = bin_stats

        # Interpret briefly
        coef = bin_stats['coef']
        pval = bin_stats['pvalue']
        if pval is not None and pval < 0.05:
            direction = "positive" if coef > 0 else "negative"
            description_lines.append(
                "Robustness (binary gender): masfem_z (binary-coded) shows a {dir} significant association "
                "(coef = {coef:.4f}, p = {pval:.3g}), IRR = {irr:.3f} (95% CI [{l:.3f}, {u:.3f}])."
                .format(dir=direction, coef=coef, pval=pval,
                        irr=bin_stats.get('irr', float('nan')),
                        l=bin_stats.get('irr_ci_lower', float('nan')),
                        u=bin_stats.get('irr_ci_upper', float('nan')))
            )
        else:
            description_lines.append(
                "Robustness (binary gender): masfem_z (binary-coded) coef = {coef:.4f} (p = {pval}). Not conclusive."
                .format(coef=coef, pval=(pval if pval is not None else 'NA'))
            )
    else:
        description_lines.append("Robustness (binary gender): could not extract masfem_z stats from the binary model.")
        results['nb_model_binary_masfem'] = None

    # 3) Robustness: OLS on log damages
    ols_res = model_output.get('ols_log_damage_robustness')
    ols_stats = _extract_from_sm_result(ols_res, 'masfem_z')
    if ols_stats is not None:
        # Interpret OLS coefficient on log damage as approximate percent change: 100*(exp(coef)-1)
        pct_change = None
        pct_ci_lower = pct_ci_upper = None
        try:
            pct_change = 100.0 * (np.exp(ols_stats['coef']) - 1.0)
            pct_ci_lower = 100.0 * (np.exp(ols_stats['ci_lower']) - 1.0) if ols_stats['ci_lower'] is not None else None
            pct_ci_upper = 100.0 * (np.exp(ols_stats['ci_upper']) - 1.0) if ols_stats['ci_upper'] is not None else None
        except Exception:
            pct_change, pct_ci_lower, pct_ci_upper = None, None, None

        ols_stats.update({
            'approx_pct_change_in_damage_per_1sd_masfem': pct_change,
            'pct_ci_lower': pct_ci_lower,
            'pct_ci_upper': pct_ci_upper
        })
        results['ols_log_damage_masfem'] = ols_stats

        # Short interpretation
        coef = ols_stats['coef']
        pval = ols_stats['pvalue']
        if pval is not None and pval < 0.05:
            description_lines.append(
                "Robustness (log damages): masfem_z coefficient = {coef:.4f} (p = {pval:.3g}). "
                "This implies an approximate {pct:.2f}% change in economic damages per 1 SD increase in masfem."
                .format(coef=coef, pval=pval, pct=(ols_stats['approx_pct_change_in_damage_per_1sd_masfem'] or 0.0))
            )
        else:
            description_lines.append(
                "Robustness (log damages): masfem_z coef = {coef:.4f} (p = {pval}). Not conclusive."
                .format(coef=coef, pval=(pval if pval is not None else 'NA'))
            )
    else:
        description_lines.append("Robustness (log damages): could not extract masfem_z stats from OLS model.")
        results['ols_log_damage_masfem'] = None

    # Build final short conclusion about whether the models support the hypothesis
    supports_primary = results.get('nb_model_supports_hypothesis')
    if supports_primary is True:
        final_conclusion = (
            "Primary evidence: The main Negative Binomial model shows a statistically significant POSITIVE "
            "association between name 'femininity' (masfem_z) and fatalities. This supports the stated hypothesis "
            "(more feminine names -> fewer precautions -> MORE fatalities)."
        )
    elif supports_primary is False:
        final_conclusion = (
            "Primary evidence: The main Negative Binomial model shows a statistically significant NEGATIVE "
            "association between name 'femininity' and fatalities. This contradicts the stated hypothesis."
        )
    else:
        final_conclusion = (
            "Primary evidence: The main Negative Binomial model does NOT provide statistically significant support "
            "for the hypothesized effect of name femininity on fatalities (inconclusive)."
        )

    description_lines.insert(0, final_conclusion)
    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }