def extract_final_answer(model_output):
    """
    Extracts statistics for the effect of 'masfem_std' from the provided model_output.

    Returns a dictionary with:
      - "object": dict of numeric results (coefficients, SEs, p-values, 95% CIs,
                  transformed % interpretation, sample size)
      - "description": short plain-language interpretation of the results relative
                       to the hypothesis that more feminine names are associated
                       with higher fatalities.

    Expects model_output to contain keys 'ols' and optionally 'neg_binomial_glm',
    as produced by the modeling function in the task.
    """
    import numpy as np

    out = {}
    desc_parts = []

    # Basic checks
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict; expected the dict returned by the modeling function."
        }

    n_obs = model_output.get('n_obs', None)

    # Extract OLS results (primary model)
    ols = model_output.get('ols', None)
    if ols is None:
        ols_obj = None
        desc_parts.append("No OLS model found in model_output.")
    else:
        try:
            beta_ols = float(ols.params['masfem_std'])
            se_ols = float(ols.bse['masfem_std'])
            t_ols = float(ols.tvalues['masfem_std']) if hasattr(ols, 'tvalues') else None
            p_ols = float(ols.pvalues['masfem_std'])
            ci_ols = ols.conf_int().loc['masfem_std'].values.astype(float)
            ci_lower_ols, ci_upper_ols = float(ci_ols[0]), float(ci_ols[1])
            # Interpret for log(1 + alldeaths): approximate percent change in (1 + deaths)
            pct_change_ols = (np.exp(beta_ols) - 1.0) * 100.0
            pct_change_ci_lower = (np.exp(ci_lower_ols) - 1.0) * 100.0
            pct_change_ci_upper = (np.exp(ci_upper_ols) - 1.0) * 100.0

            ols_obj = {
                'coef': beta_ols,
                'se': se_ols,
                't': t_ols,
                'p_value': p_ols,
                'ci_95_lower': ci_lower_ols,
                'ci_95_upper': ci_upper_ols,
                'approx_pct_change_per_1sd_masfem': pct_change_ols,
                'pct_change_ci_95_lower': pct_change_ci_lower,
                'pct_change_ci_95_upper': pct_change_ci_upper,
                'n_obs': n_obs
            }

            # Interpretation sentence for OLS
            sig_text = "statistically significant" if (p_ols < 0.05) else "not statistically significant"
            desc_parts.append(
                f"OLS: masfem_std coefficient = {beta_ols:.4f} (SE = {se_ols:.4f}), 95% CI [{ci_lower_ols:.4f}, {ci_upper_ols:.4f}], "
                f"p = {p_ols:.3g}. This is {sig_text}. "
                f"On the log(1+deaths) scale this corresponds to an approximate {pct_change_ols:.1f}% "
                f"change in (1 + fatalities) per 1 SD increase in name femininity "
                f"(95% CI: {pct_change_ci_lower:.1f}% to {pct_change_ci_upper:.1f}%)."
            )

        except Exception as e:
            ols_obj = None
            desc_parts.append(f"Failed to extract OLS stats: {e}")

    out['ols'] = ols_obj

    # Extract Negative Binomial GLM (robustness)
    nb = model_output.get('neg_binomial_glm', None)
    if nb is None:
        nb_obj = None
        desc_parts.append("No negative binomial GLM found or it failed to fit.")
    else:
        try:
            beta_nb = float(nb.params['masfem_std'])
            se_nb = float(nb.bse['masfem_std'])
            # statsmodels GLM has z-values in .tvalues or .params/.bse -> compute z
            z_nb = beta_nb / se_nb if se_nb != 0 else None
            p_nb = float(nb.pvalues['masfem_std'])
            ci_nb = nb.conf_int().loc['masfem_std'].values.astype(float)
            ci_lower_nb, ci_upper_nb = float(ci_nb[0]), float(ci_nb[1])
            # For count model, report incidence rate ratio (IRR = exp(beta))
            irr_nb = np.exp(beta_nb)
            irr_ci_lower = np.exp(ci_lower_nb)
            irr_ci_upper = np.exp(ci_upper_nb)

            nb_obj = {
                'coef': beta_nb,
                'se': se_nb,
                'z': z_nb,
                'p_value': p_nb,
                'ci_95_lower': ci_lower_nb,
                'ci_95_upper': ci_upper_nb,
                'irr': irr_nb,
                'irr_ci_95_lower': irr_ci_lower,
                'irr_ci_95_upper': irr_ci_upper,
                'n_obs': n_obs
            }

            sig_text_nb = "statistically significant" if (p_nb < 0.05) else "not statistically significant"
            desc_parts.append(
                f"Negative binomial GLM: masfem_std coefficient = {beta_nb:.4f} (SE = {se_nb:.4f}), 95% CI [{ci_lower_nb:.4f}, {ci_upper_nb:.4f}], "
                f"p = {p_nb:.3g}. This is {sig_text_nb}. "
                f"IRR = {irr_nb:.3f} (95% CI: {irr_ci_lower:.3f} to {irr_ci_upper:.3f}), meaning a 1 SD increase in name femininity is associated with a multiplicative change of {irr_nb:.3f} in expected fatalities."
            )

        except Exception as e:
            nb_obj = None
            desc_parts.append(f"Failed to extract negative binomial stats: {e}")

    out['neg_binomial_glm'] = nb_obj

    # Final concise conclusion relative to hypothesis
    # Hypothesis: higher (more feminine) names => more fatalities (positive effect).
    conclusion = "Based on the extracted coefficients: "
    if out['ols'] is not None:
        if out['ols']['p_value'] < 0.05:
            if out['ols']['coef'] > 0:
                conclusion += "OLS provides evidence consistent with the hypothesis (positive and statistically significant effect)."
            else:
                conclusion += "OLS shows a statistically significant effect but in the opposite direction (negative coefficient)."
        else:
            conclusion += "OLS does not provide statistically significant evidence for the hypothesis (effect not significant)."
    else:
        conclusion += "OLS results unavailable."

    # If NB available, compare
    if out['neg_binomial_glm'] is not None:
        if out['neg_binomial_glm']['p_value'] < 0.05:
            if out['neg_binomial_glm']['coef'] > 0:
                conclusion += " Negative-binomial robustness check is consistent (positive and significant)."
            else:
                conclusion += " Negative-binomial check indicates a significant effect but in the opposite direction."
        else:
            conclusion += " Negative-binomial check does not find a statistically significant effect."

    # Assemble final return
    return {
        "object": {
            "ols": out['ols'],
            "neg_binomial_glm": out['neg_binomial_glm'],
            "n_obs": n_obs,
            "formula_ols": model_output.get('formula_ols'),
            "formula_nb": model_output.get('formula_nb')
        },
        "description": " ".join(desc_parts) + " " + conclusion
    }