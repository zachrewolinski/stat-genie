def extract_final_answer(model_output):
    """
    Extract statistics about the genus effect (Homo sapiens vs Pan, Pongo, Papio)
    from the model_output produced by the provided modeling function.

    Returns a dictionary with keys:
      - "object": dict containing per-genus statistics (coef, se_clustered, pvalue,
                  OR, OR_95ci_lower, OR_95ci_upper, significant) and summary flags
      - "description": short human-readable interpretation of those statistics
    """
    import numpy as np

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Get available pieces
    glm_clust = model_output.get('glm_result_clustered')
    odds_table = model_output.get('odds_ratio_table')
    genus_coefs = model_output.get('genus_coefficients')

    if glm_clust is None or odds_table is None or genus_coefs is None:
        raise ValueError("model_output missing one of required keys: 'glm_result_clustered', "
                         "'odds_ratio_table', or 'genus_coefficients'")

    # Prepare parameter info
    params = glm_clust.params  # Series
    # Clustered covariance matrix and clustered p-values are available from the wrapper
    try:
        cov = glm_clust.cov_params()
    except Exception:
        # fallback: try attribute
        cov = getattr(glm_clust, '_clustered_cov', None)
        if cov is None:
            raise RuntimeError("Could not obtain clustered covariance matrix from glm_result_clustered.")

    try:
        pvalues = glm_clust.pvalues
    except Exception:
        # fallback: try to get pvalues from odds_table if present
        if 'pvalue' in odds_table.columns:
            pvalues = odds_table['pvalue']
        else:
            raise RuntimeError("Could not obtain clustered p-values from glm_result_clustered or odds_ratio_table.")

    # Compute clustered SEs from covariance if needed
    se_series = None
    try:
        se_vals = np.sqrt(np.diag(cov))
        se_series = {name: se for name, se in zip(params.index, se_vals)}
    except Exception:
        # Try to use se from odds_ratio_table if present
        if 'se_clustered' in odds_table.columns:
            se_series = odds_table['se_clustered'].to_dict()
        else:
            se_series = {}

    # For each genus coefficient, extract stats
    results = {}
    significant_all = True
    any_significant = False
    # genus_coefs keys are parameter names; iterate them
    for param_name, coef in genus_coefs.items():
        # Basic friendly genus name extraction: look for T.<Genus>] or last token after 'T.'
        genus = None
        if 'T.' in param_name:
            # e.g., C(genus, Treatment(reference="Homo sapiens"))[T.Pan]
            try:
                genus = param_name.split('T.')[1].split(']')[0]
            except Exception:
                genus = param_name
        else:
            genus = param_name

        # Get clustered SE
        se = None
        if param_name in se_series:
            se = float(se_series[param_name])
        else:
            # fallback: try to locate in odds_table
            if param_name in odds_table.index and 'se_clustered' in odds_table.columns:
                se = float(odds_table.loc[param_name, 'se_clustered'])
        # Get p-value
        pval = None
        if hasattr(pvalues, 'get') or isinstance(pvalues, dict):
            pval = float(pvalues.get(param_name, np.nan))
        else:
            try:
                pval = float(pvalues.loc[param_name])
            except Exception:
                # fallback: from odds_table
                if param_name in odds_table.index and 'pvalue' in odds_table.columns:
                    pval = float(odds_table.loc[param_name, 'pvalue'])
                else:
                    pval = float('nan')

        # Get OR and CI from odds_table if present
        OR = None; OR_lo = None; OR_hi = None
        if param_name in odds_table.index:
            OR = float(odds_table.loc[param_name, 'OR'])
            OR_lo = float(odds_table.loc[param_name, 'OR_95ci_lower'])
            OR_hi = float(odds_table.loc[param_name, 'OR_95ci_upper'])
        else:
            # compute from coef and se if available
            OR = float(np.exp(coef))
            if se is not None:
                OR_lo = float(np.exp(coef - 1.96 * se))
                OR_hi = float(np.exp(coef + 1.96 * se))

        significant = False
        if (not np.isnan(pval)) and (pval < 0.05):
            significant = True
            any_significant = True
        else:
            significant = False
            significant_all = False

        results[genus] = {
            'param_name': param_name,
            'coef_log_odds': float(coef),
            'se_clustered': (float(se) if se is not None else None),
            'pvalue_clustered': (float(pval) if not np.isnan(pval) else None),
            'odds_ratio': OR,
            'OR_95ci_lower': OR_lo,
            'OR_95ci_upper': OR_hi,
            'significant_vs_Homo_sapiens': significant,
            'interpretation': (
                "Lower odds than Homo sapiens" if coef < 0 else
                ("Higher odds than Homo sapiens" if coef > 0 else "No difference on the log-odds scale")
            )
        }

    # Overall conclusion: because model used Homo sapiens as reference, negative coefficients for non-human genera
    # indicate lower AMTL odds relative to Homo sapiens. We consider "higher AMTL in Homo sapiens" supported when
    # the non-human genera coefficients are negative and statistically significant.
    overall = {
        'homo_higher_than_all_non_human': significant_all,
        'homo_higher_than_any_non_human': any_significant,
        'num_nonhuman_genera_compared': len(results),
        'model_formula': model_output.get('formula')
    }

    # Compose a short description
    # Summarize which genera showed significant differences and direction
    sig_genus_list = [g for g, v in results.items() if v['significant_vs_Homo_sapiens']]
    nonsig_genus_list = [g for g, v in results.items() if not v['significant_vs_Homo_sapiens']]

    if len(results) == 0:
        description = "No genus coefficients were found in the model output."
    else:
        description_lines = []
        description_lines.append(
            "Interpretation (GLM binomial, logit link; clustered SEs by specimen): "
            "genus coefficients are differences in log-odds vs. Homo sapiens (reference). "
            "Negative coefficients -> lower odds of AMTL relative to Homo sapiens."
        )
        if sig_genus_list:
            description_lines.append(
                "Significant (p < 0.05) lower AMTL odds vs. Homo sapiens observed for: " +
                ", ".join(sig_genus_list) + "."
            )
        if nonsig_genus_list:
            description_lines.append(
                "Non-significant differences observed for: " + ", ".join(nonsig_genus_list) + "."
            )
        if overall['homo_higher_than_all_non_human']:
            description_lines.append(
                "Conclusion: After controlling for age, prob_male (sex proxy), and tooth class, "
                "modern humans (Homo sapiens) have statistically higher frequencies (odds) of "
                "antemortem tooth loss than all the compared non-human genera."
            )
        elif overall['homo_higher_than_any_non_human']:
            description_lines.append(
                "Conclusion: Homo sapiens show higher AMTL odds than some (but not all) compared non-human genera."
            )
        else:
            description_lines.append(
                "Conclusion: There is no evidence that Homo sapiens have higher AMTL odds than the compared non-human genera."
            )

        description = " ".join(description_lines)

    return {
        "object": {
            "per_genus_results": results,
            "overall_conclusion": overall
        },
        "description": description
    }