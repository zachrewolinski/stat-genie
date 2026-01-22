def extract_final_answer(model_output):
    """
    Extracts age-related coefficients, p-values, confidence intervals, and odds ratios
    from the fitted models returned in model_output.

    Inputs:
      model_output: dict with keys:
        - 'social_model': fitted statsmodels Logit/GLM results object (required)
        - 'majority_model': fitted statsmodels Logit/GLM results object or None
        - 'n_total': int (optional)
        - 'n_social_users': int (optional)

    Returns:
      dict with keys:
        - "object": dictionary with extracted statistics for Age_c (main effect)
                    and Age_c x Site interaction terms for both models (when present),
                    plus sample sizes.
        - "description": short plain-language interpretation of what these numbers mean
                         for whether children's reliance on majority preference develops
                         with age and whether that developmental trajectory differs by site.
    """
    import numpy as np

    def summarize_model(res):
        """Return dict of summaries for Age_c main effect and Age_c x Site interactions."""
        if res is None:
            return None

        params = res.params
        pvalues = res.pvalues
        conf = res.conf_int()
        bse = getattr(res, 'bse', None)

        summary = {}
        # Find the main Age_c coefficient
        age_name = None
        for n in params.index:
            if n == 'Age_c':
                age_name = n
                break
        if age_name is None:
            # sometimes interaction-only parameterization could occur, but we expect Age_c
            # if not found, try to find any parameter that equals exactly 'Age_c' ignoring case
            for n in params.index:
                if n.lower() == 'age_c'.lower():
                    age_name = n
                    break

        if age_name is not None:
            coef = float(params[age_name])
            p = float(pvalues[age_name])
            ci_low, ci_high = map(float, conf.loc[age_name])
            se = float(bse[age_name]) if bse is not None else None
            or_coef = float(np.exp(coef))
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
            summary['Age_c'] = {
                'param_name': age_name,
                'coef': coef,
                'se': se,
                'p_value': p,
                '95CI': [ci_low, ci_high],
                'odds_ratio': or_coef,
                'odds_ratio_95CI': list(or_ci),
                'significant_p05': p < 0.05
            }
        else:
            summary['Age_c'] = None

        # Find interaction terms that include Age_c and Site
        interaction_entries = {}
        for n in params.index:
            # typical interaction naming: 'Age_c:C(Site)[T.Site_2]' or 'Age_c:C(Site)[T.xyz]'
            if ('Age_c' in n) and ('Site' in n):
                coef = float(params[n])
                p = float(pvalues[n])
                ci_low, ci_high = map(float, conf.loc[n])
                se = float(bse[n]) if bse is not None else None
                or_coef = float(np.exp(coef))
                or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
                interaction_entries[n] = {
                    'param_name': n,
                    'coef': coef,
                    'se': se,
                    'p_value': p,
                    '95CI': [ci_low, ci_high],
                    'odds_ratio': or_coef,
                    'odds_ratio_95CI': list(or_ci),
                    'significant_p05': p < 0.05
                }
        # If none found, set empty dict
        summary['Age_c_by_Site_interactions'] = interaction_entries

        return summary

    output = {}
    # sample sizes if present
    if 'n_total' in model_output:
        output['n_total'] = int(model_output.get('n_total'))
    if 'n_social_users' in model_output:
        output['n_social_users'] = int(model_output.get('n_social_users'))

    # Extract summaries
    social_res = model_output.get('social_model', None)
    majority_res = model_output.get('majority_model', None)

    output['social_model_summary'] = summarize_model(social_res)
    output['majority_model_summary'] = summarize_model(majority_res) if majority_res is not None else None

    # Build a short plain-language description
    desc_lines = []
    desc_lines.append(f"Sample sizes: total n = {output.get('n_total', 'NA')}, "
                      f"social-users n = {output.get('n_social_users', 'NA')}.")
    # Social model interpretation
    ssum = output['social_model_summary']
    if ssum is None:
        desc_lines.append("No social model results available.")
    else:
        age_info = ssum.get('Age_c')
        if age_info is None:
            desc_lines.append("No main Age_c coefficient found in social model.")
        else:
            p = age_info['p_value']
            orv = age_info['odds_ratio']
            desc_lines.append(
                f"SocialReliance model: Age (Age_c) coef = {age_info['coef']:.3f}, "
                f"p = {p:.3f}. Odds ratio per year = {orv:.3f} "
                f"(95% CI {age_info['odds_ratio_95CI'][0]:.3f}–{age_info['odds_ratio_95CI'][1]:.3f})."
            )
            if age_info['significant_p05']:
                desc_lines.append("This indicates a statistically significant change in reliance on social information with age (p < 0.05).")
            else:
                desc_lines.append("No statistically significant evidence that reliance on social information changes with age (p >= 0.05).")

        inters = ssum.get('Age_c_by_Site_interactions', {})
        if inters:
            # summarize how many interactions significant
            n_inters = len(inters)
            n_sig = sum(1 for v in inters.values() if v['significant_p05'])
            desc_lines.append(f"SocialReliance model: found {n_inters} Age-by-Site interaction term(s); {n_sig} are individually significant (p < 0.05).")
            if n_sig > 0:
                sig_sites = [v['param_name'] for v in inters.values() if v['significant_p05']]
                desc_lines.append("Significant interaction parameters: " + ", ".join(sig_sites) + ". This suggests age-related change differs for those site(s).")
            else:
                desc_lines.append("No individual Age-by-Site interaction terms were significant; no strong evidence that age trajectories differ across sites based on individual tests.")
        else:
            desc_lines.append("No Age-by-Site interaction terms found in social model; no evidence from interactions that developmental change differs by site.")

    # Majority model interpretation (conditional on using social information)
    msum = output['majority_model_summary']
    if msum is None:
        desc_lines.append("MajorityChoice model: Not available or not fitted (too few social learners or model missing).")
    else:
        age_info = msum.get('Age_c')
        if age_info is None:
            desc_lines.append("MajorityChoice model: No main Age_c coefficient found.")
        else:
            p = age_info['p_value']
            orv = age_info['odds_ratio']
            desc_lines.append(
                f"MajorityChoice (among social users) model: Age (Age_c) coef = {age_info['coef']:.3f}, "
                f"p = {p:.3f}. Odds ratio per year = {orv:.3f} "
                f"(95% CI {age_info['odds_ratio_95CI'][0]:.3f}–{age_info['odds_ratio_95CI'][1]:.3f})."
            )
            if age_info['significant_p05']:
                desc_lines.append("This indicates a statistically significant change in preference for the majority with age among social learners (p < 0.05).")
            else:
                desc_lines.append("No statistically significant evidence that majority preference changes with age among social learners (p >= 0.05).")

        inters = msum.get('Age_c_by_Site_interactions', {})
        if inters:
            n_inters = len(inters)
            n_sig = sum(1 for v in inters.values() if v['significant_p05'])
            desc_lines.append(f"MajorityChoice model: found {n_inters} Age-by-Site interaction term(s); {n_sig} are individually significant.")
            if n_sig > 0:
                sig_sites = [v['param_name'] for v in inters.values() if v['significant_p05']]
                desc_lines.append("Significant interaction parameters: " + ", ".join(sig_sites) + ". This suggests site-specific age trends in majority preference.")
            else:
                desc_lines.append("No individual Age-by-Site interaction terms were significant in the MajorityChoice model.")
        else:
            desc_lines.append("No Age-by-Site interaction terms found in MajorityChoice model.")

    description = " ".join(desc_lines)

    return {"object": output, "description": description}