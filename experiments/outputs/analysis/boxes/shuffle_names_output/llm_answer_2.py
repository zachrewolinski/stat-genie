def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether reliance on the majority changes with age
    overall and whether that developmental effect differs across sites (interactions).
    Returns a dict with:
      - "object": nested dict of extracted numeric results (coefficients, SEs, p-values, CIs, ORs)
      - "description": short, plain-language interpretation of each extracted result
    """
    import numpy as np
    import pandas as pd
    from math import exp, sqrt

    results = {}
    desc_lines = []

    # Primary binary logit model
    logit = model_output.get('binary_majority_logit', None)
    if logit is None:
        raise ValueError("binary_majority_logit not found in model_output")

    params = logit.params
    bse = logit.bse
    pvals = logit.pvalues
    ci = logit.conf_int()
    cov = logit.cov_params()

    # 1) Main age effect in the reference site (omitted site)
    if 'AgeYears' in params.index:
        age_coef = float(params['AgeYears'])
        age_se = float(bse['AgeYears'])
        age_p = float(pvals['AgeYears'])
        age_ci = tuple(ci.loc['AgeYears'].astype(float))
        age_or = exp(age_coef)
        age_or_ci = (exp(age_ci[0]), exp(age_ci[1]))

        results['age_effect_reference_site'] = {
            'log_odds_coef': age_coef,
            'se': age_se,
            'z': age_coef / age_se if age_se != 0 else None,
            'p_value': age_p,
            '95CI_log_odds': age_ci,
            'OR_per_year': age_or,
            '95CI_OR': age_or_ci,
            'note': 'This is the effect of one additional year of age on the log-odds (and odds ratio) of choosing the majority for the reference (omitted) site.'
        }

        desc_lines.append(
            "Reference (omitted) site: per-year change in log-odds = {coef:.3f} (SE={se:.3f}), p={p:.3g}; "
            "OR per year = {or_:.3f} (95% CI {or_low:.3f}, {or_high:.3f})."
            .format(coef=age_coef, se=age_se, p=age_p, or_=age_or, or_low=age_or_ci[0], or_high=age_or_ci[1])
        )
    else:
        results['age_effect_reference_site'] = None
        desc_lines.append("No 'AgeYears' main effect found in model parameters.")

    # 2) Interaction effects: Age x Site (how age effect differs by site)
    # Interaction parameter names were created as '{site_dummy}:AgeYears' in the model code.
    interaction_names = [n for n in params.index if ':AgeYears' in n]
    interaction_summary = {}
    if len(interaction_names) == 0:
        desc_lines.append("No Age x Site interaction terms found in the fitted logit model.")
    else:
        # For each site dummy, compute the combined age effect = AgeYears + (AgeYears:SiteX)
        for inter in interaction_names:
            # site dummy name (the part before ':AgeYears')
            site_dummy = inter.split(':AgeYears')[0]
            inter_coef = float(params[inter])
            inter_se = float(bse[inter])
            inter_p = float(pvals[inter])
            inter_ci = tuple(ci.loc[inter].astype(float))

            # Combined effect for that site = age_coef (reference) + interaction coef
            if 'AgeYears' in params.index:
                combined_coef = age_coef + inter_coef
                # Var(combined) = Var(AgeYears) + Var(inter) + 2*Cov(Age,inter)
                cov_age_inter = float(cov.loc['AgeYears', inter]) if ('AgeYears' in cov.index and inter in cov.columns) else 0.0
                var_combined = float(cov.loc['AgeYears', 'AgeYears']) + float(cov.loc[inter, inter]) + 2.0 * cov_age_inter
                combined_se = sqrt(var_combined) if var_combined > 0 else None
                combined_z = combined_coef / combined_se if combined_se and combined_se != 0 else None
                # compute p-value and CI using normal approx if possible
                if combined_se:
                    from scipy.stats import norm
                    combined_p = 2.0 * (1.0 - norm.cdf(abs(combined_z)))
                    combined_ci = (combined_coef - 1.96 * combined_se, combined_coef + 1.96 * combined_se)
                else:
                    combined_p = None
                    combined_ci = (None, None)
                combined_or = exp(combined_coef) if combined_coef is not None else None
                combined_or_ci = (exp(combined_ci[0]), exp(combined_ci[1])) if None not in combined_ci else (None, None)
            else:
                combined_coef = None
                combined_se = None
                combined_p = None
                combined_ci = (None, None)
                combined_or = None
                combined_or_ci = (None, None)

            interaction_summary[site_dummy] = {
                'interaction_coef (Age:SiteDummy)': inter_coef,
                'interaction_se': inter_se,
                'interaction_p_value': inter_p,
                'interaction_95CI_log_odds': inter_ci,
                'combined_age_effect_log_odds': combined_coef,
                'combined_se': combined_se,
                'combined_z': combined_z,
                'combined_p_value_approx': combined_p,
                'combined_95CI_log_odds': combined_ci,
                'combined_OR_per_year': combined_or,
                'combined_95CI_OR': combined_or_ci,
                'note': 'combined effect = AgeYears + (AgeYears:SiteDummy) corresponds to the per-year age effect in this site.'
            }

            desc_lines.append(
                "Site dummy '{site}': interaction coef (added to reference age effect) = {intc:.3f} (p={p:.3g}); "
                "combined per-year log-odds = {comb:.3f} (approx p={cp:.3g}), OR per year = {or_:.3f}."
                .format(site=site_dummy, intc=inter_coef, p=inter_p,
                        comb=(combined_coef if combined_coef is not None else float('nan')),
                        cp=(combined_p if combined_p is not None else float('nan')),
                        or_=(combined_or if combined_or is not None else float('nan')))
            )

        results['age_by_site_interactions'] = interaction_summary

        # Joint test: H0 all interaction coefficients = 0
        try:
            # Build restriction matrix R that picks out the interaction coefficients
            param_names = list(params.index)
            k = len(param_names)
            m = len(interaction_names)
            R = np.zeros((m, k))
            for i, inter in enumerate(interaction_names):
                idx = param_names.index(inter)
                R[i, idx] = 1.0
            wtest = logit.wald_test(R)
            # wtest may have attributes 'pvalue' or 'p_value' depending on version; try both
            joint_p = getattr(wtest, 'pvalue', None)
            if joint_p is None:
                joint_p = getattr(wtest, 'p_value', None)
            # statistic
            w_stat = getattr(wtest, 'statistic', None)
            results['interaction_joint_test'] = {
                'wald_statistic': float(w_stat) if w_stat is not None else None,
                'p_value': float(joint_p) if joint_p is not None else None,
                'df': int(m)
            }
            desc_lines.append(
                "Joint Wald test for all Age x Site interactions: chi2/stat = {stat}, df = {df}, p = {p}."
                .format(stat=(float(w_stat) if w_stat is not None else 'NA'), df=m, p=(float(joint_p) if joint_p is not None else 'NA'))
            )
        except Exception as e:
            results['interaction_joint_test'] = {'error': str(e)}
            desc_lines.append("Could not compute joint Wald test for interactions: " + str(e))

    # 3) Secondary: multinomial logit age effect for choosing majority (vs baseline)
    mn = model_output.get('multinomial_choice_mnlogit', None)
    if mn is None:
        desc_lines.append("Multinomial model not found in model_output.")
        results['multinomial_age_effect_majority'] = None
    else:
        # params is a DataFrame: index = parameter names, columns = outcome categories (non-baseline)
        try:
            mn_params = mn.params
            mn_pvals = mn.pvalues
            # Based on the original model, 2 = majority, 1 = unchosen baseline. Try to find column '2'
            col_candidates = list(mn_params.columns)
            majority_col = None
            for c in col_candidates:
                if str(c) == '2' or c == 2:
                    majority_col = c
                    break
            if majority_col is None:
                # fallback: choose the first non-baseline column (user should verify mapping)
                majority_col = col_candidates[0]
            if 'AgeYears' in mn_params.index:
                mn_age_coef = float(mn_params.loc['AgeYears', majority_col])
                mn_age_p = float(mn_pvals.loc['AgeYears', majority_col])
                # MNLogit does not directly provide bse in same shape but it is available
                try:
                    mn_age_se = float(mn.bse.loc['AgeYears', majority_col])
                except Exception:
                    mn_age_se = None
                mn_age_ci = None
                if mn_age_se is not None:
                    mn_age_ci = (mn_age_coef - 1.96 * mn_age_se, mn_age_coef + 1.96 * mn_age_se)
                mn_age_or = exp(mn_age_coef)
                results['multinomial_age_effect_majority'] = {
                    'outcome_column': str(majority_col),
                    'log_odds_coef': mn_age_coef,
                    'se': mn_age_se,
                    'p_value': mn_age_p,
                    '95CI_log_odds_approx': mn_age_ci,
                    'OR_per_year': mn_age_or,
                    'note': "From MNLogit: effect of AgeYears on choosing the 'majority' outcome vs baseline category."
                }
                desc_lines.append(
                    "Multinomial (majority vs baseline): AgeYears coef = {coef:.3f} (SE={se}), p={p:.3g}; OR={or_.3f}."
                    .format(coef=mn_age_coef, se=(mn_age_se if mn_age_se is not None else 'NA'), p=mn_age_p, or_=mn_age_or)
                )
            else:
                results['multinomial_age_effect_majority'] = None
                desc_lines.append("No AgeYears parameter found in multinomial model parameters.")
        except Exception as e:
            results['multinomial_age_effect_majority'] = {'error': str(e)}
            desc_lines.append("Error extracting multinomial age effect: " + str(e))

    # Compose a short description string summarizing main conclusions
    description = ";\n".join(desc_lines)

    return {
        "object": results,
        "description": description
    }