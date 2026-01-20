def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for:
      - ReaderView main effect
      - ReaderView:Dyslexia interaction
    Also computes the marginal effect of ReaderView for:
      - Non-dyslexic readers (Dyslexia=0) -> ReaderView coef
      - Dyslexic readers (Dyslexia=1) -> ReaderView coef + interaction coef
    Converts log-scale coefficients into percent changes (approx: 100*(exp(beta)-1)),
    and returns a decision about whether Reader View improves reading speed for
    dyslexic individuals (based on interaction significance and sign).
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    res = model_output

    # Helper to safely access parameter names (handles potential different naming)
    params = res.params
    cov = res.cov_params()

    # Determine parameter names
    # Expecting 'ReaderView', 'Dyslexia', and 'ReaderView:Dyslexia' (or similar)
    def find_name(contains):
        for name in params.index:
            if all(tok in name for tok in contains):
                return name
        return None

    name_rv = find_name(['ReaderView'])
    name_dys = find_name(['Dyslexia'])
    name_int = find_name(['ReaderView', 'Dyslexia'])  # looks for interaction containing both

    # Prepare result container
    out = {
        'coefficients': {},
        'marginal_effects_log': {},
        'marginal_effects_pct': {},
        'p_values': {},
        '95CI_log': {},
        '95CI_pct': {},
        'conclusion': None
    }

    # z for 95% CI (normal approx). For cluster-robust fits this is common.
    z95 = stats.norm.ppf(0.975)

    # Function for CI and pct conversion
    def summarize_coef(name):
        if name is None or name not in params.index:
            return None
        b = float(params[name])
        # If covariance matrix doesn't contain name, set se from res.bse if present
        try:
            var = float(cov.loc[name, name])
            se = sqrt(var)
        except Exception:
            # fallback to bse attribute if available
            try:
                se = float(res.bse[name])
            except Exception:
                se = None
        # p-value from results if available
        pval = float(res.pvalues[name]) if name in res.pvalues.index else None
        # CI on log scale
        if se is not None:
            ci_low = b - z95 * se
            ci_high = b + z95 * se
        else:
            ci_low = ci_high = None
        # percent effect and CI transformed
        pct = (np.exp(b) - 1) * 100
        pct_ci = ( (np.exp(ci_low) - 1) * 100 if ci_low is not None else None,
                   (np.exp(ci_high) - 1) * 100 if ci_high is not None else None )
        return {
            'beta': b,
            'se': se,
            'p_value': pval,
            'ci_log': (ci_low, ci_high),
            'pct_effect': pct,
            'ci_pct': pct_ci
        }

    # Summarize main and interaction
    rv_summary = summarize_coef(name_rv)
    int_summary = summarize_coef(name_int)

    out['coefficients']['ReaderView_name'] = name_rv
    out['coefficients']['Interaction_name'] = name_int
    out['coefficients']['ReaderView'] = rv_summary
    out['coefficients']['ReaderView:Dyslexia'] = int_summary

    # Compute marginal effects: ReaderView effect when Dyslexia=0 and Dyslexia=1
    # For Dyslexia=0:
    if rv_summary is not None:
        out['marginal_effects_log']['Dyslexia=0'] = rv_summary['beta']
        out['marginal_effects_pct']['Dyslexia=0'] = rv_summary['pct_effect']
        out['p_values']['ReaderView_Dys0'] = rv_summary['p_value']
        out['95CI_log']['ReaderView_Dys0'] = rv_summary['ci_log']
        out['95CI_pct']['ReaderView_Dys0'] = rv_summary['ci_pct']
    else:
        out['marginal_effects_log']['Dyslexia=0'] = None

    # For Dyslexia=1: beta_rv + beta_int
    if rv_summary is not None and int_summary is not None:
        b_rv = rv_summary['beta']
        b_int = int_summary['beta']
        # linear combination
        b_sum = b_rv + b_int
        # variance: var(r) + var(int) + 2*cov(r,int)
        try:
            var_r = float(cov.loc[name_rv, name_rv])
            var_int = float(cov.loc[name_int, name_int])
            cov_r_int = float(cov.loc[name_rv, name_int])
            se_sum = sqrt(var_r + var_int + 2 * cov_r_int)
        except Exception:
            # fallback: attempt to approximate using bse (less ideal)
            try:
                se_r = float(res.bse[name_rv])
                se_int = float(res.bse[name_int])
                se_sum = sqrt(se_r**2 + se_int**2)
            except Exception:
                se_sum = None

        # p-value using normal approx
        if se_sum is not None and se_sum > 0:
            z = b_sum / se_sum
            p_comb = 2 * (1 - stats.norm.cdf(abs(z)))
            ci_low = b_sum - z95 * se_sum
            ci_high = b_sum + z95 * se_sum
            pct = (np.exp(b_sum) - 1) * 100
            pct_ci = ((np.exp(ci_low) - 1) * 100, (np.exp(ci_high) - 1) * 100)
        else:
            p_comb = None
            ci_low = ci_high = None
            pct = (np.exp(b_sum) - 1) * 100
            pct_ci = (None, None)

        out['marginal_effects_log']['Dyslexia=1'] = b_sum
        out['marginal_effects_pct']['Dyslexia=1'] = pct
        out['p_values']['ReaderView_Dys1'] = p_comb
        out['95CI_log']['ReaderView_Dys1'] = (ci_low, ci_high)
        out['95CI_pct']['ReaderView_Dys1'] = pct_ci
    else:
        out['marginal_effects_log']['Dyslexia=1'] = None

    # Interaction significance & conclusion:
    interaction_coef = int_summary['beta'] if int_summary is not None else None
    interaction_p = int_summary['p_value'] if int_summary is not None else None

    conclusion = {
        'interpretation_rule': 'ReaderView helps dyslexic readers more if interaction > 0 and statistically significant (p < 0.05).',
        'interaction_coef': interaction_coef,
        'interaction_p_value': interaction_p
    }

    if interaction_coef is None:
        conclusion['decision'] = 'Inconclusive: interaction term not available in model output.'
    else:
        if (interaction_coef > 0) and (interaction_p is not None) and (interaction_p < 0.05):
            conclusion['decision'] = 'Yes — evidence that Reader View improves (relatively) more for dyslexic readers (interaction positive & significant).'
        elif (interaction_coef < 0) and (interaction_p is not None) and (interaction_p < 0.05):
            conclusion['decision'] = 'No — evidence that Reader View benefits dyslexic readers less (interaction negative & significant).'
        else:
            conclusion['decision'] = 'Inconclusive — interaction is not statistically significant at alpha=0.05.'

    out['conclusion'] = conclusion

    # Prepare a brief textual summary
    summary_lines = []
    # ReaderView main
    if rv_summary is not None:
        summary_lines.append(
            f"ReaderView (Dyslexia=0) log-coef = {rv_summary['beta']:.4f}, "
            f"pct ≈ {rv_summary['pct_effect']:.2f}%, p = {rv_summary['p_value']:.3g}"
        )
    # Interaction
    if int_summary is not None:
        summary_lines.append(
            f"Interaction ReaderView:Dyslexia log-coef = {int_summary['beta']:.4f}, "
            f"pct change (additive on log scale), p = {int_summary['p_value']:.3g}"
        )
    # Dyslexic marginal
    if out['marginal_effects_log']['Dyslexia=1'] is not None:
        summary_lines.append(
            f"ReaderView effect for Dyslexia=1 log-coef = {out['marginal_effects_log']['Dyslexia=1']:.4f}, "
            f"pct ≈ {out['marginal_effects_pct']['Dyslexia=1']:.2f}%, p = {out['p_values']['ReaderView_Dys1']:.3g}"
        )

    summary_lines.append("Conclusion: " + conclusion['decision'])
    description = " | ".join(summary_lines)

    # Return the object and description text
    return {
        "object": out,
        "description": description
    }