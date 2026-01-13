def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of HasChildren on extramarital affairs
    from the provided fitted model objects.

    Expects model_output to be a dict with keys:
      - 'logit' : GLMResultsWrapper for Binomial model (no interaction)
      - 'logit_interaction' : GLMResultsWrapper for Binomial model (HasChildren * IsMale)
      - 'negbin' : GLMResultsWrapper for NegativeBinomial model (with interaction)

    Returns a dict with keys:
      - "object": nested dict containing coefficients, SEs, p-values, 95% CIs,
                  and exponentiated effects (odds ratios or incidence-rate ratios)
                  for HasChildren (overall) and for gender-specific effects when an
                  interaction model is available.
      - "description": plain-language summary interpreting those results.
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    out = {"object": {}, "description": ""}

    # Helper to format CI and exponentiated values
    def ci_dict(coef, se, z=1.96):
        lower = coef - z * se
        upper = coef + z * se
        return (lower, upper)

    # 1) Extract from simple logistic model (no interaction)
    try:
        logit = model_output['logit']
        varname = 'HasChildren'
        coef = float(logit.params[varname])
        se = float(logit.bse[varname])
        p = float(logit.pvalues[varname])
        ci_low, ci_high = map(float, logit.conf_int().loc[varname])
        or_ = float(np.exp(coef))
        or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

        out['object']['logit'] = {
            'coef': coef,
            'se': se,
            'p_value': p,
            '95%_CI_coef': (ci_low, ci_high),
            'odds_ratio': or_,
            '95%_CI_odds_ratio': or_ci,
            'interpretation': (
                "Log-odds coefficient < 0 implies lower odds of any affair when having children; "
                "odds ratio < 1 implies same in multiplicative terms."
            )
        }
    except Exception as e:
        out['object']['logit'] = {'error': f'Could not extract logit results: {e}'}

    # 2) Extract from logistic model with interaction (HasChildren * IsMale)
    try:
        logit_i = model_output['logit_interaction']
        pname_int = 'HasChildren_x_IsMale'
        # main effect (effect of HasChildren when IsMale=0, i.e., females if coding is 1=male)
        coef_H = float(logit_i.params['HasChildren'])
        se_H = float(logit_i.bse['HasChildren'])
        p_H = float(logit_i.pvalues['HasChildren'])
        ci_H = tuple(map(float, logit_i.conf_int().loc['HasChildren']))

        # interaction term
        coef_int = float(logit_i.params[pname_int])
        se_int = float(logit_i.bse[pname_int])
        p_int = float(logit_i.pvalues[pname_int])
        ci_int = tuple(map(float, logit_i.conf_int().loc[pname_int]))

        # Compute gender-specific linear effects:
        # Females (IsMale=0): coef_f = coef_H
        coef_f = coef_H
        se_f = se_H
        ci_f = ci_H
        or_f = float(np.exp(coef_f))
        or_ci_f = (float(np.exp(ci_f[0])), float(np.exp(ci_f[1])))
        p_f = p_H

        # Males (IsMale=1): coef_m = coef_H + coef_int
        coef_m = coef_H + coef_int
        # compute SE of sum using covariance
        cov = logit_i.cov_params()
        var_H = float(cov.loc['HasChildren', 'HasChildren'])
        var_int = float(cov.loc[pname_int, pname_int])
        cov_H_int = float(cov.loc['HasChildren', pname_int])
        var_m = var_H + var_int + 2.0 * cov_H_int
        se_m = sqrt(max(var_m, 0.0))
        # CI for male linear combo
        ci_m = ci_dict(coef_m, se_m)
        or_m = float(np.exp(coef_m))
        or_ci_m = (float(np.exp(ci_m[0])), float(np.exp(ci_m[1])))
        # p-value for male effect using z
        z_m = coef_m / se_m if se_m > 0 else np.nan
        p_m = float(2.0 * stats.norm.sf(abs(z_m))) if se_m > 0 else np.nan

        out['object']['logit_interaction'] = {
            'HasChildren_female': {
                'coef': coef_f,
                'se': se_f,
                'p_value': p_f,
                '95%_CI_coef': ci_f,
                'odds_ratio': or_f,
                '95%_CI_odds_ratio': or_ci_f,
            },
            'HasChildren_male': {
                'coef': coef_m,
                'se': se_m,
                'p_value': p_m,
                '95%_CI_coef': ci_m,
                'odds_ratio': or_m,
                '95%_CI_odds_ratio': or_ci_m,
            },
            'interaction_term': {
                'name': pname_int,
                'coef': coef_int,
                'se': se_int,
                'p_value': p_int,
                '95%_CI_coef': ci_int
            },
            'notes': (
                "Main HasChildren coefficient refers to IsMale=0 group (females under coding 1=male). "
                "Male effect is main + interaction; its SE and p-value computed via delta method."
            )
        }
    except Exception as e:
        out['object']['logit_interaction'] = {'error': f'Could not extract logit interaction results: {e}'}

    # 3) Extract from negative binomial model (count) with interaction
    try:
        nb = model_output['negbin']
        pname_int = 'HasChildren_x_IsMale'

        # main effect (IsMale=0)
        coef_nb_H = float(nb.params['HasChildren'])
        se_nb_H = float(nb.bse['HasChildren'])
        p_nb_H = float(nb.pvalues['HasChildren'])
        ci_nb_H = tuple(map(float, nb.conf_int().loc['HasChildren']))
        irr_nb_H = float(np.exp(coef_nb_H))
        irr_ci_nb_H = (float(np.exp(ci_nb_H[0])), float(np.exp(ci_nb_H[1])))

        # interaction
        coef_nb_int = float(nb.params[pname_int])
        se_nb_int = float(nb.bse[pname_int])
        p_nb_int = float(nb.pvalues[pname_int])
        ci_nb_int = tuple(map(float, nb.conf_int().loc[pname_int]))

        # Female (IsMale=0)
        coef_nb_f = coef_nb_H
        se_nb_f = se_nb_H
        irr_nb_f = irr_nb_H
        irr_ci_nb_f = irr_ci_nb_H
        p_nb_f = p_nb_H

        # Male (IsMale=1): sum
        coef_nb_m = coef_nb_H + coef_nb_int
        cov_nb = nb.cov_params()
        var_nb_H = float(cov_nb.loc['HasChildren', 'HasChildren'])
        var_nb_int = float(cov_nb.loc[pname_int, pname_int])
        cov_nb_H_int = float(cov_nb.loc['HasChildren', pname_int])
        var_nb_m = var_nb_H + var_nb_int + 2.0 * cov_nb_H_int
        se_nb_m = sqrt(max(var_nb_m, 0.0))
        ci_nb_m = ci_dict(coef_nb_m, se_nb_m)
        irr_nb_m = float(np.exp(coef_nb_m))
        irr_ci_nb_m = (float(np.exp(ci_nb_m[0])), float(np.exp(ci_nb_m[1])))
        z_nb_m = coef_nb_m / se_nb_m if se_nb_m > 0 else np.nan
        p_nb_m = float(2.0 * stats.norm.sf(abs(z_nb_m))) if se_nb_m > 0 else np.nan

        out['object']['negbin_interaction'] = {
            'HasChildren_female': {
                'coef': coef_nb_f,
                'se': se_nb_f,
                'p_value': p_nb_f,
                '95%_CI_coef': ci_nb_H,
                'incidence_rate_ratio': irr_nb_f,
                '95%_CI_IRR': irr_ci_nb_f,
            },
            'HasChildren_male': {
                'coef': coef_nb_m,
                'se': se_nb_m,
                'p_value': p_nb_m,
                '95%_CI_coef': ci_nb_m,
                'incidence_rate_ratio': irr_nb_m,
                '95%_CI_IRR': irr_ci_nb_m,
            },
            'interaction_term': {
                'name': pname_int,
                'coef': coef_nb_int,
                'se': se_nb_int,
                'p_value': p_nb_int,
                '95%_CI_coef': ci_nb_int
            },
            'notes': "Negative binomial coefficients are on the log count scale; exponentiated values are incidence rate ratios (IRR)."
        }

    except Exception as e:
        out['object']['negbin_interaction'] = {'error': f'Could not extract negbin results: {e}'}

    # 4) Build a concise description interpreting the results (based primarily on simple logit)
    try:
        # Use simple logit results as primary inference
        l = out['object'].get('logit')
        summary_lines = []
        if l and 'error' not in l:
            coef = l['coef']; p = l['p_value']; or_ = l['odds_ratio']
            # direction
            if coef < 0:
                direction = "associated with lower odds of engaging in an extramarital affair"
            elif coef > 0:
                direction = "associated with higher odds of engaging in an extramarital affair"
            else:
                direction = "no association (coefficient exactly zero)"
            # significance
            sig = "statistically significant (p < 0.05)" if p < 0.05 else f"not statistically significant (p = {p:.3f})"
            summary_lines.append(
                f"In the primary logistic model, having children is {direction}; "
                f"odds ratio = {or_:.2f}, {sig}."
            )
        else:
            summary_lines.append("Primary logistic model results unavailable for summary.")

        # Interaction info
        li = out['object'].get('logit_interaction')
        if li and 'error' not in li:
            int_p = li['interaction_term']['p_value']
            female_or = li['HasChildren_female']['odds_ratio']
            male_or = li['HasChildren_male']['odds_ratio']
            summary_lines.append(
                f"The interaction model shows HasChildren * IsMale interaction p = {int_p:.3f}. "
                f"Estimated odds ratio for females = {female_or:.2f}; for males = {male_or:.2f}."
            )
            if int_p < 0.05:
                summary_lines.append("The interaction is statistically significant, indicating the effect of children differs by gender.")
            else:
                summary_lines.append("The interaction is not statistically significant, so we do not have strong evidence the effect differs by gender.")
        else:
            summary_lines.append("Logistic interaction model results unavailable for summary.")

        # Negative binomial check
        nbj = out['object'].get('negbin_interaction')
        if nbj and 'error' not in nbj:
            nb_f = nbj['HasChildren_female']
            nb_m = nbj['HasChildren_male']
            summary_lines.append(
                f"As a robustness check, negative-binomial results (IRR) are: females IRR = {nb_f['incidence_rate_ratio']:.2f} (p = {nb_f['p_value']:.3f}), "
                f"males IRR = {nb_m['incidence_rate_ratio']:.2f} (p = {nb_m['p_value']:.3f})."
            )
        else:
            summary_lines.append("Negative-binomial robustness results unavailable for summary.")

        out['description'] = " ".join(summary_lines)
    except Exception as e:
        out['description'] = f"Could not build description: {e}"

    return out