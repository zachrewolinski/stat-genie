def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (logistic regression)
    that tested: win ~ RelGroupSize_z * RelDistance_z + MaleDiff_z + FemaleDiff_z + TotalSize_z + C(dyad)
    
    Returns a dictionary with:
      - "object": a dict containing coefficients, std errors, p-values, 95% CIs, odds ratios,
                  interaction-derived marginal effects of a 1-SD increase in RelGroupSize_z
                  at RelDistance_z = +1 (closer to focal home) and -1 (closer to other home),
                  and predicted probabilities for representative combinations.
      - "description": a brief interpretation of the results in the context of the task.
    
    The function is defensive: it checks for presence of expected terms and will include
    warnings/messages for any missing terms.
    """
    import numpy as np
    import pandas as pd
    from scipy.special import expit  # logistic function

    res = model_output  # statsmodels results wrapper
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    try:
        conf = res.conf_int()
        conf.columns = ['2.5%', '97.5%']
    except Exception:
        # fallback: compute approx conf using bse
        z = 1.96
        conf = pd.DataFrame({
            '2.5%': params - z * bse,
            '97.5%': params + z * bse
        })

    # Helper to safely extract a coefficient row (or None)
    def get_term(name):
        if name in params.index:
            coef = float(params[name])
            se = float(bse[name]) if name in bse.index else None
            p = float(pvalues[name]) if name in pvalues.index else None
            ci_low = float(conf.loc[name, '2.5%']) if name in conf.index else None
            ci_high = float(conf.loc[name, '97.5%']) if name in conf.index else None
            return {'coef': coef, 'se': se, 'pvalue': p, 'ci_low': ci_low, 'ci_high': ci_high}
        else:
            return None

    # Expected term names
    term_RGS = 'RelGroupSize_z'
    term_RD = 'RelDistance_z'
    term_int = 'RelGroupSize_z:RelDistance_z'  # statsmodels uses ':' for interaction
    term_inter_alt = 'RelDistance_z:RelGroupSize_z'  # just in case order swapped

    rgs = get_term(term_RGS)
    rd = get_term(term_RD)
    interaction = get_term(term_int) or get_term(term_inter_alt)
    intercept = get_term('Intercept')

    # Compute odds ratios and their CIs for available terms
    def or_and_ci(term):
        if term is None:
            return None
        or_val = np.exp(term['coef'])
        ci_low = np.exp(term['ci_low']) if term['ci_low'] is not None else None
        ci_high = np.exp(term['ci_high']) if term['ci_high'] is not None else None
        return {'OR': or_val, 'OR_ci_low': ci_low, 'OR_ci_high': ci_high}

    rgs_or = or_and_ci(rgs)
    rd_or = or_and_ci(rd)
    int_or = or_and_ci(interaction)

    # Compute marginal effect of a 1-SD increase in RelGroupSize_z on log-odds at specified RelDistance_z values.
    # delta_logit = beta_RGS + beta_int * RelDistance_value
    # SE(delta_logit) uses variance-covariance matrix
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    def marginal_effect_at(rel_dist_value):
        if rgs is None:
            return None
        beta_rgs = rgs['coef']
        beta_int = interaction['coef'] if interaction is not None else 0.0
        delta_logit = beta_rgs + beta_int * rel_dist_value

        # compute SE using var-cov if available
        se_delta = None
        ci_low = None
        ci_high = None
        if cov is not None:
            names = list(cov.index)
            # default cov names must include both terms; if not, try swapped name
            name_rgs = term_RGS if term_RGS in names else None
            name_int = None
            if interaction is not None:
                # find which index corresponds to interaction term
                if term_int in names:
                    name_int = term_int
                elif term_inter_alt in names:
                    name_int = term_inter_alt
            # If both found, compute variance
            if name_rgs is not None and name_int is not None:
                var_rgs = cov.loc[name_rgs, name_rgs]
                var_int = cov.loc[name_int, name_int]
                covar = cov.loc[name_rgs, name_int]
                var_delta = var_rgs + (rel_dist_value ** 2) * var_int + 2 * rel_dist_value * covar
                se_delta = float(np.sqrt(var_delta)) if var_delta >= 0 else None
        # approximate CI on logit scale if se available
        if se_delta is not None:
            z = 1.96
            ci_low = delta_logit - z * se_delta
            ci_high = delta_logit + z * se_delta

        # transform to odds ratio and probability at baseline (other covariates = 0, reference dyad)
        or_effect = float(np.exp(delta_logit))
        or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high))) if (ci_low is not None and ci_high is not None) else (None, None)

        # predicted probabilities for two example RelGroupSize_z values:
        # baseline: RelGroupSize_z = -1 (smaller), vs +1 (larger) at the given rel_dist_value,
        # with other covariates set to 0 and using intercept if available.
        def pred_prob(rgs_val, rd_val):
            # use intercept if present (reference dyad)
            intercept_coef = intercept['coef'] if intercept is not None else 0.0
            beta_rd = rd['coef'] if rd is not None else 0.0
            beta_rgs_local = beta_rgs
            beta_int_local = beta_int
            logit = intercept_coef + beta_rgs_local * rgs_val + beta_rd * rd_val + beta_int_local * (rgs_val * rd_val)
            prob = float(expit(logit))
            return {'rgs': rgs_val, 'rd': rd_val, 'logit': float(logit), 'prob': prob}

        prob_minus1 = pred_prob(-1.0, rel_dist_value)
        prob_plus1 = pred_prob(+1.0, rel_dist_value)
        prob_diff = prob_plus1['prob'] - prob_minus1['prob']

        return {
            'rel_dist_value': rel_dist_value,
            'delta_logit': float(delta_logit),
            'delta_se': float(se_delta) if se_delta is not None else None,
            'delta_logit_ci': (float(ci_low), float(ci_high)) if (ci_low is not None and ci_high is not None) else (None, None),
            'OR_for_1sd_increase_RelGroupSize': or_effect,
            'OR_ci': or_ci,
            'predicted_prob_when_RelGroupSize_-1': prob_minus1,
            'predicted_prob_when_RelGroupSize_+1': prob_plus1,
            'predicted_prob_difference(+1 minus -1)': prob_diff
        }

    me_at_plus1 = marginal_effect_at(+1.0)  # contest closer to focal group's home
    me_at_minus1 = marginal_effect_at(-1.0)  # contest closer to other group's home

    # Build output object
    output_object = {
        'term_summary': {
            'Intercept': intercept,
            'RelGroupSize_z': rgs,
            'RelDistance_z': rd,
            'RelGroupSize_z:RelDistance_z': interaction
        },
        'odds_ratios': {
            'RelGroupSize_z': rgs_or,
            'RelDistance_z': rd_or,
            'Interaction': int_or
        },
        'marginal_effects_of_1sd_RelGroupSize': {
            'at_RelDistance_+1 (closer to focal home)': me_at_plus1,
            'at_RelDistance_-1 (closer to other home)': me_at_minus1
        },
        'notes': {
            'covariance_available': cov is not None,
            'dyad_fixed_effects_present': any(idx.startswith('C(dyad)') or idx.startswith('C(dyad)') for idx in params.index)
        }
    }

    # Short description interpretation
    # We'll produce cautious language: indicate sign and significance where available.
    def sign_and_sig(term):
        if term is None:
            return "term not in model"
        s = "positive" if term['coef'] > 0 else "negative" if term['coef'] < 0 else "null"
        sig = "significant (p<0.05)" if (term['pvalue'] is not None and term['pvalue'] < 0.05) else \
              "marginal/non-significant (p>=0.05)" if term['pvalue'] is not None else "p-value unavailable"
        return f"{s}, {sig} (coef={term['coef']:.3f}, p={term['pvalue']:.3g})"

    rgs_desc = sign_and_sig(rgs)
    rd_desc = sign_and_sig(rd)
    int_desc = sign_and_sig(interaction)

    description_lines = [
        "Summary interpretation (based on model coefficients; other covariates held at mean/zero):",
        f"- RelGroupSize_z: {rgs_desc}. A positive coefficient means larger focal groups are more likely to win.",
        f"- RelDistance_z: {rd_desc}. A positive coefficient means contests closer to the focal group's home increase its win probability.",
        f"- Interaction (RelGroupSize_z x RelDistance_z): {int_desc}. A positive interaction means the advantage of being numerically larger is stronger when the contest is nearer the focal group's home.",
        "",
        "Marginal effects (example):",
        ("- At RelDistance_z = +1 (contest relatively closer to focal group's home): "
         "the log-odds change for a 1-SD increase in RelGroupSize_z is given under 'marginal_effects_of_1sd_RelGroupSize'."),
        ("- At RelDistance_z = -1 (contest relatively closer to the other group's home): "
         "same as above; compare the ORs and predicted probability differences to see how location moderates the size effect."),
        "",
        "Caveats:",
        "- Predictions and marginal effects use the model intercept (reference dyad) and set other standardized covariates to 0.",
        "- If dyad fixed effects are many, the intercept applies to the reference dyad; dyad-specific intercepts will shift probabilities for particular dyads.",
        "- If covariance matrix is unavailable, uncertainty for interaction-derived marginal effects will not be provided."
    ]

    return {'object': output_object, 'description': "\n".join(description_lines)}