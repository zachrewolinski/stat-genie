def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of being female on mortgage acceptance
    from the model output returned by the provided `model` function.

    Returns a dictionary with:
      - "object": dict of numeric results (coef, robust SE, z, p, odds ratio and 95% CI,
                  average marginal effect, n_obs, significance at alpha=0.05)
      - "description": short interpretation of the numbers in context
    """
    import numpy as np
    from scipy.stats import norm

    # Helper to safely get values from possible containers
    def _get_from(obj, key):
        if obj is None:
            return None
        try:
            return obj[key]
        except Exception:
            try:
                return getattr(obj, key)
            except Exception:
                return None

    # Try to pull values from common places in model_output
    res = model_output.get('mle_result')
    robust_se = model_output.get('robust_se')
    robust_cov = model_output.get('robust_cov')
    ame = model_output.get('avg_marginal_effect_female', model_output.get('female_avg_marginal_effect'))
    summary = model_output.get('summary_dict', {})

    # 1) Coefficient (MLE)
    coef = None
    if res is not None:
        try:
            coef = float(res.params['female'])
        except Exception:
            # fallback to summary dict
            coef = summary.get('female_coef_mle', None)
    else:
        coef = summary.get('female_coef_mle', None)

    # 2) Robust SE for female
    se = None
    if robust_se is not None:
        try:
            # robust_se might be a Series or dict-like
            se = float(robust_se['female'])
        except Exception:
            try:
                se = float(robust_se.loc['female'])
            except Exception:
                se = None
    # If robust_se missing but robust_cov present, derive se from diagonal
    if se is None and robust_cov is not None:
        try:
            se = float(np.sqrt(robust_cov.loc['female', 'female']))
        except Exception:
            try:
                # if robust_cov is numpy array with index in summary
                se = float(np.sqrt(np.asarray(robust_cov)['female','female']))
            except Exception:
                se = None
    # Fallback to model-based SE if nothing else available
    if se is None and res is not None:
        try:
            se = float(res.bse['female'])
        except Exception:
            se = None

    # 3) Compute z-statistic and two-sided p-value using normal approximation
    z = None
    p_value = None
    if coef is not None and se is not None and se > 0:
        z = coef / se
        p_value = 2.0 * (1.0 - norm.cdf(abs(z)))

    # 4) Odds ratio and robust 95% CI (derived from coef +/- z*se)
    odds_ratio = None
    odds_ratio_ci95 = [None, None]
    if coef is not None:
        odds_ratio = float(np.exp(coef))
        if se is not None:
            z_975 = norm.ppf(0.975)
            ci_lo = coef - z_975 * se
            ci_hi = coef + z_975 * se
            odds_ratio_ci95 = [float(np.exp(ci_lo)), float(np.exp(ci_hi))]

    # 5) Average marginal effect
    avg_marginal_effect = None
    if ame is not None:
        try:
            avg_marginal_effect = float(ame)
        except Exception:
            avg_marginal_effect = None
    # If not present, try summary
    if avg_marginal_effect is None:
        avg_marginal_effect = summary.get('female_avg_marginal_effect', summary.get('female_avg_marginal_effect'))

    # 6) n_obs if available
    n_obs = summary.get('n_obs', None)

    # 7) Significance at 5% level (using computed p-value)
    significant_0_05 = None
    if p_value is not None:
        significant_0_05 = (p_value < 0.05)

    # Build output object
    object_out = {
        'female_coef_mle': None if coef is None else float(coef),
        'female_robust_se': None if se is None else float(se),
        'z_value': None if z is None else float(z),
        'p_value_two_sided': None if p_value is None else float(p_value),
        'female_odds_ratio': odds_ratio,
        'female_odds_ratio_robust_CI_95': odds_ratio_ci95,
        'female_avg_marginal_effect': avg_marginal_effect,
        'n_obs': n_obs,
        'significant_at_0.05': significant_0_05
    }

    # Short description / interpretation
    # Keep it concise and factual; highlight direction, magnitude, and significance if computable.
    if coef is None:
        description = "Could not find the female coefficient in the provided model output."
    else:
        desc_parts = []
        desc_parts.append(
            f"Adjusted log-odds coefficient for female = {coef:.3f}"
            if coef is not None else "Coefficient for female unavailable"
        )
        if se is not None:
            desc_parts.append(f"(robust SE = {se:.3f}, z = {z:.2f}, p = {p_value:.3f})")
        if odds_ratio is not None:
            desc_parts.append(
                f"which corresponds to an odds ratio = {odds_ratio:.3f}"
            )
            if odds_ratio_ci95[0] is not None:
                desc_parts.append(
                    f"(95% robust CI: {odds_ratio_ci95[0]:.3f} to {odds_ratio_ci95[1]:.3f})"
                )
        if avg_marginal_effect is not None:
            desc_parts.append(
                f"Average marginal effect: being female is associated with a {avg_marginal_effect*100:.2f} percentage-point change in approval probability."
            )
        if significant_0_05 is not None:
            sig_txt = "statistically significant at the 5% level" if significant_0_05 else "not statistically significant at the 5% level"
            desc_parts.append(f"The effect is {sig_txt} (two-sided).")
        # Note controls and caution
        desc_parts.append(
            "Estimates are conditional on included controls (race, self-employment, marital status, bad credit history, mortgage & consumer credit scores, PI ratio, LTV, and housing expense ratio); this is an association, not a proven causal effect."
        )
        description = " ".join(desc_parts)

    return {"object": object_out, "description": description}