def extract_final_answer(model_output):
    """
    Extracts the estimated effect of applicant gender (female) on mortgage acceptance
    from a fitted statsmodels binary outcome model (Logit or GLM results wrapper).
    
    Returns a dict with:
      - "object": a dict of numeric results (coefficients, SEs, p-values, odds ratios, 95% CIs)
      - "description": brief human-readable interpretation of the results and significance
    
    The function handles models that include:
      - 'female' (main effect)
      - optional interaction 'female_black' (female * black)
    
    If 'female_black' is present, the function reports:
      - effect of being female for non-Black applicants (female coef)
      - effect of being female for Black applicants (female + female_black), with SE/p-value via delta method
    """
    import math
    import numpy as np

    res = model_output  # statsmodels results wrapper expected

    # Helper: normal two-sided p-value from z using math.erfc (no scipy dependency)
    def two_sided_p_from_z(z):
        return float(math.erfc(abs(z) / math.sqrt(2.0)))

    # Ensure we can access key pieces
    try:
        params = res.params  # pandas Series
        cov = res.cov_params()  # DataFrame or ndarray
        pvalues = getattr(res, "pvalues", None)
        bse = getattr(res, "bse", None)
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract model quantities from model_output: {e}"
        }

    # Convert to dict-like lookups
    param_names = list(params.index)
    def has(name):
        return name in param_names

    results_obj = {}
    descriptions = []

    if not has('female'):
        return {
            "object": None,
            "description": "The model does not contain a 'female' coefficient; cannot estimate gender effect."
        }

    # Get female main effect
    beta_f = float(params['female'])
    se_f = float(bse['female']) if bse is not None and 'female' in bse.index else None
    p_f = float(pvalues['female']) if pvalues is not None and 'female' in pvalues.index else None
    # 95% CI using normal approx if conf_int not directly used
    z_crit = 1.96
    if se_f is not None:
        ci_low_f = beta_f - z_crit * se_f
        ci_high_f = beta_f + z_crit * se_f
    else:
        # try using conf_int if available
        try:
            ci_df = res.conf_int()
            ci_low_f = float(ci_df.loc['female', 0])
            ci_high_f = float(ci_df.loc['female', 1])
        except Exception:
            ci_low_f = None
            ci_high_f = None

    or_f = float(math.exp(beta_f))
    or_ci_f = [float(math.exp(ci_low_f)) if ci_low_f is not None else None,
               float(math.exp(ci_high_f)) if ci_high_f is not None else None]

    results_obj['female_main'] = {
        "coef": beta_f,
        "std_err": se_f,
        "p_value": p_f,
        "odds_ratio": or_f,
        "95%CI_coef": [ci_low_f, ci_high_f],
        "95%CI_odds_ratio": or_ci_f
    }

    # Interpret significance for non-Black applicants (black=0)
    sig_f = None
    if p_f is not None:
        sig_f = (p_f < 0.05)
        descriptions.append(
            f"For non-Black applicants (black=0), the estimated log-odds effect of being female is {beta_f:.4f} "
            f"(SE={se_f:.4f}, p={p_f:.3g}). Odds ratio = {or_f:.3f}, 95% CI = [{or_ci_f[0]:.3f}, {or_ci_f[1]:.3f}]. "
            + ("This is statistically significant at the 5% level." if sig_f else "Not statistically significant at the 5% level.")
        )
    else:
        descriptions.append(
            f"For non-Black applicants (black=0), the estimated log-odds effect of being female is {beta_f:.4f}."
        )

    # If interaction is present, compute female effect for Black applicants using delta method
    if has('female_black'):
        beta_fb = float(params['female_black'])
        # Variance of sum = Var(female) + Var(female_black) + 2*Cov(female, female_black)
        try:
            # cov may be DataFrame or ndarray
            cov_df = cov
            cov_ff = float(cov_df.loc['female', 'female'])
            cov_fbf = float(cov_df.loc['female_black', 'female_black'])
            cov_cross = float(cov_df.loc['female', 'female_black'])
        except Exception:
            # fallback: if cov is ndarray and param order is known
            try:
                cov_arr = np.asarray(cov)
                idx_f = param_names.index('female')
                idx_fb = param_names.index('female_black')
                cov_ff = float(cov_arr[idx_f, idx_f])
                cov_fbf = float(cov_arr[idx_fb, idx_fb])
                cov_cross = float(cov_arr[idx_f, idx_fb])
            except Exception as e:
                return {
                    "object": None,
                    "description": f"Could not extract covariance matrix entries needed for delta method: {e}"
                }

        beta_f_black = beta_f + beta_fb
        var_sum = cov_ff + cov_fbf + 2.0 * cov_cross
        se_f_black = float(math.sqrt(var_sum)) if var_sum >= 0 else None
        p_f_black = None
        if se_f_black is not None:
            z = beta_f_black / se_f_black if se_f_black != 0 else float('inf') if beta_f_black != 0 else 0.0
            p_f_black = two_sided_p_from_z(z)
            ci_low = beta_f_black - z_crit * se_f_black
            ci_high = beta_f_black + z_crit * se_f_black
        else:
            ci_low = ci_high = None

        or_f_black = float(math.exp(beta_f_black))
        or_ci_f_black = [float(math.exp(ci_low)) if ci_low is not None else None,
                         float(math.exp(ci_high)) if ci_high is not None else None]

        results_obj['female_black_interaction'] = {
            "interaction_coef": beta_fb,
            "interaction_std_err": float(math.sqrt(cov_fbf)),
            # main+interaction (effect of female for Black applicants)
            "female_effect_black_coef": beta_f_black,
            "female_effect_black_std_err": se_f_black,
            "female_effect_black_p_value": p_f_black,
            "female_effect_black_odds_ratio": or_f_black,
            "female_effect_black_95%CI_coef": [ci_low, ci_high],
            "female_effect_black_95%CI_odds_ratio": or_ci_f_black
        }

        sig_fb = (p_f_black is not None and p_f_black < 0.05)
        descriptions.append(
            f"For Black applicants (black=1), the estimated log-odds effect of being female is {beta_f_black:.4f} "
            f"(SE={se_f_black:.4f}, p={p_f_black:.3g}). Odds ratio = {or_f_black:.3f}, "
            f"95% CI = [{or_ci_f_black[0]:.3f}, {or_ci_f_black[1]:.3f}]. "
            + ("This is statistically significant at the 5% level." if sig_fb else "Not statistically significant at the 5% level.")
        )

        # Short summary about moderation
        # Test whether the interaction itself is significant (does gender effect differ by race?)
        p_inter = float(pvalues['female_black']) if pvalues is not None and 'female_black' in pvalues.index else None
        if p_inter is not None:
            if p_inter < 0.05:
                descriptions.append(
                    "The female_black interaction coefficient is statistically significant (p = {:.3g}), "
                    "which suggests the effect of gender on acceptance differs by race (Black vs non-Black)."
                    .format(p_inter)
                )
            else:
                descriptions.append(
                    "The female_black interaction coefficient is not statistically significant (p = {:.3g}), "
                    "so there is no strong evidence that the gender effect differs between Black and non-Black applicants."
                    .format(p_inter)
                )

    # Final concise decision about whether gender affects approval overall
    # We'll consider gender effect for the reference group (non-Black). If interaction exists and is significant,
    # we already reported moderation.
    overall_statement = ""
    if 'female_black_interaction' in results_obj:
        # If either female effect for non-Black or for Black is significant, state that gender affects acceptance for that subgroup
        sig_nonblack = (p_f is not None and p_f < 0.05)
        p_black = results_obj['female_black_interaction']['female_effect_black_p_value']
        sig_black = (p_black is not None and p_black < 0.05)
        if sig_nonblack or sig_black:
            parts = []
            if sig_nonblack:
                parts.append("non-Black applicants")
            if sig_black:
                parts.append("Black applicants")
            overall_statement = "Gender (female) has a statistically detectable effect on acceptance for: " + ", ".join(parts) + "."
        else:
            overall_statement = "No statistically detectable effect of gender on acceptance for either subgroup at the 5% level."
    else:
        # No interaction: look at female main effect
        if p_f is not None and p_f < 0.05:
            overall_statement = "Gender (female) has a statistically significant effect on acceptance (p < 0.05)."
        else:
            overall_statement = "No statistically significant effect of gender (female) on acceptance at the 5% level."

    description_full = "\n".join(descriptions + ["", overall_statement])

    return {
        "object": results_obj,
        "description": description_full
    }