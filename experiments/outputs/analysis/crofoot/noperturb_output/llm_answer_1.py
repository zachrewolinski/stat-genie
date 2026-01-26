def extract_final_answer(model_output):
    """
    Extract relevant statistics from a fitted statsmodels binary outcome result (BinaryResultsWrapper).
    Returns a dictionary with:
      - "object": dict of extracted numeric results (coefficients, SEs, z, p, 95% CIs, odds ratios)
      - "description": brief interpretation regarding whether size advantage and location predict winning,
                       and whether being closer to home moderates the size advantage.
    """
    import numpy as np
    from math import exp
    try:
        from scipy.stats import norm
    except Exception:
        # Minimal replacement for normal CDF if scipy not available
        def _norm_cdf(x):
            return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))
        class norm:
            @staticmethod
            def cdf(x):
                return _norm_cdf(x)

    params = model_output.params  # pandas Series
    names = list(params.index)

    # Obtain covariance matrix used for inference (should reflect cluster-robust cov if fit that way)
    try:
        cov = model_output.cov_params()
    except Exception:
        # fallback to default covariance (may be HC or model default)
        try:
            cov = model_output.cov_params_default
        except Exception:
            cov = None

    # Helper to get numeric series of standard errors, z, p, conf if cov unavailable
    def se_from_cov(covmat):
        if covmat is None:
            # fallback to model_output.bse and pvalues/conf_int if available
            bse = getattr(model_output, 'bse', None)
            pvals = getattr(model_output, 'pvalues', None)
            try:
                ci = model_output.conf_int()
            except Exception:
                ci = None
            return bse, pvals, ci
        else:
            se = np.sqrt(np.diag(covmat))
            se = np.array(se)
            se_series = None
            try:
                import pandas as pd
                se_series = pd.Series(se, index=names)
            except Exception:
                # fallback: create dict
                se_series = dict(zip(names, se))
            return se_series, None, None

    se_series, pvals_fallback, ci_fallback = se_from_cov(cov)

    # Function to compute z, p, CI from beta and SE
    def stats_from_beta_se(beta, se):
        z = beta / se
        p = 2 * (1 - norm.cdf(abs(z)))
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se
        return float(beta), float(se), float(z), float(p), float(ci_lower), float(ci_upper)

    # find interaction term name containing both SizeAdv_c and FocalCloser
    interaction_name = None
    for n in names:
        if ('SizeAdv_c' in n) and ('FocalCloser' in n):
            interaction_name = n
            break

    # core variables expected
    def _get_name(target):
        if target in names:
            return target
        # fallback: find name containing target
        for n in names:
            if target in n:
                return n
        raise KeyError(f"Variable {target} not found in model parameters. Available: {names}")

    name_size = _get_name('SizeAdv_c')
    name_dist = _get_name('DistAdv_c')

    # Extract coefficients
    beta_size = float(params[name_size])
    beta_dist = float(params[name_dist])
    beta_inter = float(params[interaction_name]) if interaction_name is not None else 0.0

    # Get SEs
    if cov is not None:
        # se for individual coefficients
        se_size = float(np.sqrt(cov.loc[name_size, name_size]))
        se_dist = float(np.sqrt(cov.loc[name_dist, name_dist]))
        se_inter = float(np.sqrt(cov.loc[interaction_name, interaction_name])) if interaction_name is not None else None
    else:
        # fallback to model_output.bse and pvalues/conf_int
        se_size = float(model_output.bse[name_size])
        se_dist = float(model_output.bse[name_dist])
        se_inter = float(model_output.bse[interaction_name]) if interaction_name is not None else None

    # Stats for main SizeAdv_c effect when FocalCloser == 0 (this is the main term)
    beta0, se0, z0, p0, ci0_low, ci0_up = stats_from_beta_se(beta_size, se_size)

    # Stats for SizeAdv_c when FocalCloser == 1 (beta_size + beta_inter)
    if interaction_name is not None:
        beta1 = beta_size + beta_inter
        # variance = var(size) + var(inter) + 2*cov(size, inter)
        if cov is not None:
            var1 = cov.loc[name_size, name_size] + cov.loc[interaction_name, interaction_name] + 2 * cov.loc[name_size, interaction_name]
            se1 = float(np.sqrt(var1))
        else:
            # approximate by summing variances (conservative, ignores covariance)
            se1 = float(np.sqrt(se_size**2 + se_inter**2))
        beta1, se1, z1, p1, ci1_low, ci1_up = stats_from_beta_se(beta1, se1)
    else:
        # no interaction found; size effect same regardless of FocalCloser
        beta1, se1, z1, p1, ci1_low, ci1_up = beta0, se0, z0, p0, ci0_low, ci0_up

    # DistAdv stats
    beta_d, se_d, z_d, p_d, ci_d_low, ci_d_up = stats_from_beta_se(beta_dist, se_dist)

    # Interaction term stats
    if interaction_name is not None:
        beta_i = beta_inter
        se_i = se_inter
        beta_i, se_i, z_i, p_i, ci_i_low, ci_i_up = stats_from_beta_se(beta_i, se_i)
    else:
        beta_i = se_i = z_i = p_i = ci_i_low = ci_i_up = None

    # Odds ratios and CIs
    def or_and_ci(beta, se):
        or_ = exp(beta)
        ci_low = exp(beta - 1.96 * se)
        ci_up = exp(beta + 1.96 * se)
        return float(or_), float(ci_low), float(ci_up)

    or0, or0_low, or0_up = or_and_ci(beta0, se0)
    or1, or1_low, or1_up = or_and_ci(beta1, se1)
    ordist, ordist_low, ordist_up = or_and_ci(beta_d, se_d)
    if beta_i is not None:
        ori, ori_low, ori_up = or_and_ci(beta_i, se_i)
    else:
        ori = ori_low = ori_up = None

    # Simple significance verdicts (alpha = 0.05)
    sig_size_when_far = (p0 < 0.05)
    sig_size_when_close = (p1 < 0.05)
    sig_dist = (p_d < 0.05)
    sig_interaction = (p_i < 0.05) if p_i is not None else False

    # Build output object
    output = {
        'SizeAdv_c_when_FocalCloser_0': {
            'coef': beta0, 'se': se0, 'z': z0, 'p': p0,
            '95%_CI': [ci0_low, ci0_up],
            'OR': or0, 'OR_95%_CI': [or0_low, or0_up],
            'significant_at_0.05': sig_size_when_far
        },
        'SizeAdv_c_when_FocalCloser_1': {
            'coef': beta1, 'se': se1, 'z': z1, 'p': p1,
            '95%_CI': [ci1_low, ci1_up],
            'OR': or1, 'OR_95%_CI': [or1_low, or1_up],
            'significant_at_0.05': sig_size_when_close
        },
        'Interaction_term (SizeAdv_c:FocalCloser)': {
            'coef': beta_i, 'se': se_i, 'z': z_i, 'p': p_i,
            '95%_CI': [ci_i_low, ci_i_up] if beta_i is not None else None,
            'OR': ori, 'OR_95%_CI': [ori_low, ori_up] if ori is not None else None,
            'significant_at_0.05': sig_interaction
        },
        'DistAdv_c': {
            'coef': beta_d, 'se': se_d, 'z': z_d, 'p': p_d,
            '95%_CI': [ci_d_low, ci_d_up],
            'OR': ordist, 'OR_95%_CI': [ordist_low, ordist_up],
            'significant_at_0.05': sig_dist
        },
        # model reference
        'model_params_index': names
    }

    # Short textual interpretation
    parts = []
    # Size adv when focal not closer
    parts.append(
        f"Size advantage (SizeAdv_c) when focal group is NOT closer to home: coef={beta0:.3f}, p={p0:.3f}, "
        f"OR={or0:.3f} (95% CI [{or0_low:.3f}, {or0_up:.3f}]). "
        + ("Statistically significant." if sig_size_when_far else "Not statistically significant.")
    )
    # Size adv when focal closer
    parts.append(
        f"Size advantage when focal group IS closer to home: coef={beta1:.3f}, p={p1:.3f}, "
        f"OR={or1:.3f} (95% CI [{or1_low:.3f}, {or1_up:.3f}]). "
        + ("Statistically significant." if sig_size_when_close else "Not statistically significant.")
    )
    # Interaction
    if interaction_name is not None:
        parts.append(
            f"Interaction (SizeAdv_c x FocalCloser): coef={beta_i:.3f}, p={p_i:.3f}. "
            + ("Evidence that being closer to home changes the size advantage effect." if sig_interaction else "No strong evidence of moderation by being closer to home.")
        )
    else:
        parts.append("No interaction term found in the fitted model output.")

    # DistAdv
    parts.append(
        f"Location advantage (DistAdv_c): coef={beta_d:.3f}, p={p_d:.3f}, OR={ordist:.3f} "
        f"(95% CI [{ordist_low:.3f}, {ordist_up:.3f}]). "
        + ("Statistically significant." if sig_dist else "Not statistically significant.")
    )

    description = " ".join(parts)

    return {'object': output, 'description': description}