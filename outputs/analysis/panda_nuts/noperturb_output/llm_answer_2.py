def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% CIs for the key predictors
    (age, sex_M, help_Y, age:help_Y, sex_M:help_Y) from a statsmodels MixedLM or OLS result
    object and provide a short interpretation about whether age, sex, and receiving help
    influence nut-cracking efficiency.

    Returns:
        dict with keys:
          - "object": dict containing model_type, results for each term, and random-effect variance (if present)
          - "description": human-readable summary interpretation (alpha = 0.05)
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Helper to get attributes robustly
    def safe_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Get parameter estimates
    try:
        params = safe_attr(res, 'params')
        bse = safe_attr(res, 'bse')
    except Exception as e:
        raise RuntimeError(f"Unable to read params/bse from model_output: {e}")

    # p-values: if not present, approximate using normal/t distribution depending on df_resid
    pvalues = safe_attr(res, 'pvalues', None)
    if pvalues is None and params is not None and bse is not None:
        # approximate z-test
        tvals = params / bse
        # If df_resid available, use t dist; otherwise normal
        df_resid = safe_attr(res, 'df_resid', None)
        if df_resid is not None and not np.isnan(df_resid):
            pvalues = 2 * stats.t.sf(np.abs(tvals), df_resid)
        else:
            pvalues = 2 * stats.norm.sf(np.abs(tvals))

    # confidence intervals
    try:
        conf = res.conf_int()
        # conf_int may be DataFrame or array-like; make into DataFrame indexed by param names
        if isinstance(conf, (list, tuple, np.ndarray)):
            conf = pd.DataFrame(conf, index=params.index, columns=['2.5%', '97.5%'])
        else:
            # statsmodels returns DataFrame with columns [0,1] sometimes
            conf = pd.DataFrame(conf)
            if conf.shape[1] >= 2:
                conf.columns = ['2.5%', '97.5%']
    except Exception:
        # fallback compute using normal approx
        z = stats.norm.ppf(0.975)
        conf = pd.DataFrame({
            '2.5%': params - z * bse,
            '97.5%': params + z * bse
        }, index=params.index)

    # Identify parameter names of interest (account for interaction name ordering)
    possible_terms = {
        'age': ['age'],
        'sex_M': ['sex_M'],
        'help_Y': ['help_Y'],
        'age:help_Y': ['age:help_Y', 'age:help_Y', 'help_Y:age'],
        'sex_M:help_Y': ['sex_M:help_Y', 'sex_M:help_Y', 'help_Y:sex_M']
    }

    found_terms = {}
    for key, variants in possible_terms.items():
        found = None
        for v in variants:
            if v in params.index:
                found = v
                break
        # also try patsy-style 'C(hammer)[T.x]' etc are not relevant
        found_terms[key] = found

    # Build results dictionary for requested terms
    terms_results = {}
    alpha = 0.05
    for logical_name, param_name in found_terms.items():
        if param_name is None:
            terms_results[logical_name] = None
            continue
        coef = float(params[param_name])
        se = float(bse[param_name]) if bse is not None and param_name in bse.index else None
        pval = float(pvalues[param_name]) if pvalues is not None and param_name in pvalues.index else None
        ci_low = float(conf.loc[param_name, '2.5%']) if param_name in conf.index else None
        ci_high = float(conf.loc[param_name, '97.5%']) if param_name in conf.index else None
        sig = None
        if pval is not None:
            sig = (pval < alpha)
        terms_results[logical_name] = {
            'param_name': param_name,
            'coef': coef,
            'se': se,
            'p_value': pval,
            '95%_CI': (ci_low, ci_high),
            'significant_at_0.05': sig
        }

    # Random effect variance if available (MixedLM)
    random_effects_info = None
    try:
        # MixedLMResults has cov_re or random_effects and scale, cov_re
        if hasattr(res, 'cov_re'):
            cov_re = res.cov_re
            # cov_re is numpy array or DataFrame; extract intercept variance if 1x1
            if np.shape(cov_re) == (1, 1):
                random_effects_info = {'intercept_variance': float(cov_re[0, 0])}
            else:
                random_effects_info = {'cov_re': cov_re}
        elif hasattr(res, 'random_effects'):
            # random_effects is a dict by group; return variance of the intercepts if present
            re = res.random_effects
            # collect intercepts if present
            intercepts = []
            for g, vals in re.items():
                if isinstance(vals, dict) and 'Group' in vals:
                    # unlikely format; skip
                    pass
                else:
                    try:
                        # vals could be array-like: first element is intercept
                        intercepts.append(float(np.asarray(vals).ravel()[0]))
                    except Exception:
                        pass
            if intercepts:
                random_effects_info = {'intercept_sd': float(np.std(intercepts, ddof=1))}
    except Exception:
        random_effects_info = None

    # Compose human-readable description
    sig_terms = []
    nonsig_terms = []
    missing_terms = []
    for t, info in terms_results.items():
        if info is None:
            missing_terms.append(t)
            continue
        if info['significant_at_0.05'] is True:
            sig_terms.append((t, info))
        else:
            nonsig_terms.append((t, info))

    desc_lines = []
    desc_lines.append(f"Model type: {type(res).__name__}. Alpha for significance: {alpha}.")
    if missing_terms:
        desc_lines.append("The model does not include the following requested terms: " + ", ".join(missing_terms) + ".")
    # Summarize significant terms
    if sig_terms:
        for t, info in sig_terms:
            direction = "positive" if info['coef'] > 0 else "negative"
            desc_lines.append(
                f"Significant effect: {t} (parameter '{info['param_name']}') — coef={info['coef']:.3f} "
                f"({direction}), p={info['p_value']:.3g}, 95% CI=[{info['95%_CI'][0]:.3f}, {info['95%_CI'][1]:.3f}]."
            )
    else:
        desc_lines.append("No focal predictors reached significance at alpha=0.05.")

    # Provide brief interpretation about interactions if present
    # If interaction age:help_Y is significant, interpret direction relative to main effect
    inter_age = terms_results.get('age:help_Y')
    if inter_age is not None:
        if inter_age['significant_at_0.05']:
            desc_lines.append(
                "The age x help interaction is significant, meaning the effect of age on log-efficiency differs "
                "between sessions with and without help (interaction coef = "
                f"{inter_age['coef']:.3f}, p={inter_age['p_value']:.3g})."
            )
        else:
            desc_lines.append(
                "The age x help interaction is not significant, implying no strong evidence that the relationship "
                "between age and log-efficiency differs when help is received."
            )
    inter_sex = terms_results.get('sex_M:help_Y')
    if inter_sex is not None:
        if inter_sex['significant_at_0.05']:
            desc_lines.append(
                "The sex x help interaction is significant, meaning the difference between males and females in "
                "log-efficiency depends on whether help was received "
                f"(interaction coef = {inter_sex['coef']:.3f}, p={inter_sex['p_value']:.3g})."
            )
        else:
            desc_lines.append(
                "The sex x help interaction is not significant, implying no strong evidence that the male/female "
                "difference in log-efficiency changes when help is received."
            )

    # Add note on random effects if available
    if random_effects_info:
        desc_lines.append(f"Random-effects info (approx.): {random_effects_info}.")

    desc_lines.append(
        "Notes: Coefficients are on the log(1 + nuts_per_minute) scale. A positive coefficient means higher "
        "log-efficiency (more nuts per minute), negative means lower. Interactions modify main effects when present."
    )

    description = " ".join(desc_lines)

    # Return a structured object and interpretation
    output = {
        'model_type': type(res).__name__,
        'terms': terms_results,
        'random_effects': random_effects_info,
        'alpha': alpha
    }

    return {
        "object": output,
        "description": description
    }