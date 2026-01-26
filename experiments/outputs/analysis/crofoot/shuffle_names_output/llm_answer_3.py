def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, confidence intervals, and compute location-specific
    marginal effects of relative group size (RelSizeDiff_c) from a clustered
    statsmodels results object (the output of get_robustcov_results).

    Returns:
      {
        "object": {
            "params": {param_name: coef, ...},
            "pvalues": {param_name: pval, ...},
            "conf_int": {param_name: [ci_low, ci_high], ...},
            "marginal_effects_by_location": {
                location_name: {
                    "interaction_term": interaction_name_or_None,
                    "interaction_coef": ...,
                    "interaction_p": ...,
                    "marginal_effect": coef_main + coef_interaction,
                    "marginal_se": ...,
                    "marginal_p": ...,
                    "marginal_ci": [ci_low, ci_high]
                }, ...
            },
            "summary_flags": {
                "RelSizeDiff_c_significant": True/False/None,
                "any_interaction_significant": True/False/None
            }
        },
        "description": "Short interpretation string..."
      }
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Prepare containers
    out = {}
    try:
        params = res.params.copy()
        pvalues = res.pvalues.copy()
        ci = res.conf_int().copy()
        cov = res.cov_params().copy()
    except Exception as e:
        # If the object doesn't have the expected attributes, return that error
        return {
            "object": None,
            "description": f"Provided model_output does not expose expected attributes (params, pvalues, conf_int, cov_params). Error: {e}"
        }

    # Convert basic results to plain dicts
    out['params'] = params.to_dict()
    out['pvalues'] = pvalues.to_dict()
    # conf_int returns DataFrame with two columns 0 (low), 1 (high)
    conf_int_dict = {name: [float(ci.loc[name, 0]), float(ci.loc[name, 1])] for name in ci.index}
    out['conf_int'] = conf_int_dict

    # Identify main relative size term and interaction terms
    main_name = 'RelSizeDiff_c'
    interaction_suffix = ':RelSizeDiff_c'
    interactions = [n for n in params.index if n.endswith(interaction_suffix)]
    # Also consider the reversed order if colon naming differs (unlikely given code, but safe)
    if not interactions:
        interactions = [n for n in params.index if n.startswith('RelSizeDiff_c:')]

    marginal_effects_by_location = {}

    # Gather main effect if present
    main_present = main_name in params.index
    coef_main = float(params[main_name]) if main_present else None
    p_main = float(pvalues[main_name]) if main_present else None

    # For each interaction, compute marginal effect = main + interaction.
    # Compute standard error of the linear combination using covariance matrix.
    for inter in interactions:
        # Derive location name (the prefix before the colon)
        if inter.endswith(interaction_suffix):
            loc = inter[:-len(interaction_suffix)]
        elif inter.startswith('RelSizeDiff_c:'):
            loc = inter[len('RelSizeDiff_c:'):]
        else:
            loc = inter.replace(':RelSizeDiff_c', '')

        coef_inter = float(params[inter])
        p_inter = float(pvalues[inter])
        interaction_term = inter

        if main_present:
            # var(m) = var(main) + var(inter) + 2*cov(main,inter)
            var_main = float(cov.loc[main_name, main_name])
            var_inter = float(cov.loc[inter, inter])
            cov_main_inter = float(cov.loc[main_name, inter])
            var_margin = var_main + var_inter + 2.0 * cov_main_inter
            # numerical safeguard
            var_margin = max(var_margin, 0.0)
            se_margin = float(np.sqrt(var_margin))
            margin = coef_main + coef_inter
            # compute z and p
            if se_margin > 0:
                z = margin / se_margin
                p_margin = float(2 * (1 - stats.norm.cdf(abs(z))))
            else:
                z = None
                p_margin = None
            # CI
            zcrit = stats.norm.ppf(0.975)
            ci_low = margin - zcrit * se_margin
            ci_high = margin + zcrit * se_margin
        else:
            # If main absent, the interaction is the effect for that location only (depending on coding)
            margin = coef_inter
            se_margin = float(np.sqrt(float(cov.loc[inter, inter])))
            if se_margin > 0:
                z = margin / se_margin
                p_margin = float(2 * (1 - stats.norm.cdf(abs(z))))
            else:
                z = None
                p_margin = None
            zcrit = stats.norm.ppf(0.975)
            ci_low = margin - zcrit * se_margin
            ci_high = margin + zcrit * se_margin

        marginal_effects_by_location[loc] = {
            "interaction_term": interaction_term,
            "interaction_coef": coef_inter,
            "interaction_p": p_inter,
            "marginal_effect": float(margin),
            "marginal_se": float(se_margin),
            "marginal_z": float(z) if z is not None else None,
            "marginal_p": p_margin,
            "marginal_ci": [float(ci_low), float(ci_high)]
        }

    # If there are location dummy terms themselves, include them in summary (optional)
    location_dummies = [n for n in params.index if n.startswith('Location_')]
    # Summarize significance
    relsize_significant = None
    if main_present:
        relsize_significant = (p_main < 0.05)

    any_inter_significant = None
    if interactions:
        any_inter_significant = any(float(pvalues.get(inter, 1.0)) < 0.05 for inter in interactions)

    out['marginal_effects_by_location'] = marginal_effects_by_location
    out['location_dummies'] = {n: float(params[n]) for n in location_dummies} if location_dummies else {}
    out['summary_flags'] = {
        "RelSizeDiff_c_significant": relsize_significant,
        "any_interaction_significant": any_inter_significant
    }

    # Compose a concise human-readable description
    desc_lines = []
    if main_present:
        desc_lines.append(
            f"Main effect RelSizeDiff_c: coef={coef_main:.3f}, p={p_main:.3g}, 95%CI=[{conf_int_dict.get(main_name, ['NA','NA'])[0]:.3f}, {conf_int_dict.get(main_name, ['NA','NA'])[1]:.3f}]."
        )
        if relsize_significant:
            desc_lines.append("Overall, relative group size (RelSizeDiff) has a statistically significant association with winning (p < 0.05).")
        else:
            desc_lines.append("Overall, relative group size (RelSizeDiff) is not statistically significant (p >= 0.05).")
    else:
        desc_lines.append("No main RelSizeDiff_c term found in the model results; interpret interactions if present.")

    if marginal_effects_by_location:
        for loc, info in marginal_effects_by_location.items():
            sig_str = "significant" if (info["marginal_p"] is not None and info["marginal_p"] < 0.05) else "not significant"
            desc_lines.append(
                f"Location '{loc}': marginal effect of RelSizeDiff = {info['marginal_effect']:.3f} (SE={info['marginal_se']:.3f}, p={info['marginal_p']:.3g}) -> {sig_str}."
            )
    else:
        desc_lines.append("No location-specific interactions with RelSizeDiff_c were found in the model.")

    if any_inter_significant:
        desc_lines.append("There is evidence that the effect of relative group size differs by contest location (significant interaction(s)).")
    else:
        desc_lines.append("No evidence that the effect of relative group size differs by contest location (interaction terms not significant).")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}