def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, confidence intervals, odds ratios, and
    marginal effects of relative group size at different location advantage
    values from a fitted statsmodels GLMResultsWrapper (logistic).
    
    Returns:
      {
        "object": {
          "params": {param_name: float, ...},
          "pvalues": {param_name: float, ...},
          "bse": {param_name: float, ...},
          "ci_95": {param_name: (ci_low, ci_high), ...},
          "odds_ratios": {param_name: OR, ...},
          "odds_ratio_ci_95": {param_name: (OR_low, OR_high), ...},
          "marginal_effects_of_size_at": {
             dist_value: {
               "coef": float,
               "se": float,
               "z": float,
               "p": float,
               "odds_ratio": float,
               "odds_ratio_ci_95": (low, high)
             }, ...
          }
        },
        "description": "Plain-language interpretation of the key results"
      }
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Basic extracts
    params = res.params.copy()            # pandas Series
    pvalues = res.pvalues.copy()
    bse = res.bse.copy()
    try:
        ci = res.conf_int()               # DataFrame with 2 columns
    except Exception:
        # fallback: compute approx CI using bse and normal quantile
        zcrit = stats.norm.ppf(0.975)
        ci = pd.DataFrame({
            0: params - zcrit * bse,
            1: params + zcrit * bse
        }, index=params.index)

    # Covariance matrix (robust cluster cov if model was fit with cov_type='cluster')
    cov = res.cov_params()
    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(cov, index=params.index, columns=params.index)

    # Odds ratios and CI
    or_dict = {}
    or_ci_dict = {}
    for name in params.index:
        or_dict[name] = float(np.exp(params.loc[name]))
        or_ci_dict[name] = (float(np.exp(ci.loc[name, 0])), float(np.exp(ci.loc[name, 1])))

    # Identify interaction term between size_diff_z and dist_diff_z (patsy uses ':' order unpredictable)
    inter_name = None
    for nm in params.index:
        if ':' in nm and 'size_diff_z' in nm and 'dist_diff_z' in nm:
            inter_name = nm
            break

    size_name = 'size_diff_z'
    dist_name = 'dist_diff_z'

    # Prepare marginal effects of size at dist = -1, 0, +1 (z-units)
    marginal_effects = {}
    if size_name in params.index:
        beta_size = float(params.loc[size_name])
        for dist_val in [-1.0, 0.0, 1.0]:
            if inter_name is not None and inter_name in params.index:
                beta_inter = float(params.loc[inter_name])
                coef = beta_size + beta_inter * dist_val
                # variance: var(size) + dist_val^2 * var(inter) + 2 * dist_val * cov(size,inter)
                var_size = cov.loc[size_name, size_name]
                var_inter = cov.loc[inter_name, inter_name]
                cov_si = cov.loc[size_name, inter_name]
                var_coef = var_size + (dist_val ** 2) * var_inter + 2.0 * dist_val * cov_si
            else:
                coef = beta_size
                var_coef = cov.loc[size_name, size_name]
            se_coef = float(np.sqrt(var_coef)) if var_coef >= 0 else float(np.nan)
            z_stat = float(coef / se_coef) if se_coef > 0 else float('nan')
            p_val = float(2.0 * stats.norm.sf(abs(z_stat))) if se_coef > 0 else float('nan')
            or_val = float(np.exp(coef))
            # CI for coef
            ci_low = coef - stats.norm.ppf(0.975) * se_coef
            ci_high = coef + stats.norm.ppf(0.975) * se_coef
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))
            marginal_effects[dist_val] = {
                "coef": float(coef),
                "se": float(se_coef),
                "z": float(z_stat),
                "p": float(p_val),
                "odds_ratio": or_val,
                "odds_ratio_ci_95": (or_ci_low, or_ci_high)
            }
    else:
        marginal_effects = {}

    # Pack coefficient-level info
    coeffs = {name: float(params.loc[name]) for name in params.index}
    pvals = {name: float(pvalues.loc[name]) for name in params.index}
    bses = {name: float(bse.loc[name]) for name in params.index}
    cis = {name: (float(ci.loc[name, 0]), float(ci.loc[name, 1])) for name in params.index}

    result_object = {
        "params": coeffs,
        "pvalues": pvals,
        "bse": bses,
        "ci_95": cis,
        "odds_ratios": or_dict,
        "odds_ratio_ci_95": or_ci_dict,
        "marginal_effects_of_size_at": marginal_effects
    }

    # Construct a concise description / interpretation
    def sig(p): return (p < 0.05) if (p is not None and not np.isnan(p)) else False
    desc_lines = []

    # Size main effect (note when interaction exists, main effect is effect at dist_diff_z = 0)
    if size_name in params.index:
        p_sz = pvalues.loc[size_name]
        coef_sz = params.loc[size_name]
        or_sz = or_dict[size_name]
        ci_sz = ci.loc[size_name].tolist()
        desc_lines.append(
            f"Relative group size (size_diff_z): coef = {coef_sz:.3f}, OR = {or_sz:.3f}, "
            f"95% CI for coef = ({ci_sz[0]:.3f}, {ci_sz[1]:.3f}), p = {p_sz:.3f}."
        )
        if sig(p_sz):
            desc_lines.append("This indicates a statistically significant association: larger focal groups have higher odds of winning (at mean location advantage).")
        else:
            desc_lines.append("This effect is not statistically significant at alpha=0.05 (at mean location advantage).")

    # Location main effect
    if dist_name in params.index:
        p_dt = pvalues.loc[dist_name]
        coef_dt = params.loc[dist_name]
        or_dt = or_dict[dist_name]
        ci_dt = ci.loc[dist_name].tolist()
        desc_lines.append(
            f"Location advantage (dist_diff_z): coef = {coef_dt:.3f}, OR = {or_dt:.3f}, "
            f"95% CI for coef = ({ci_dt[0]:.3f}, {ci_dt[1]:.3f}), p = {p_dt:.3f}."
        )
        if sig(p_dt):
            desc_lines.append("This indicates being relatively closer to the home-range center increases the odds of the focal group winning.")
        else:
            desc_lines.append("No statistically significant main effect of location advantage at alpha=0.05.")

    # Interaction
    if inter_name is not None:
        p_it = pvalues.loc[inter_name]
        coef_it = params.loc[inter_name]
        or_it = or_dict[inter_name]
        ci_it = ci.loc[inter_name].tolist()
        desc_lines.append(
            f"Interaction ({inter_name}): coef = {coef_it:.3f}, OR = {or_it:.3f}, "
            f"95% CI for coef = ({ci_it[0]:.3f}, {ci_it[1]:.3f}), p = {p_it:.3f}."
        )
        if sig(p_it):
            desc_lines.append("The interaction is statistically significant: the effect of relative group size on winning depends on location advantage.")
            # add marginal outcomes summary
            me_lines = []
            for dv in sorted(marginal_effects.keys()):
                me = marginal_effects[dv]
                me_lines.append(
                    f"At dist_diff_z = {dv:+.1f}: size coef = {me['coef']:.3f}, OR = {me['odds_ratio']:.3f}, p = {me['p']:.3f}"
                )
            desc_lines.append("Marginal effects of size at selected location values: " + "; ".join(me_lines) + ".")
        else:
            desc_lines.append("No evidence of a significant interaction: the effect of size does not appear to depend on location advantage at alpha=0.05.")
    else:
        desc_lines.append("No interaction term between size_diff_z and dist_diff_z was found in the model; the effect of size is constant across location in this specification.")

    # Controls: m_diff_z and FocalCloser if present
    for ctl in ['m_diff_z', 'FocalCloser']:
        if ctl in params.index:
            p_ctl = pvalues.loc[ctl]
            coef_ctl = params.loc[ctl]
            or_ctl = or_dict[ctl]
            desc_lines.append(
                f"Control {ctl}: coef = {coef_ctl:.3f}, OR = {or_ctl:.3f}, p = {p_ctl:.3f}."
            )

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}