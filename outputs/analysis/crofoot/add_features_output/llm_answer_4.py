def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, confidence intervals, odds ratios,
    and marginal effect of relative group size inside vs outside focal territory
    from a fitted statsmodels results object (including cluster-robust results).

    Returns a dictionary with:
      - "object": dict containing numeric outputs (coefficients, SEs, p-values, ORs, CIs,
                    marginal effects and their SE/p-values/ORs/CIs when computable)
      - "description": a short plain-language summary of what those statistics mean
                       for how relative group size and contest location influence
                       the probability the focal group wins.
    """
    import numpy as np
    import pandas as pd
    from math import exp
    from scipy import stats

    res = model_output

    # Try to get parameter names and tables in a robust way
    try:
        params = res.params.copy()
    except Exception:
        # If model_output wraps results differently, try to access .results or .model
        if hasattr(res, 'results'):
            res = res.results
            params = res.params.copy()
        else:
            raise ValueError("Provided model_output does not appear to be a statsmodels results object with .params")

    # Covariance matrix for computing SE of linear combinations (may be robust)
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        # Fallback: try attribute name variations
        if hasattr(res, 'cov_params_default'):
            cov = res.cov_params_default()
        else:
            cov = None

    # Helper to find parameter name by matching substrings (handles factor-coding naming)
    def find_param(*parts):
        parts = [str(p) for p in parts]
        for name in params.index:
            if all(p in name for p in parts):
                return name
        return None

    # Identify parameter names
    rel_name = find_param('rel_size_log_z') or find_param('rel_size_log') or find_param('rel_size')
    in_name = find_param('in_focal_territory')  # could be 'in_focal_territory' or 'in_focal_territory[T.1]'
    inter_name = None
    if rel_name and in_name:
        # Interaction often uses ':' between variable names
        # Try exact pattern
        inter_name = find_param(f"{rel_name}:{in_name}") or find_param(f"{in_name}:{rel_name}")
    # If not found by combination, try any parameter containing both substrings
    if inter_name is None and rel_name is not None and in_name is not None:
        for name in params.index:
            if rel_name in name and in_name in name and name != rel_name and name != in_name:
                inter_name = name
                break

    dist_name = find_param('dist_adv_z') or find_param('dist_adv')
    m_name = find_param('m_diff_z') or find_param('m_diff')
    f_name = find_param('f_diff_z') or find_param('f_diff')
    intercept_name = find_param('Intercept') or find_param('const') or find_param('Intercept') or ('Intercept' if 'Intercept' in params.index else None)
    if intercept_name is None:
        # fallback to first param if single intercept exists
        intercept_name = params.index[0] if len(params.index) > 0 else None

    # Prepare output container
    out = {
        'coefficients': {},
        'odds_ratios': {},
        'conf_int': {},
        'pvalues': {},
        'marginal_effects': {}
    }

    # Utility to safely extract param stats
    def extract_param_stats(name):
        if name is None or name not in params.index:
            return None
        coef = float(params.loc[name])
        # standard error
        try:
            se = float(res.bse.loc[name])
        except Exception:
            # try diagonal of cov matrix
            if cov is not None and name in cov.index:
                se = float(np.sqrt(cov.loc[name, name]))
            else:
                se = None
        # pvalue
        pval = None
        try:
            pval = float(res.pvalues.loc[name])
        except Exception:
            # compute z from coef/se
            if se is not None and se > 0:
                z = coef / se
                pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        # conf int
        ci = None
        try:
            ci_df = res.conf_int()
            if name in ci_df.index:
                ci = (float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1]))
        except Exception:
            if se is not None:
                # approximate 95% CI on link scale
                ci = (coef - 1.96 * se, coef + 1.96 * se)
        # odds ratio and its CI (by exponentiating)
        or_val = float(np.exp(coef))
        or_ci = None
        if ci is not None:
            or_ci = (float(np.exp(ci[0])), float(np.exp(ci[1])))
        return {'name': name, 'coef': coef, 'se': se, 'pvalue': pval, 'ci': ci, 'or': or_val, 'or_ci': or_ci}

    # Extract for key predictors
    for name_label, name in [('rel_size', rel_name), ('in_focal', in_name), ('interaction', inter_name),
                             ('dist_adv', dist_name), ('m_diff', m_name), ('f_diff', f_name),
                             ('intercept', intercept_name)]:
        statsd = extract_param_stats(name)
        out['coefficients'][name_label] = statsd
        if statsd is not None:
            out['pvalues'][name_label] = statsd['pvalue']
            out['odds_ratios'][name_label] = {'or': statsd['or'], 'or_ci': statsd['or_ci']}
            out['conf_int'][name_label] = statsd['ci']

    # Compute marginal effect of rel_size when in_focal_territory == 0 (baseline) and == 1
    # Effect on log-odds: beta_rel (+ beta_interaction if in_focal_territory==1)
    if rel_name is not None:
        beta_rel = out['coefficients']['rel_size']['coef'] if out['coefficients']['rel_size'] is not None else None
        beta_inter = out['coefficients']['interaction']['coef'] if out['coefficients']['interaction'] is not None else 0.0
        # When in_focal_territory == 0
        eff_out = None
        se_out = None
        p_out = None
        if beta_rel is not None:
            eff_out = float(beta_rel)
            # SE for baseline is simply se(rel)
            se_out = out['coefficients']['rel_size']['se']
            if se_out is not None and se_out > 0:
                z = eff_out / se_out
                p_out = float(2 * (1 - stats.norm.cdf(abs(z))))
        # When in_focal_territory == 1
        eff_in = None
        se_in = None
        p_in = None
        if beta_rel is not None:
            eff_in = float(beta_rel + (beta_inter if beta_inter is not None else 0.0))
            # SE for sum = sqrt(var(rel)+var(inter)+2*cov(rel,inter))
            if cov is not None and rel_name in cov.index and inter_name in cov.index:
                var_rel = float(cov.loc[rel_name, rel_name])
                var_inter = float(cov.loc[inter_name, inter_name])
                cov_rel_inter = float(cov.loc[rel_name, inter_name])
                se_in = float(np.sqrt(var_rel + var_inter + 2 * cov_rel_inter))
                if se_in > 0:
                    z = eff_in / se_in
                    p_in = float(2 * (1 - stats.norm.cdf(abs(z))))
            else:
                # try using bse values alone (conservative, assumes independence)
                se_rel = out['coefficients']['rel_size']['se']
                se_inter = out['coefficients']['interaction']['se']
                if se_rel is not None and se_inter is not None:
                    se_in = float(np.sqrt(se_rel ** 2 + se_inter ** 2))
                    z = eff_in / se_in if se_in > 0 else None
                    p_in = float(2 * (1 - stats.norm.cdf(abs(z)))) if z is not None else None
        # Convert to odds ratios
        or_out = float(np.exp(eff_out)) if eff_out is not None else None
        or_in = float(np.exp(eff_in)) if eff_in is not None else None
        or_out_ci = None
        or_in_ci = None
        if se_out is not None:
            ci_out = (eff_out - 1.96 * se_out, eff_out + 1.96 * se_out)
            or_out_ci = (float(np.exp(ci_out[0])), float(np.exp(ci_out[1])))
        if se_in is not None:
            ci_in = (eff_in - 1.96 * se_in, eff_in + 1.96 * se_in)
            or_in_ci = (float(np.exp(ci_in[0])), float(np.exp(ci_in[1])))

        out['marginal_effects']['rel_size_when_not_in_focal_territory'] = {
            'log_odds_coef': eff_out, 'se': se_out, 'pvalue': p_out,
            'odds_ratio': or_out, 'or_ci': or_out_ci
        }
        out['marginal_effects']['rel_size_when_in_focal_territory'] = {
            'log_odds_coef': eff_in, 'se': se_in, 'pvalue': p_in,
            'odds_ratio': or_in, 'or_ci': or_in_ci
        }

    # Build a concise description string interpreting the main results
    desc_parts = []
    # Interpret relative size baseline effect
    rel_stats = out['coefficients']['rel_size']
    if rel_stats is not None:
        p = rel_stats['pvalue']
        coef = rel_stats['coef']
        orv = out['odds_ratios']['rel_size']['or']
        desc_parts.append(
            f"Relative group size (log ratio) has coefficient {coef:.3f} (OR={orv:.3f})."
            + (f" p={p:.3f}." if p is not None else "")
        )
    else:
        desc_parts.append("Relative group size effect not found in model output.")

    # Interpret interaction
    inter_stats = out['coefficients']['interaction']
    if inter_stats is not None:
        p_int = inter_stats['pvalue']
        coef_int = inter_stats['coef']
        or_int = out['odds_ratios']['interaction']['or']
        desc_parts.append(
            f"Interaction (rel_size × in_focal_territory) coef={coef_int:.3f} (OR={or_int:.3f})."
            + (f" p={p_int:.3f}." if p_int is not None else "")
            + " A significant positive interaction would mean the size advantage is larger when contests occur in the focal group's territory."
        )
    else:
        desc_parts.append("No interaction term found (or different naming); cannot evaluate moderation by contest location.")

    # Interpret distance advantage
    dist_stats = out['coefficients']['dist_adv']
    if dist_stats is not None:
        p = dist_stats['pvalue']
        coef = dist_stats['coef']
        orv = out['odds_ratios']['dist_adv']['or']
        desc_parts.append(
            f"Location advantage (dist_adv_z) coef={coef:.3f} (OR={orv:.3f})."
            + (f" p={p:.3f}." if p is not None else "")
            + " Positive values mean contests nearer the focal group's center favor the focal group."
        )

    # Summarize marginal effects if available
    me_in = out['marginal_effects'].get('rel_size_when_in_focal_territory')
    me_out = out['marginal_effects'].get('rel_size_when_not_in_focal_territory')
    if me_out is not None and me_in is not None:
        desc_parts.append(
            f"When contests are outside the focal territory, a one-unit increase in rel_size_log_z "
            f"changes odds by a factor of {me_out['odds_ratio']:.3f} "
            + (f"(95% CI [{me_out['or_ci'][0]:.3f}, {me_out['or_ci'][1]:.3f}]), p={me_out['pvalue']:.3f})."
               if me_out['or_ci'] is not None else ".")
        )
        desc_parts.append(
            f"When contests are inside the focal territory, a one-unit increase in rel_size_log_z "
            f"changes odds by a factor of {me_in['odds_ratio']:.3f} "
            + (f"(95% CI [{me_in['or_ci'][0]:.3f}, {me_in['or_ci'][1]:.3f}]), p={me_in['pvalue']:.3f})."
               if me_in['or_ci'] is not None else ".")
        )
    # Final assembled description
    description = " ".join(desc_parts)

    return {'object': out, 'description': description}