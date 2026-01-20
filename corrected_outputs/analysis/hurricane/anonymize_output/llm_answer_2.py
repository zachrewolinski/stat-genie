def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and percent-change interpretations
    for the key predictors (MasFem_z and FemaleName) from the primary (and optionally robustness)
    statsmodels results objects contained in model_output.

    Returns a dictionary with keys:
      - "object": dict with extracted numeric statistics for primary and robustness models
      - "description": brief human-readable interpretation about whether the results support
                       the hypothesis that more feminine hurricane names are associated with
                       higher fatalities (i.e., fewer precautionary measures).
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing at least 'primary_model'.")

    primary = model_output.get('primary_model', None)
    if primary is None:
        raise ValueError("model_output does not contain 'primary_model' or it is None.")

    def extract_from_result(res, varnames):
        out = {}
        # pull confidence intervals once
        try:
            ci_df = res.conf_int(alpha=0.05)
        except Exception:
            ci_df = None
        for v in varnames:
            if v in res.params.index:
                beta = float(res.params[v])
                se = float(res.bse[v]) if hasattr(res, 'bse') and v in res.bse.index else None
                p = float(res.pvalues[v]) if hasattr(res, 'pvalues') and v in res.pvalues.index else None
                if ci_df is not None and v in ci_df.index:
                    ci_lower = float(ci_df.loc[v, 0])
                    ci_upper = float(ci_df.loc[v, 1])
                else:
                    ci_lower = ci_upper = None
                # Interpret effect on (Deaths + 1): percent change = (exp(beta) - 1) * 100
                try:
                    pct_change = (np.exp(beta) - 1.0) * 100.0
                    pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0 if ci_lower is not None else None
                    pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0 if ci_upper is not None else None
                except Exception:
                    pct_change = pct_ci_lower = pct_ci_upper = None

                out[v] = {
                    'coef': beta,
                    'se': se,
                    'p_value': p,
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper,
                    'pct_change_on_DeathsPlus1': pct_change,
                    'pct_change_ci_lower': pct_ci_lower,
                    'pct_change_ci_upper': pct_ci_upper,
                    'significant_at_0.05': (p is not None and p < 0.05)
                }
            else:
                out[v] = None
        # add sample size
        try:
            out['_nobs'] = int(res.nobs)
        except Exception:
            out['_nobs'] = None
        return out

    vars_of_interest = ['MasFem_z', 'FemaleName']
    primary_stats = extract_from_result(primary, vars_of_interest)

    robustness_res = model_output.get('robustness_model', None)
    robustness_stats = None
    if robustness_res is not None:
        robustness_stats = extract_from_result(robustness_res, vars_of_interest)

    # Build a concise interpretation for the hypothesis
    interpret_lines = []
    # Primary model interpretation
    m = primary_stats.get('MasFem_z')
    f = primary_stats.get('FemaleName')
    if m is None:
        interpret_lines.append("MasFem_z not present in primary model results.")
    else:
        coef = m['coef']
        p = m['p_value']
        pct = m['pct_change_on_DeathsPlus1']
        sig = m['significant_at_0.05']
        ci_l = m['ci_95_lower']; ci_u = m['ci_95_upper']
        interpret_lines.append(
            "Primary model (outcome = LogDeaths): MasFem_z coef = {coef:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], p = {p:.4g}."
            .format(coef=coef, ci_l=(ci_l if ci_l is not None else float('nan')),
                    ci_u=(ci_u if ci_u is not None else float('nan')), p=(p if p is not None else float('nan')))
        )
        if pct is not None:
            interpret_lines.append(
                "  Interpreted on (Deaths+1): a 1 SD increase in name femininity is associated with a {pct:.2f}% change "
                "in (Deaths+1) (95% CI [{pct_l:.2f}%, {pct_u:.2f}%])."
                .format(pct=pct,
                        pct_l=(m['pct_change_ci_lower'] if m['pct_change_ci_lower'] is not None else float('nan')),
                        pct_u=(m['pct_change_ci_upper'] if m['pct_change_ci_upper'] is not None else float('nan')))
            )
        # Conclusion regarding hypothesis:
        if sig and coef > 0:
            interpret_lines.append("  This result is statistically significant and in the expected direction -> supports the hypothesis.")
        elif sig and coef < 0:
            interpret_lines.append("  This result is statistically significant but in the opposite direction -> does NOT support the hypothesis.")
        else:
            interpret_lines.append("  The effect is not statistically significant at alpha=0.05 -> no strong evidence for the hypothesis in the primary model.")

    # FemaleName (binary) interpretation
    if f is None:
        interpret_lines.append("FemaleName not present in primary model results.")
    else:
        coef = f['coef']; p = f['p_value']; pct = f['pct_change_on_DeathsPlus1']
        ci_l = f['ci_95_lower']; ci_u = f['ci_95_upper']; sig = f['significant_at_0.05']
        interpret_lines.append(
            "Primary model: FemaleName (binary) coef = {coef:.4f}, 95% CI [{ci_l:.4f}, {ci_u:.4f}], p = {p:.4g}."
            .format(coef=coef, ci_l=(ci_l if ci_l is not None else float('nan')),
                    ci_u=(ci_u if ci_u is not None else float('nan')), p=(p if p is not None else float('nan')))
        )
        if pct is not None:
            interpret_lines.append(
                "  Interpreted on (Deaths+1): having a female grammatical name vs male is associated with a {pct:.2f}% change "
                "in (Deaths+1) (95% CI [{pct_l:.2f}%, {pct_u:.2f}%])."
                .format(pct=pct,
                        pct_l=(f['pct_change_ci_lower'] if f['pct_change_ci_lower'] is not None else float('nan')),
                        pct_u=(f['pct_change_ci_upper'] if f['pct_change_ci_upper'] is not None else float('nan')))
            )
        if sig and coef > 0:
            interpret_lines.append("  Statistically significant and in the expected direction -> supports hypothesis.")
        elif sig and coef < 0:
            interpret_lines.append("  Statistically significant but opposite direction -> does NOT support hypothesis.")
        else:
            interpret_lines.append("  Not statistically significant at alpha=0.05 -> no strong evidence from the binary name indicator.")

    # Robustness model brief note
    if robustness_stats is not None:
        interpret_lines.append("Robustness model (outcome = damage proxy): statistics extracted; examine MasFem_z and FemaleName similarly for consistency.")
    else:
        interpret_lines.append("No robustness model available or it was not included in model_output.")

    description = " ".join(interpret_lines)

    return {
        "object": {
            "primary_model": primary_stats,
            "robustness_model": robustness_stats
        },
        "description": description
    }