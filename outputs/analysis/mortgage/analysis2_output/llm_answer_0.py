def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of the 'Female' indicator on mortgage approval
    from the model output produced by the provided `model` function.

    Returns:
      {
        "object": {
           "coef_log_odds": float or None,
           "coef_se": float or None,
           "coef_pvalue": float or None,
           "coef_ci_lower": float or None,
           "coef_ci_upper": float or None,
           "marginal_effect": float or None,         # in probability points (not percent)
           "marginal_effect_se": float or None,
           "marginal_effect_pvalue": float or None,
           "marginal_effect_ci_lower": float or None,
           "marginal_effect_ci_upper": float or None
        },
        "description": str
      }
    """
    res = model_output.get('model_results_robust')
    margeff = model_output.get('marginal_effects')

    out = {
        "coef_log_odds": None,
        "coef_se": None,
        "coef_pvalue": None,
        "coef_ci_lower": None,
        "coef_ci_upper": None,
        "marginal_effect": None,
        "marginal_effect_se": None,
        "marginal_effect_pvalue": None,
        "marginal_effect_ci_lower": None,
        "marginal_effect_ci_upper": None
    }

    # Helper for safe extraction from series/dataframes
    def safe_get(series_or_df, key, default=None):
        try:
            return series_or_df[key]
        except Exception:
            try:
                return series_or_df.loc[key]
            except Exception:
                return default

    # 1) Extract coefficient (log-odds) and robust SE/p-value/conf-int from robust results
    if res is not None:
        try:
            # params, bse, pvalues, conf_int should be available on the results wrapper
            if 'Female' in getattr(res, 'params').index:
                out['coef_log_odds'] = float(res.params['Female'])
                out['coef_se'] = float(res.bse['Female']) if hasattr(res, 'bse') else None
                out['coef_pvalue'] = float(res.pvalues['Female']) if hasattr(res, 'pvalues') else None
                try:
                    ci = res.conf_int().loc['Female']
                    out['coef_ci_lower'] = float(ci[0])
                    out['coef_ci_upper'] = float(ci[1])
                except Exception:
                    # conf_int may accept alpha or be named differently; try fallback
                    try:
                        ci = res.conf_int()
                        if 'Female' in ci.index:
                            out['coef_ci_lower'] = float(ci.loc['Female'].iloc[0])
                            out['coef_ci_upper'] = float(ci.loc['Female'].iloc[1])
                    except Exception:
                        pass
            else:
                # If 'Female' not found exactly, try fuzzy match
                names = list(res.params.index)
                matches = [n for n in names if 'Female' in str(n)]
                if matches:
                    name = matches[0]
                    out['coef_log_odds'] = float(res.params[name])
                    out['coef_se'] = float(res.bse[name]) if hasattr(res, 'bse') else None
                    out['coef_pvalue'] = float(res.pvalues[name]) if hasattr(res, 'pvalues') else None
                    try:
                        ci = res.conf_int().loc[name]
                        out['coef_ci_lower'] = float(ci[0])
                        out['coef_ci_upper'] = float(ci[1])
                    except Exception:
                        pass
        except Exception:
            # leave coef fields as None on failure
            pass

    # 2) Extract average marginal effect (AME) for Female from margeff results (probability scale)
    if margeff is not None:
        try:
            # Preferred: use summary_frame() which often returns a DataFrame indexed by variable names
            sf = None
            try:
                sf = margeff.summary_frame()
            except Exception:
                # some statsmodels versions expose summary() but not summary_frame; try attributes
                sf = None

            row = None
            if isinstance(sf, (dict,)) is False and sf is not None:
                # sf typically is a DataFrame - locate the Female row
                if 'Female' in sf.index:
                    row = sf.loc['Female']
                else:
                    matches = [i for i in sf.index if 'Female' in str(i)]
                    if matches:
                        row = sf.loc[matches[0]]

                if row is not None:
                    # assume columns: [marginal effect, std err, z, p, ci_lower, ci_upper] in that order
                    try:
                        out['marginal_effect'] = float(row.iloc[0])
                    except Exception:
                        out['marginal_effect'] = None
                    try:
                        out['marginal_effect_se'] = float(row.iloc[1]) if len(row) > 1 else None
                    except Exception:
                        out['marginal_effect_se'] = None
                    try:
                        out['marginal_effect_pvalue'] = float(row.iloc[3]) if len(row) > 3 else None
                    except Exception:
                        out['marginal_effect_pvalue'] = None
                    try:
                        out['marginal_effect_ci_lower'] = float(row.iloc[4]) if len(row) > 4 else None
                        out['marginal_effect_ci_upper'] = float(row.iloc[5]) if len(row) > 5 else None
                    except Exception:
                        pass

            # Fallback: try direct attributes on margeff object (margeff, margeff_se, pvalues)
            if out['marginal_effect'] is None:
                try:
                    # margeff.margeff is array aligned with exog names; need index of 'Female'
                    names = None
                    try:
                        names = list(margeff.model.exog_names)
                    except Exception:
                        try:
                            names = list(margeff.margeff_names)
                        except Exception:
                            names = None

                    idx = None
                    if names:
                        if 'Female' in names:
                            idx = names.index('Female')
                        else:
                            matches = [i for i, n in enumerate(names) if 'Female' in str(n)]
                            if matches:
                                idx = matches[0]

                    if hasattr(margeff, 'margeff') and idx is not None:
                        out['marginal_effect'] = float(margeff.margeff[idx])
                    if hasattr(margeff, 'margeff_se') and idx is not None:
                        out['marginal_effect_se'] = float(margeff.margeff_se[idx])
                    if hasattr(margeff, 'pvalues') and idx is not None:
                        out['marginal_effect_pvalue'] = float(margeff.pvalues[idx])
                    # conf int fallback if available
                    try:
                        if hasattr(margeff, 'conf_int'):
                            ci = margeff.conf_int()
                            if idx is not None and len(ci) > idx:
                                out['marginal_effect_ci_lower'] = float(ci[idx, 0])
                                out['marginal_effect_ci_upper'] = float(ci[idx, 1])
                    except Exception:
                        pass
                except Exception:
                    pass

        except Exception:
            pass

    # 3) Build human-readable description/interpretation
    desc_parts = []
    if out['coef_log_odds'] is not None:
        desc_parts.append(
            f"Log-odds coefficient for Female = {out['coef_log_odds']:.4f} "
            f"(SE = {out['coef_se']:.4f}, p = {out['coef_pvalue']:.4f})"
            if out['coef_se'] is not None and out['coef_pvalue'] is not None
            else f"Log-odds coefficient for Female = {out['coef_log_odds']:.4f}"
        )
        if out['coef_ci_lower'] is not None and out['coef_ci_upper'] is not None:
            desc_parts.append(
                f"95% CI for log-odds: [{out['coef_ci_lower']:.4f}, {out['coef_ci_upper']:.4f}]"
            )

    if out['marginal_effect'] is not None:
        # Convert to percentage points for readability
        me_pct = out['marginal_effect'] * 100
        se_pct = out['marginal_effect_se'] * 100 if out['marginal_effect_se'] is not None else None
        pval = out['marginal_effect_pvalue']
        desc_me = f"Average marginal effect of Female on approval = {me_pct:.2f} percentage points"
        if se_pct is not None and pval is not None:
            desc_me += f" (SE = {se_pct:.2f} pp, p = {pval:.4f})"
        elif se_pct is not None:
            desc_me += f" (SE = {se_pct:.2f} pp)"
        if out['marginal_effect_ci_lower'] is not None and out['marginal_effect_ci_upper'] is not None:
            ci_low_pct = out['marginal_effect_ci_lower'] * 100
            ci_high_pct = out['marginal_effect_ci_upper'] * 100
            desc_me += f", 95% CI = [{ci_low_pct:.2f} pp, {ci_high_pct:.2f} pp]"
        desc_parts.append(desc_me)

        # Add plain-language statement about direction and significance
        if pval is not None:
            if pval < 0.05:
                sig = "statistically significant at the 5% level"
            elif pval < 0.10:
                sig = "marginally significant at the 10% level"
            else:
                sig = "not statistically significant"
            direction = "higher" if out['marginal_effect'] > 0 else "lower" if out['marginal_effect'] < 0 else "no difference"
            desc_parts.append(f"Being female is associated with a {direction} probability of approval and this effect is {sig} (by the marginal effect).")

    if not desc_parts:
        description = "Could not extract statistics for 'Female' from the provided model output."
    else:
        description = " ; ".join(desc_parts)

    return {
        "object": out,
        "description": description
    }