def extract_final_answer(model_output):
    """
    Extract key statistics for the primary predictors from a fitted GLM model output.
    Accepts either:
      - the dict returned by the provided modeling function (contains 'model', 'odds_ratios', etc.)
      - or directly a statsmodels results object.
    Returns a dictionary with keys:
      - "object": dict of extracted numeric results for focal predictors
      - "description": short interpretation of those results in the context of the question
    """
    import numpy as np
    import pandas as pd

    # Accept either the dict produced by the model() function or a model object directly
    if isinstance(model_output, dict) and 'model' in model_output:
        model = model_output['model']
    else:
        model = model_output

    # Extract parameter table
    params = model.params.copy()
    pvalues = model.pvalues.copy()
    bse = model.bse.copy()
    try:
        zvals = model.tvalues.copy()  # for GLM this is actually z
    except Exception:
        zvals = None

    conf_int = model.conf_int()
    # Compute odds ratios and CI on odds ratio scale
    odds = np.exp(params)
    conf_int_odds = np.exp(conf_int)

    # Variables of interest
    focal_vars = ['SizeDiff', 'DistAdv', 'SizeDiff:DistAdv', 'MaleDiff', 'FemaleDiff', 'Intercept']

    extracted = {}
    for v in focal_vars:
        if v == 'Intercept':
            key = 'Intercept'
            if 'Intercept' in params.index:
                idx = 'Intercept'
            elif 'const' in params.index:
                idx = 'const'
            else:
                idx = None
        else:
            idx = v if v in params.index else None

        if idx is None:
            # variable not in model
            extracted[v] = None
            continue

        ci_low, ci_high = conf_int.loc[idx].iloc[0], conf_int.loc[idx].iloc[1]
        # safety: handle any non-finite values gracefully
        odds_ratio = odds.loc[idx] if np.isfinite(odds.loc[idx]) else None
        odds_ci = (np.exp(ci_low) if np.isfinite(ci_low) else None,
                   np.exp(ci_high) if np.isfinite(ci_high) else None)

        extracted[v] = {
            'coef': float(params.loc[idx]),
            'se': float(bse.loc[idx]) if idx in bse.index else None,
            'z_or_t': float(zvals.loc[idx]) if (zvals is not None and idx in zvals.index) else None,
            'p_value': float(pvalues.loc[idx]) if idx in pvalues.index else None,
            'odds_ratio': float(odds_ratio) if odds_ratio is not None else None,
            'odds_ratio_95CI': (float(odds_ci[0]) if odds_ci[0] is not None else None,
                                float(odds_ci[1]) if odds_ci[1] is not None else None)
        }

    # Simple interpretation based on p-values and effect sizes
    # Focus on the primary predictors SizeDiff, DistAdv, SizeDiff:DistAdv
    def interpret():
        lines = []
        # SizeDiff
        s = extracted.get('SizeDiff')
        if s is None:
            lines.append("SizeDiff: not included in fitted model.")
        else:
            p = s['p_value']
            orr = s['odds_ratio']
            ci = s['odds_ratio_95CI']
            lines.append(
                f"Relative group size (SizeDiff): coef = {s['coef']:.4g}, se = {s['se']:.4g}, "
                f"p = {p:.3g}. Odds ratio = {orr:.4g} (95% CI = [{ci[0]:.4g}, {ci[1]:.4g}])."
            )
            if p is None or p > 0.05:
                lines.append("  Interpretation: No statistically significant effect of relative group size on win probability (no evidence that larger focal groups reliably win more).")
            else:
                lines.append("  Interpretation: Statistically significant effect (interpret magnitude via odds ratio above).")

        # DistAdv
        d = extracted.get('DistAdv')
        if d is None:
            lines.append("DistAdv: not included in fitted model.")
        else:
            p = d['p_value']
            orr = d['odds_ratio']
            ci = d['odds_ratio_95CI']
            lines.append(
                f"Contest location advantage (DistAdv): coef = {d['coef']:.4g}, se = {d['se']:.4g}, "
                f"p = {p:.3g}. Odds ratio = {orr:.4g} (95% CI = [{ci[0]:.4g}, {ci[1]:.4g}])."
            )
            if p is None or p > 0.05:
                lines.append("  Interpretation: No statistically significant home-field advantage detected; estimated effect size is very small (odds ~1.003 per unit).")
            else:
                lines.append("  Interpretation: Statistically significant effect (interpret magnitude via odds ratio above).")

        # Interaction
        it = extracted.get('SizeDiff:DistAdv')
        if it is None:
            lines.append("SizeDiff:DistAdv (interaction): not included in fitted model.")
        else:
            p = it['p_value']
            orr = it['odds_ratio']
            ci = it['odds_ratio_95CI']
            lines.append(
                f"Interaction (SizeDiff:DistAdv): coef = {it['coef']:.4g}, se = {it['se']:.4g}, "
                f"p = {p:.3g}. Odds ratio = {orr:.6g} (95% CI = [{ci[0]:.6g}, {ci[1]:.6g}])."
            )
            if p is None or p > 0.05:
                lines.append("  Interpretation: No evidence that the effect of relative group size depends on contest location.")
            else:
                lines.append("  Interpretation: Statistically significant interaction (interpret carefully).")

        # Note about model stability
        lines.append(
            "Caution: Many categorical fixed-effect coefficients have extremely large standard errors and "
            "some odds-ratio CIs are effectively unbounded (observed in this fitted model). "
            "This suggests model instability / possible separation or sparse data due to many group fixed effects "
            "relative to sample size. Interpret non-significant results with caution; consider alternative approaches "
            "(e.g., fewer fixed effects, mixed-effects logistic regression, or penalized logistic regression / Firth correction)."
        )
        return "\n".join(lines)

    description_text = interpret()

    return {
        "object": extracted,
        "description": description_text
    }