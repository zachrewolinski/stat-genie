def extract_final_answer(model_output):
    """
    Extract statistics for the 'female' coefficient from a fitted statsmodels LogitResults
    (or BinaryResultsWrapper) object and return a dict with numeric results and a short interpretation.
    
    Returns:
      {
        "object": { ... numeric results ... },
        "description": "brief interpretation string"
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output
    var = 'female'

    # Basic existence checks
    if not hasattr(res, 'params'):
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels results object (missing .params).")

    if var not in res.params.index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(res.params.index)}")

    # Coefficient and basic inference stats
    coef = float(res.params[var])
    # standard error (bse should exist)
    se = float(res.bse[var]) if hasattr(res, 'bse') else float(np.nan)
    # z / t value: statsmodels exposes .tvalues (or .zvalues); fall back to coef/se
    try:
        stat = float(res.tvalues[var])
    except Exception:
        try:
            stat = float(res.zvalues[var])
        except Exception:
            stat = coef / se if se != 0 else float(np.nan)
    # p-value
    p_value = float(res.pvalues[var]) if hasattr(res, 'pvalues') else float(np.nan)

    # 95% confidence interval for coefficient
    try:
        ci_df = res.conf_int()
        # conf_int() may return an array or DataFrame
        if isinstance(ci_df, np.ndarray):
            # find index position for var
            pos = list(res.params.index).index(var)
            ci_low, ci_high = float(ci_df[pos, 0]), float(ci_df[pos, 1])
        else:
            # DataFrame-like
            ci_row = ci_df.loc[var]
            ci_low, ci_high = float(ci_row.iloc[0]), float(ci_row.iloc[1])
    except Exception:
        ci_low, ci_high = float(np.nan), float(np.nan)

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else float(np.nan)
    or_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else float(np.nan)

    # Try to compute average marginal effect (change in probability) for 'female'
    marginal_effect = None
    marginal_pvalue = None
    try:
        me = res.get_margeff(at='overall')  # average marginal effects
        me_df = me.summary_frame()
        # Common column names: 'dy/dx' or 'effect'
        if 'dy/dx' in me_df.columns:
            marginal_effect = float(me_df.loc[var, 'dy/dx'])
        elif 'effect' in me_df.columns:
            marginal_effect = float(me_df.loc[var, 'effect'])
        else:
            # fallback: first column
            marginal_effect = float(me_df.loc[var, me_df.columns[0]])
        # marginal effects p-values (if available)
        if hasattr(me, 'pvalues') and var in me.pvalues.index:
            marginal_pvalue = float(me.pvalues[var])
    except Exception:
        # If marginal effects computation fails, leave as None (this is optional)
        marginal_effect = None
        marginal_pvalue = None

    # Interpretation summary
    significance = "statistically significant (p < 0.05)" if (not np.isnan(p_value) and p_value < 0.05) else "not statistically significant (p >= 0.05 or p unavailable)"
    direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no change in")
    odds_direction = "increase" if odds_ratio > 1 else ("decrease" if odds_ratio < 1 else "no change")
    desc = (
        f"Logistic regression result for variable '{var}': coefficient = {coef:.4f} "
        f"(SE={se:.4f}, stat={stat:.3f}, p={p_value:.3g}). 95% CI for coefficient: [{ci_low:.4f}, {ci_high:.4f}]. "
        f"Odds ratio = {odds_ratio:.3f} (95% CI: [{or_ci_low:.3f}, {or_ci_high:.3f}]).\n"
        f"Interpretation: being female is associated with {direction} log-odds of mortgage approval; "
        f"in odds-ratio terms this is {odds_direction} in the odds of approval. The effect is {significance}."
    )
    if marginal_effect is not None:
        me_p_text = f" (p={marginal_pvalue:.3g})" if marginal_pvalue is not None else ""
        desc += f" Average marginal effect (approx. change in probability for being female) = {marginal_effect:.4f}{me_p_text}."

    result_object = {
        "coef": coef,
        "std_err": se,
        "stat": stat,
        "p_value": p_value,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": (or_ci_low, or_ci_high),
        "marginal_effect": marginal_effect,
        "marginal_effect_pvalue": marginal_pvalue,
    }

    return {"object": result_object, "description": desc}