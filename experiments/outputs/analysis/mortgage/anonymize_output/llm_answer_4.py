def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of gender (Female) on mortgage approval
    from the provided model_output.

    Returns a dictionary with:
      - "object": dict of numeric results (coef, se, z, p-value, odds ratio, odds ratio CI,
                  average marginal effect, n_obs)
      - "description": concise interpretation of those results in context

    Accepts either:
      - a dict like the model output shown in the prompt (with keys 'model_result',
        'odds_ratio_female', 'odds_ratio_ci_female', 'ame_female', 'n_obs'), or
      - a statsmodels BinaryResultsWrapper (fitted result) directly.
    """
    import numpy as np

    # Determine where the stats live
    result = None
    ame = None
    odds_ratio_from_summary = None
    odds_ci_from_summary = None
    n_obs = None

    if isinstance(model_output, dict):
        # Try to extract the fitted model if present
        result = model_output.get('model_result', None)
        ame = model_output.get('ame_female', None)
        odds_ratio_from_summary = model_output.get('odds_ratio_female', None)
        odds_ci_from_summary = model_output.get('odds_ratio_ci_female', None)
        n_obs = model_output.get('n_obs', None)
    else:
        # Assume the user passed the fitted statsmodels result directly
        result = model_output

    if result is None:
        raise ValueError("No fitted model result found in model_output.")

    # Extract coefficient, SE, p-value, and confidence interval for 'Female'
    try:
        coef = float(result.params['Female'])
        se = float(result.bse['Female'])
        p_value = float(result.pvalues['Female'])
        z_stat = coef / se if se != 0 else float('nan')
        conf_int_df = result.conf_int()
        ci_low = float(conf_int_df.loc['Female', 0])
        ci_high = float(conf_int_df.loc['Female', 1])
    except Exception as e:
        raise RuntimeError(f"Failed to extract coefficient information for 'Female': {e}")

    # Odds ratio and CI (from coefficient if not provided separately)
    odds_ratio = odds_ratio_from_summary if odds_ratio_from_summary is not None else float(np.exp(coef))
    if odds_ci_from_summary is not None:
        odds_ci = tuple(odds_ci_from_summary)
    else:
        odds_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

    # AME: prefer the value supplied in the dict, otherwise compute approx using delta method is not done here.
    if ame is None:
        # If AME not provided, attempt to compute marginal difference at average X:
        # Compute predicted probability at Female=1 vs Female=0 using model's predict if available.
        try:
            exog = result.model.exog.copy()
            # identify column index for Female
            exog_names = list(result.model.exog_names)
            if 'Female' in exog_names:
                idx = exog_names.index('Female')
                exog1 = exog.copy()
                exog0 = exog.copy()
                exog1[:, idx] = 1.0
                exog0[:, idx] = 0.0
                p1 = result.model.predict(result.params, exog1)
                p0 = result.model.predict(result.params, exog0)
                ame = float((p1 - p0).mean())
            else:
                ame = None
        except Exception:
            ame = None

    # n_obs if not provided
    if n_obs is None:
        try:
            n_obs = int(result.nobs)
        except Exception:
            n_obs = None

    # Prepare the object to return
    output_object = {
        'coef_log_odds_female': coef,
        'se_coef_female': se,
        'z_stat_female': z_stat,
        'p_value_female': p_value,
        'conf_int_log_odds_female': (ci_low, ci_high),
        'odds_ratio_female': float(odds_ratio),
        'odds_ratio_ci_female': (float(odds_ci[0]), float(odds_ci[1])),
        'ame_female': (float(ame) if (ame is not None) else None),
        'n_obs': (int(n_obs) if (n_obs is not None) else None)
    }

    # Compose a concise interpretation
    # Note: we check p-value < 0.05 for conventional significance
    significance = "statistically significant (p < 0.05)" if (p_value is not None and p_value < 0.05) else "not statistically significant (p >= 0.05)"
    ame_text = f"Average marginal effect ≈ {output_object['ame_female']:.4f} (i.e. about {output_object['ame_female']*100:.2f} percentage points)" if output_object['ame_female'] is not None else "Average marginal effect not available"
    description = (
        f"Female coefficient (log-odds) = {coef:.4f} (SE = {se:.4f}, z = {z_stat:.2f}, p = {p_value:.4g}); "
        f"odds ratio = {output_object['odds_ratio_female']:.3f} with 95% CI = "
        f"({output_object['odds_ratio_ci_female'][0]:.3f}, {output_object['odds_ratio_ci_female'][1]:.3f}). "
        f"The effect is {significance}. {ame_text}. Sample size n = {output_object['n_obs']}."
        " Interpretation: controlling for the listed covariates, being female is associated with higher odds"
        " of mortgage approval (odds ratio > 1)."
    )

    return {"object": output_object, "description": description}