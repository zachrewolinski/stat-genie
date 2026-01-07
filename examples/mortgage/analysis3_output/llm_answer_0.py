def extract_final_answer(model_output):
    """
    Extract the effect of the 'female' indicator on mortgage approval from a fitted model output.

    Input:
      - model_output: either
          * a dict containing at least the fitted statsmodels result under key 'model' (as in the example),
            and optionally 'odds_ratios' and 'conf_odds', or
          * directly a fitted statsmodels result object.

    Returns:
      - dict with keys:
          - "object": dict of extracted statistics (log-odds coef, SE, p-value, odds ratio, 95% CI for odds ratio,
                      boolean indicating significance at alpha=0.05, and a short textual conclusion)
          - "description": brief explanation of what's returned and how to interpret it
    """
    import numpy as np
    import pandas as pd

    # Accept either a dict containing 'model' or the model result itself
    if isinstance(model_output, dict):
        res = model_output.get('model', None)
        odds_input = model_output.get('odds_ratios', None)
        conf_odds_input = model_output.get('conf_odds', None)
    else:
        res = model_output
        odds_input = None
        conf_odds_input = None

    if res is None:
        raise ValueError("No fitted model found in model_output (expected key 'model' or a model result).")

    # Core numeric outputs from statsmodels result
    params = getattr(res, 'params', None)
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)
    conf_int = None
    try:
        conf_int = res.conf_int()
        # conf_int columns are typically [0,1]; rename for safety
        if isinstance(conf_int, pd.DataFrame):
            conf_int.columns = ['2.5%', '97.5%']
    except Exception:
        conf_int = None

    # Ensure 'female' is present
    if params is None or 'female' not in params.index:
        raise KeyError("The fitted model does not contain a parameter named 'female'.")

    # Extract log-odds coefficient, SE, p-value
    coef = float(params['female'])
    se = float(bse['female']) if (bse is not None and 'female' in bse.index) else None
    pval = float(pvalues['female']) if (pvalues is not None and 'female' in pvalues.index) else None

    # Odds ratio: prefer provided odds_ratios if available, otherwise compute from params
    if odds_input is not None:
        # Convert to pandas Series if needed
        odds_series = pd.Series(odds_input) if not isinstance(odds_input, pd.Series) else odds_input
        if 'female' in odds_series.index:
            odds_ratio = float(odds_series['female'])
        else:
            odds_ratio = float(np.exp(coef))
    else:
        odds_ratio = float(np.exp(coef))

    # 95% CI for odds ratio: prefer provided conf_odds, otherwise compute from conf_int
    ci_low = ci_high = None
    if conf_odds_input is not None:
        conf_odds_df = conf_odds_input if isinstance(conf_odds_input, pd.DataFrame) else pd.DataFrame(conf_odds_input)
        # Expect rows indexed by parameter names and columns ['2.5%', '97.5%']
        if 'female' in conf_odds_df.index:
            # handle possible column names
            if '2.5%' in conf_odds_df.columns and '97.5%' in conf_odds_df.columns:
                ci_low = float(conf_odds_df.loc['female', '2.5%'])
                ci_high = float(conf_odds_df.loc['female', '97.5%'])
            else:
                # fallback: take first two columns
                ci_low = float(conf_odds_df.iloc[conf_odds_df.index.get_loc('female'), 0])
                ci_high = float(conf_odds_df.iloc[conf_odds_df.index.get_loc('female'), 1])
    elif conf_int is not None:
        # compute odds CI by exponentiating the log-odds CI
        if 'female' in conf_int.index:
            ci_low = float(np.exp(conf_int.loc['female', '2.5%']))
            ci_high = float(np.exp(conf_int.loc['female', '97.5%']))

    # Determine statistical significance at alpha = 0.05 (if p-value available)
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Construct a short textual conclusion
    if (ci_low is not None) and (ci_high is not None) and (pval is not None):
        if significant:
            conclusion = (
                f"Controlling for the listed covariates, female applicants have higher odds of mortgage approval: "
                f"odds ratio = {odds_ratio:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f}), p = {pval:.3g}."
            )
        else:
            conclusion = (
                f"Controlling for the listed covariates, there is no statistically significant difference in approval odds by gender: "
                f"odds ratio = {odds_ratio:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f}), p = {pval:.3g}."
            )
    else:
        # Partial info available
        parts = [f"odds ratio = {odds_ratio:.3f}"]
        if ci_low is not None and ci_high is not None:
            parts.append(f"95% CI {ci_low:.3f}–{ci_high:.3f}")
        if pval is not None:
            parts.append(f"p = {pval:.3g}")
        conclusion = "Controlling for covariates, female effect: " + ", ".join(parts) + "."

    output_object = {
        'coef_log_odds': coef,
        'std_err': se,
        'p_value': pval,
        'odds_ratio': odds_ratio,
        'ci_odds_ratio_lower': ci_low,
        'ci_odds_ratio_upper': ci_high,
        'statistically_significant_at_0.05': significant,
        'conclusion': conclusion
    }

    description = (
        "This extracts the estimated effect of the 'female' indicator from the fitted logistic model. "
        "The 'coef_log_odds' is the coefficient on the log-odds scale; exponentiating it gives 'odds_ratio'. "
        "If p-value < 0.05, the effect is marked as statistically significant at the 5% level. "
        "All estimates are conditional on the control variables included in the model."
    )

    return {"object": output_object, "description": description}