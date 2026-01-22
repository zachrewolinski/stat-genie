def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted logistic regression model output.

    Returns:
      {
        "object": {
            "coefficient": float,
            "std_err": float,
            "z": float,
            "p_value": float,
            "odds_ratio": float,
            "odds_ratio_95ci": [float_lower, float_upper],
            "n_obs": int
        },
        "description": str  # brief interpretation in context
      }
    """
    import numpy as np
    import pandas as pd

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    res = model_output.get('model', None)
    if res is None:
        raise ValueError("model_output does not contain a 'model' key with a fitted statsmodels result object.")

    # Attempt to extract statistics for the 'female' predictor
    try:
        coef = float(res.params['female'])
        se = float(res.bse['female'])
        pval = float(res.pvalues['female'])
        # z-stat can be taken from params / bse or from result.zvalues if available
        try:
            zstat = float(res.tvalues['female'])
        except Exception:
            zstat = float(coef / se) if se != 0 else float('nan')
        # odds ratio and confidence interval for the odds ratio
        odds_ratio = float(np.exp(coef))
        ci = res.conf_int().loc['female'].values  # [lower, upper] on log-odds scale
        or_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]
        # sample size
        n_obs = int(model_output.get('n_obs', getattr(res, 'nobs', np.nan)))
    except Exception as e:
        raise RuntimeError(f"Failed to extract 'female' coefficient statistics: {e}")

    # Construct a concise interpretation
    # Percent change in odds
    pct_change = (odds_ratio - 1.0) * 100.0
    sig_text = "statistically significant" if pval < 0.05 else "not statistically significant"
    description = (
        f"Controlling for the listed covariates (race, credit scores, DTI, LTV, etc.), the estimated log-odds "
        f"coefficient for female = {coef:.3f} (SE = {se:.3f}, z = {zstat:.3f}, p = {pval:.3g}). "
        f"This corresponds to an odds ratio = {odds_ratio:.3f} (95% CI {or_ci[0]:.3f} to {or_ci[1]:.3f}), "
        f"meaning female applicants have about {pct_change:.1f}% {'higher' if odds_ratio>1 else 'lower'} odds "
        f"of mortgage approval than male applicants, holding other variables constant. The result is {sig_text} "
        f"at the α = 0.05 level. Sample size used = {n_obs}."
    )

    result_object = {
        "coefficient": coef,
        "std_err": se,
        "z": zstat,
        "p_value": pval,
        "odds_ratio": odds_ratio,
        "odds_ratio_95ci": or_ci,
        "n_obs": n_obs
    }

    return {"object": result_object, "description": description}