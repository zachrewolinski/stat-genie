def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator from a fitted statsmodels Logit result.

    Returns a dictionary with:
      - "object": a dict containing numeric statistics (coefficient, SE, z, p-value,
                  odds ratio, 95% CI for coef and odds ratio, sample size, and
                  average marginal effect if available).
      - "description": a short plain-English interpretation of those statistics
                       in the context of whether gender affects mortgage acceptance.
    """
    import numpy as np
    import pandas as pd

    results = model_output

    # Ensure the results object has parameters
    if not hasattr(results, "params"):
        raise TypeError("Provided model_output does not appear to be a statsmodels results object with .params")

    # Ensure 'female' was in the model
    params_index = list(results.params.index)
    if "female" not in params_index:
        raise KeyError("'female' not found in model parameters. Available parameters: " + ", ".join(params_index))

    # Extract coefficient and related statistics
    coef = float(results.params["female"])
    # Standard error
    try:
        se = float(results.bse["female"])
    except Exception:
        # fallback: compute from cov_params if available
        cov = getattr(results, "cov_params", None)
        if cov is not None and "female" in cov.index:
            se = float(np.sqrt(cov.loc["female", "female"]))
        else:
            se = float("nan")

    # z (or t) statistic and p-value
    z_stat = coef / se if (se != 0 and not np.isnan(se)) else float("nan")
    # p-value from results if available
    p_value = float(results.pvalues["female"]) if ("female" in results.pvalues.index) else float("nan")

    # Confidence interval for coefficient (95%)
    try:
        ci_df = results.conf_int()
        ci_lower, ci_upper = float(ci_df.loc["female", 0]), float(ci_df.loc["female", 1])
    except Exception:
        # approximate using normal-based CI
        crit = 1.96
        ci_lower, ci_upper = coef - crit * se, coef + crit * se

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    odds_ci_lower, odds_ci_upper = float(np.exp(ci_lower)), float(np.exp(ci_upper))

    # Sample size if available
    nobs = int(results.nobs) if hasattr(results, "nobs") else None

    # Try to compute average marginal effect (AME) for 'female' if possible
    ame = None
    ame_se = None
    ame_p = None
    try:
        # get_margeff may raise if model object doesn't support it
        margeff = results.get_margeff(at="overall")
        # margeff.summary() contains human output; extract numeric results
        # margeff.margeff is array; margeff.margeff_se is array
        # We need to find index for 'female'
        me_index = list(margeff.summary_frame().index)
        if "female" in me_index:
            sf = margeff.summary_frame()
            ame = float(sf.loc["female", "dy/dx"])
            ame_se = float(sf.loc["female", "Std. Err."])
            # compute p-value for marginal effect if not present directly
            ame_z = ame / ame_se if (ame_se != 0 and not np.isnan(ame_se)) else float("nan")
            # two-sided p-value
            from scipy import stats
            ame_p = float(2 * (1 - stats.norm.cdf(abs(ame_z)))) if not np.isnan(ame_z) else None
    except Exception:
        # silently continue if marginal effects cannot be computed
        ame = ame_se = ame_p = None

    # Build structured object to return
    result_object = {
        "coef_female": coef,
        "std_err_female": se,
        "z_female": z_stat,
        "p_value_female": p_value,
        "95CI_coef_female": (ci_lower, ci_upper),
        "odds_ratio_female": odds_ratio,
        "95CI_odds_ratio_female": (odds_ci_lower, odds_ci_upper),
        "nobs": nobs,
        "avg_marginal_effect_female": ame,
        "ame_se": ame_se,
        "ame_p_value": ame_p,
    }

    # Build a short interpretation
    alpha = 0.05
    if np.isnan(p_value):
        significance_text = "p-value not available"
    else:
        significance_text = (
            f"statistically significant at alpha={alpha}"
            if p_value < alpha
            else f"not statistically significant at alpha={alpha}"
        )

    # Direction text
    if not np.isnan(odds_ratio):
        if odds_ratio > 1:
            direction = "higher odds of acceptance compared with males"
        elif odds_ratio < 1:
            direction = "lower odds of acceptance compared with males"
        else:
            direction = "no change in odds compared with males"
    else:
        direction = "effect direction unclear (odds ratio not available)"

    description_lines = [
        f"The logistic regression coefficient for 'female' = {coef:.4f} (SE = {se:.4f}), z = {z_stat:.2f}, p = {p_value:.4f}.",
        f"This corresponds to an odds ratio = {odds_ratio:.3f} with 95% CI = ({odds_ci_lower:.3f}, {odds_ci_upper:.3f}).",
        f"Interpretation: Female applicants have {direction}, conditional on the included controls; this effect is {significance_text}.",
    ]

    if ame is not None:
        description_lines.append(
            f"Average marginal effect (change in probability) for being female ≈ {ame:.4f} (SE = {ame_se:.4f}, p = {ame_p:.4f})."
        )

    description = " ".join(description_lines)

    return {"object": result_object, "description": description}