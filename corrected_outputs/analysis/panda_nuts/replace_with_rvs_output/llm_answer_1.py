def extract_final_answer(model_output):
    """
    Extract and summarize coefficients, standard errors, p-values, and 95% CIs
    for the focal predictors (age, Sex_Male, Help_Received) from a fitted
    statsmodels MixedLMResults (MixedLMResultsWrapper).

    Returns a dictionary with:
      - "object": dict keyed by predictor with numeric results and a short
                  interpretation line
      - "description": plain-English explanation of what the returned numbers mean
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Try to retrieve parameter estimates, standard errors, p-values and CIs
    params = getattr(res, "params", None)
    bse = getattr(res, "bse", None)
    # pvalues may be present; otherwise compute from normal approximation
    pvalues = getattr(res, "pvalues", None)
    try:
        conf = res.conf_int()
    except Exception:
        conf = None

    if params is None or bse is None:
        raise ValueError("Model output does not contain .params or .bse attributes.")

    # Compute p-values if not available
    if pvalues is None:
        z = params / bse
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(z)))

    # Prepare output for focal predictors
    focal_preds = ["age", "Sex_Male", "Help_Received"]
    results = {}
    for pred in focal_preds:
        if pred in params.index:
            coef = float(params[pred])
            se = float(bse[pred])
            p = float(pvalues[pred])
            if conf is not None and pred in conf.index:
                ci_low = float(conf.loc[pred, 0])
                ci_high = float(conf.loc[pred, 1])
            else:
                # approximate 95% CI from coef +/- 1.96*se
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se
            signif = bool(p < 0.05)

            # Brief interpretation specific to variable type
            if pred == "age":
                interp = (f"Each additional year of age is associated with a change of "
                          f"{coef:.3f} nuts/min (95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
                          f"p = {p:.3g}. This is "
                          f"{'statistically significant' if signif else 'not statistically significant'}.")
            elif pred == "Sex_Male":
                interp = (f"Being male (vs female) is associated with a change of "
                          f"{coef:.3f} nuts/min (95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
                          f"p = {p:.3g}. This is "
                          f"{'statistically significant' if signif else 'not statistically significant'}.")
            else:  # Help_Received
                interp = (f"Receiving help (vs not) is associated with a change of "
                          f"{coef:.3f} nuts/min (95% CI [{ci_low:.3f}, {ci_high:.3f}]), "
                          f"p = {p:.3g}. This is "
                          f"{'statistically significant' if signif else 'not statistically significant'}.")

            results[pred] = {
                "coef": coef,
                "std_err": se,
                "p_value": p,
                "95%_CI": [ci_low, ci_high],
                "significant_at_0.05": signif,
                "interpretation": interp
            }
        else:
            results[pred] = {
                "error": f"Predictor '{pred}' not found in model parameters."
            }

    # Optionally include random-intercept variance if available
    random_info = {}
    try:
        cov_re = getattr(res, "cov_re", None)
        if cov_re is not None:
            # For a random intercept only model, the variance is cov_re[0,0]
            # cov_re might be a ndarray or DataFrame
            if hasattr(cov_re, "iloc"):
                var_re = float(cov_re.iloc[0, 0])
            else:
                var_re = float(np.asarray(cov_re)[0, 0])
            random_info["random_intercept_variance"] = var_re
        # residual scale (residual variance)
        if hasattr(res, "scale"):
            random_info["residual_variance_scale"] = float(res.scale)
    except Exception:
        pass

    # Compose description
    description = (
        "Returned object contains coefficient estimate, standard error, two-sided p-value, "
        "95% confidence interval, a boolean for significance at alpha=0.05, and a short "
        "interpretation for each focal predictor (age, Sex_Male, Help_Received). "
        "Coefficients are in units of nuts opened per minute. "
        "Positive coefficients mean an increase in efficiency (nuts/min); negative mean a decrease. "
        "For Sex_Male, the coefficient is the difference (male minus female). For Help_Received, "
        "the coefficient is the difference (received help minus did not receive help)."
    )

    return {"object": {"predictors": results, "random_effects": random_info}, "description": description}