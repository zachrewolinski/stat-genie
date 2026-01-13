def extract_final_answer(model_output):
    """
    Extracts the coefficient, uncertainty, and inference for the Genus_Homo effect
    from a fitted statsmodels GLM (or clustered robust results) object and returns
    a short interpretation about whether modern humans have higher AMTL.

    Returns:
      {
        "object": { ... numeric results and conclusion ... },
        "description": "Short textual interpretation in context"
      }
    """
    import numpy as np
    from math import exp
    try:
        # Try to access parameter names and values
        params = model_output.params
    except Exception:
        raise ValueError("model_output does not expose .params")

    # Identify the parameter name for Genus_Homo (allow small flexibility)
    param_name = None
    if 'Genus_Homo' in params.index:
        param_name = 'Genus_Homo'
    else:
        # try to find any parameter that contains the substring
        matches = [n for n in params.index if 'Genus_Homo' in n]
        if len(matches) >= 1:
            param_name = matches[0]

    if param_name is None:
        raise KeyError("Could not find a parameter named 'Genus_Homo' in model_output.params")

    # Extract coefficient
    coef = float(params[param_name])

    # Standard error: try .bse then fallback to sqrt of diagonal of cov_params()
    se = None
    try:
        se = float(model_output.bse[param_name])
    except Exception:
        # fallback
        try:
            cov = model_output.cov_params()
            se = float(np.sqrt(np.diag(cov))[list(params.index).index(param_name)])
        except Exception:
            raise ValueError("Could not obtain standard error for parameter '{}'".format(param_name))

    # z-statistic and two-sided p-value: prefer model's pvalues if available
    try:
        p_value = float(model_output.pvalues[param_name])
        # compute z from coef/se for reporting consistency
        z_stat = float(coef / se) if se != 0 else float('nan')
    except Exception:
        # fallback: compute z and p using normal approximation
        z_stat = float(coef / se) if se != 0 else float('nan')
        try:
            # Use normal distribution survival function for two-sided p
            from math import erf, sqrt
            # use scipy-style normal cdf if available
            try:
                from scipy import stats as _scistats
                p_value = float(2.0 * (1.0 - _scistats.norm.cdf(abs(z_stat))))
            except Exception:
                # approximate with error function
                # cdf = 0.5*(1+erf(z/sqrt(2)))
                p_value = float(2.0 * (1.0 - (0.5 * (1.0 + erf(abs(z_stat) / sqrt(2.0))))))
        except Exception:
            p_value = float('nan')

    # 95% confidence interval for coefficient: try model's conf_int()
    try:
        ci = model_output.conf_int().loc[param_name].astype(float).tolist()
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        # fallback normal approx
        z_crit = 1.959963984540054  # approx for 95% two-sided
        ci_low = coef - z_crit * se
        ci_high = coef + z_crit * se

    # Transform to odds ratio scale (logit link)
    try:
        or_val = float(exp(coef))
        or_ci_low = float(exp(ci_low))
        or_ci_high = float(exp(ci_high))
    except Exception:
        or_val = or_ci_low = or_ci_high = float('nan')

    # Form conclusion: positive coefficient and p < 0.05 means evidence of higher AMTL in Homo
    alpha = 0.05
    if np.isfinite(p_value):
        if (coef > 0) and (p_value < alpha):
            conclusion = ("Yes: The Genus_Homo coefficient is positive (coef = {coef:.4f}), "
                          "and statistically significant (p = {p:.3g}), indicating higher "
                          "probability (odds ratio = {or_: .3f}, 95% CI [{or_lo:.3f}, {or_hi:.3f}]) "
                          "of AMTL in Homo sapiens compared to the non-human primates, "
                          "after adjusting for age, sex, and tooth class.").format(
                              coef=coef, p=p_value, or_=or_val, or_lo=or_ci_low, or_hi=or_ci_high)
        else:
            conclusion = ("No strong evidence: The Genus_Homo coefficient is {sign} (coef = {coef:.4f}), "
                          "with p = {p:.3g}, so we cannot conclude a statistically significant higher "
                          "AMTL frequency in Homo sapiens after adjusting for the covariates. "
                          "On the odds ratio scale: OR = {or_: .3f}, 95% CI [{or_lo:.3f}, {or_hi:.3f}].").format(
                              sign="positive" if coef > 0 else "negative or null", coef=coef, p=p_value,
                              or_=or_val, or_lo=or_ci_low, or_hi=or_ci_high)
    else:
        conclusion = ("Could not compute p-value reliably for the Genus_Homo effect; "
                      "reporting estimates without a formal significance decision.")

    # Prepare object to return
    result_object = {
        "parameter": param_name,
        "coef": coef,
        "se": se,
        "z": z_stat,
        "p_value": p_value,
        "ci_95_coef": [ci_low, ci_high],
        "odds_ratio": or_val,
        "ci_95_odds_ratio": [or_ci_low, or_ci_high],
        "alpha": alpha,
        "conclusion": conclusion
    }

    description = (
        "Extracted the Genus_Homo regression coefficient from the fitted binomial GLM (logit link). "
        "The coefficient is on the log-odds scale; exponentiated to give the odds ratio. "
        "The conclusion states whether Homo sapiens show a statistically significantly higher "
        "probability of antemortem tooth loss (AMTL) compared to the pooled non-human primate genera, "
        "after adjusting for age (Age_z), probabilistic sex (ProbMale_z), and tooth class, "
        "using a two-sided test at alpha = {alpha}."
    ).format(alpha=alpha)

    return {"object": result_object, "description": description}