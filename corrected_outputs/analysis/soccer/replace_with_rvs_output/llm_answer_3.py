def extract_final_answer(model_output):
    """
    Extracts the estimate and inference for the 'dark_skin' coefficient from a fitted statsmodels GLM result
    (expected to be a Negative Binomial model with log(games) offset and clustered robust covariances).
    
    Returns a dictionary with keys:
      - "object": a dict with numeric results (coef, se, pvalue, conf_int, IRR, IRR_conf_int, nobs)
      - "description": a short interpretation of what the numbers mean for the research question.
    """
    import numpy as np

    # Initialize output structure
    out = {"object": None, "description": ""}

    # Basic checks
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a statsmodels results object.") from e

    if 'dark_skin' not in params.index:
        raise KeyError("The fitted model does not contain a parameter named 'dark_skin'.")

    # Extract coefficient, robust standard error, p-value
    coef = float(params['dark_skin'])
    # bse, pvalues, conf_int should be available on the results object (robust result returned by get_robustcov_results)
    try:
        se = float(model_output.bse['dark_skin'])
    except Exception:
        # fallback: try to compute se from cov_params if available
        try:
            cov = model_output.cov_params()
            se = float(np.sqrt(np.abs(cov.loc['dark_skin', 'dark_skin'])))
        except Exception as e:
            raise RuntimeError("Could not extract standard error for 'dark_skin'.") from e

    try:
        pvalue = float(model_output.pvalues['dark_skin'])
    except Exception:
        pvalue = None

    # Confidence interval for coefficient (on log-rate scale)
    try:
        ci = model_output.conf_int().loc['dark_skin']
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # fallback: use normal approx if conf_int unavailable
        z = 1.96
        ci_lower = coef - z * se
        ci_upper = coef + z * se

    # Convert to incidence rate ratio (IRR) by exponentiating coef and CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Number of observations if available
    nobs = getattr(model_output, 'nobs', None)
    if nobs is not None:
        try:
            nobs = int(nobs)
        except Exception:
            pass

    # Assemble numeric result object
    numeric_result = {
        "coef_log_rate": coef,
        "se_log_rate": se,
        "p_value": pvalue,
        "conf_int_log_rate": [ci_lower, ci_upper],
        "IRR": irr,
        "IRR_conf_int": [irr_ci_lower, irr_ci_upper],
        "nobs": nobs
    }

    # Interpretation text
    # Note: model is Negative Binomial with log(games) offset -> IRR is multiplicative effect on red card rate per game
    if pvalue is None:
        sig_text = "p-value unavailable"
    else:
        sig_text = ("statistically significant (p < 0.05)"
                    if pvalue < 0.05 else f"not statistically significant (p = {pvalue:.3f})")

    # Percent change interpretation
    pct_change = (irr - 1) * 100.0
    pct_ci_lower = (irr_ci_lower - 1) * 100.0
    pct_ci_upper = (irr_ci_upper - 1) * 100.0

    description = (
        f"The model coefficient for 'dark_skin' on the log rate scale is {coef:.4f} "
        f"(SE = {se:.4f}), corresponding to an incidence rate ratio (IRR) of {irr:.3f}.\n"
        f"This means that, holding controls constant and accounting for games as exposure, "
        f"dark-skinned players have an estimated {pct_change:.1f}% change in the red-card rate per game "
        f"compared to light-skinned players (95% CI for IRR: [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}]; "
        f"i.e. percent change CI [{pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%]).\n"
        f"The effect is {sig_text}.\n\n"
        f"Notes: The model is a Negative Binomial GLM with log(games) as an offset, so the IRR is multiplicative "
        f"on the expected number of red cards per game. Results above are taken from the fitted model output "
        f"and use the available (clustered) robust standard errors/confidence intervals when present."
    )

    out["object"] = numeric_result
    out["description"] = description

    return out