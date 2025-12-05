def extract_final_answer(model_output):
    """
    Extract the effect of 'LiveBait' on fish-per-hour from the model output.
    Chooses Negative Binomial model if available (and Poisson is overdispersed),
    otherwise uses the Poisson model.

    Returns:
      {
        "object": {
            "model_used": "neg_bin" or "poisson",
            "dispersion": <float or None>,
            "coef": <float>,
            "se": <float>,
            "z_or_t": <float>,           # statsmodels uses z for GLM; numeric test stat
            "pvalue": <float>,
            "ci_lower": <float>,
            "ci_upper": <float>,
            "irr": <float>,              # incidence rate ratio = exp(coef)
            "irr_ci_lower": <float>,
            "irr_ci_upper": <float>
        },
        "description": "<brief interpretation>"
      }
    """
    import numpy as np

    # Basic validation
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict. Expected dict with keys 'poisson_model', 'dispersion', 'neg_bin_model'."
        }

    dispersion = model_output.get('dispersion', None)
    neg_bin = model_output.get('neg_bin_model', None)
    poisson = model_output.get('poisson_model', None)

    # Prefer Negative Binomial if present (since Poisson reported overdispersion)
    model = None
    model_used = None
    if neg_bin is not None:
        model = neg_bin
        model_used = 'neg_bin'
    elif poisson is not None:
        model = poisson
        model_used = 'poisson'
    else:
        return {
            "object": None,
            "description": "No fitted model found in model_output under keys 'neg_bin_model' or 'poisson_model'."
        }

    # Extract stats for 'LiveBait'
    try:
        params = model.params
        bse = model.bse  # standard errors
        pvalues = model.pvalues
        # conf_int returns a DataFrame-like two-column array indexed by parameter name
        ci = model.conf_int()
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to read parameters from the selected model ({model_used}): {e}"
        }

    if 'LiveBait' not in params.index:
        return {
            "object": None,
            "description": "The model does not contain a parameter named 'LiveBait'. Check variable names used in the model."
        }

    coef = float(params.loc['LiveBait'])
    se = float(bse.loc['LiveBait'])
    pval = float(pvalues.loc['LiveBait'])
    ci_lower = float(ci.loc['LiveBait', 0])
    ci_upper = float(ci.loc['LiveBait', 1])

    # Test statistic: statsmodels GLM uses z-value (coef / se)
    z_or_t = coef / se if se != 0 else np.nan

    # Incidence Rate Ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    result_obj = {
        "model_used": model_used,
        "dispersion": float(dispersion) if dispersion is not None else None,
        "coef": coef,
        "se": se,
        "z_or_t": float(z_or_t),
        "pvalue": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper
    }

    # Interpretation: short and focused on whether LiveBait affects fish-per-hour
    if pval < 0.05:
        significance = "statistically significant (p < 0.05)"
    elif pval < 0.10:
        significance = "marginally significant (p < 0.10)"
    else:
        significance = "not statistically significant (p >= 0.10)"

    description = (
        f"Using the {'negative binomial' if model_used=='neg_bin' else 'Poisson'} model "
        f"(dispersion={result_obj['dispersion']:.2f}), the estimated log-rate coefficient for LiveBait is "
        f"{result_obj['coef']:.3f} (SE={result_obj['se']:.3f}, z={result_obj['z_or_t']:.2f}, p={result_obj['pvalue']:.3g}). "
        f"This corresponds to an incidence rate ratio (IRR) = {result_obj['irr']:.3f} "
        f"with 95% CI [{result_obj['irr_ci_lower']:.3f}, {result_obj['irr_ci_upper']:.3f}]. "
        f"The effect is {significance}. "
        f"Interpreting the IRR: groups using live bait are estimated to catch about {result_obj['irr']:.2f}× the rate "
        f"of fish per hour compared with groups not using live bait (holding other model covariates constant)."
    )

    return {
        "object": result_obj,
        "description": description
    }