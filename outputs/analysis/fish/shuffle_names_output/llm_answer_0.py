def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% confidence intervals,
    and incidence rate ratios (IRRs = exp(coef)) from the final fitted model
    returned in `model_output`.

    Returns a dict with:
      - "object": dict containing model_family, dispersion, and per-variable stats
      - "description": brief interpretation in the context of fish caught per hour

    Expects `model_output` to be the dictionary returned by the modeling function,
    with keys including 'final_model', 'final_family', and 'dispersion'.
    """
    import math
    import numpy as np

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model function.")

    final_model = model_output.get('final_model')
    final_family = model_output.get('final_family', None)
    dispersion = model_output.get('dispersion', None)

    if final_model is None:
        raise ValueError("model_output does not contain 'final_model'.")

    # Extract basic result objects (statsmodels GLMResults-like)
    try:
        params = final_model.params
        bse = final_model.bse
        pvalues = final_model.pvalues
        ci = final_model.conf_int()  # DataFrame with columns [0,1] (lower, upper)
    except Exception as e:
        raise RuntimeError(f"Unable to extract statistics from final_model: {e}")

    effects = {}
    for var in params.index:
        coef = float(params.loc[var])
        se = float(bse.loc[var]) if var in bse.index else None
        pval = float(pvalues.loc[var]) if var in pvalues.index else None

        # confidence interval
        try:
            ci_low = float(ci.loc[var, 0])
            ci_high = float(ci.loc[var, 1])
        except Exception:
            # fallback if conf_int returns different indexing
            try:
                ci_low = float(ci.loc[var].iloc[0])
                ci_high = float(ci.loc[var].iloc[1])
            except Exception:
                ci_low = None
                ci_high = None

        # incidence rate ratio (IRR) and its CI
        irr = math.exp(coef)
        irr_ci = [math.exp(ci_low) if ci_low is not None else None,
                  math.exp(ci_high) if ci_high is not None else None]

        effects[var] = {
            'coef_log_rate': coef,                    # log(rate ratio)
            'std_err': se,
            'p_value': pval,
            'conf_int_95_log': [ci_low, ci_high],
            'IRR': irr,                               # multiplicative effect on fish/hour
            'IRR_95_CI': irr_ci,
            'significant_at_0.05': (pval is not None and pval < 0.05)
        }

    result_object = {
        'model_family': final_family,
        'dispersion': float(dispersion) if dispersion is not None else None,
        'effects': effects
    }

    description = (
        "This model is a log-linear count model (log link) with log(hours) used as an offset, "
        "so each coefficient is the log of the rate ratio for fish caught per hour. "
        "Exp(coef) (labeled IRR) gives the multiplicative change in the expected fish/hour "
        "for a one-unit increase in the covariate (or presence vs absence for binary vars). "
        "The returned 'effects' dictionary includes coefficient, SE, p-value, 95% CI on the "
        "log scale, the IRR and its 95% CI, and a boolean for significance at alpha=0.05. "
        "The 'dispersion' value >1 indicates overdispersion (the model output showed NegativeBinomial selected)."
    )

    return {'object': result_object, 'description': description}