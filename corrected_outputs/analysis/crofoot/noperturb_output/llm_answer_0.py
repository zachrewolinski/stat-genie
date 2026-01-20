def extract_final_answer(model_output):
    """
    Extract coefficients, (clustered) standard errors, p-values, 95% CIs, and odds ratios
    for the key predictors: 'log_size_ratio', 'LocationAdv', 'SizeLoc_interaction'.

    Returns:
      {
        "object": {
            "<var>": {
                "coef": float,
                "bse": float,
                "z": float,
                "p": float,
                "ci95_lower": float,
                "ci95_upper": float,
                "odds_ratio": float,
                "or_ci95_lower": float,
                "or_ci95_upper": float
            }, ...
        },
        "description": str
      }
    The function is robust to the ClusteredResultsWrapper defined in the modeling
    code (which exposes .fitted, .params, .cov_cluster, .bse_cluster) as well as
    to a standard statsmodels results object or a robustcov results object.
    """
    import math
    import numpy as np
    import pandas as pd

    # Helper: obtain params (pandas Series)
    if hasattr(model_output, 'params'):
        params = model_output.params
    elif hasattr(model_output, 'fitted') and hasattr(model_output.fitted, 'params'):
        params = model_output.fitted.params
    else:
        raise ValueError("Could not find parameter vector on model_output")

    # Ensure params is a pandas Series
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Parameter vector could not be converted to pandas.Series")

    # Helper: obtain standard errors (prefer clustered/bse_cluster if available)
    bse = None
    # Preferred attributes in order
    candidates = [
        'bse',             # statsmodels result or wrapper property
        'bse_cluster',     # our wrapper
    ]
    for attr in candidates:
        if hasattr(model_output, attr):
            candidate = getattr(model_output, attr)
            # If candidate is a pandas Series with index aligned to params, use it
            if isinstance(candidate, pd.Series):
                bse = candidate.reindex(params.index)
            else:
                # candidate may be numpy array; try to align by position
                try:
                    bse = pd.Series(np.asarray(candidate), index=params.index)
                except Exception:
                    bse = None
            if bse is not None:
                break

    # If still None, try extracting covariance and compute bse = sqrt(diag(cov))
    if bse is None:
        cov = None
        if hasattr(model_output, 'cov_cluster') and model_output.cov_cluster is not None:
            cov = model_output.cov_cluster
        elif hasattr(model_output, 'cov_params'):
            try:
                cov = model_output.cov_params()
            except Exception:
                cov = None
        if cov is not None:
            try:
                cov_arr = np.asarray(cov)
                bse = pd.Series(np.sqrt(np.diag(cov_arr)), index=params.index)
            except Exception:
                bse = None

    # Final fallback: try the underlying fitted result if present
    if bse is None and hasattr(model_output, 'fitted') and hasattr(model_output.fitted, 'bse'):
        try:
            fbse = model_output.fitted.bse
            if isinstance(fbse, pd.Series):
                bse = fbse.reindex(params.index)
            else:
                bse = pd.Series(np.asarray(fbse), index=params.index)
        except Exception:
            bse = None

    if bse is None:
        raise ValueError("Could not determine standard errors (clustered or otherwise) from model_output")

    # Compute z, p-values (two-sided), 95% CI, and odds ratios
    # Use math.erfc to compute two-sided p: p = erfc(|z|/sqrt(2))
    z = params / bse
    p_vals = z.abs().apply(lambda val: math.erfc(abs(val) / math.sqrt(2)))
    ci_lower = params - 1.96 * bse
    ci_upper = params + 1.96 * bse
    or_vals = np.exp(params)
    or_ci_lower = np.exp(ci_lower)
    or_ci_upper = np.exp(ci_upper)

    # Variables of interest
    vars_of_interest = ['log_size_ratio', 'LocationAdv', 'SizeLoc_interaction']
    effects = {}
    missing = []
    for v in vars_of_interest:
        if v in params.index:
            effects[v] = {
                "coef": float(params.loc[v]),
                "bse": float(bse.loc[v]),
                "z": float(z.loc[v]),
                "p": float(p_vals.loc[v]),
                "ci95_lower": float(ci_lower.loc[v]),
                "ci95_upper": float(ci_upper.loc[v]),
                "odds_ratio": float(or_vals.loc[v]),
                "or_ci95_lower": float(or_ci_lower.loc[v]),
                "or_ci95_upper": float(or_ci_upper.loc[v])
            }
        else:
            missing.append(v)

    # Build a concise interpretation
    lines = []
    if len(missing) > 0:
        lines.append("Warning: the following predictors were not found in the model results: {}".format(", ".join(missing)))

    for v, stats in effects.items():
        sig = "statistically significant (p < 0.05)" if stats["p"] < 0.05 else "not statistically significant (p >= 0.05)"
        # Interpret sign
        direction = "positive" if stats["coef"] > 0 else ("zero" if stats["coef"] == 0 else "negative")
        # Special phrasing for interaction
        if v == 'SizeLoc_interaction' or v == 'SizeLoc_interaction'.replace('_', ''):
            interpret = (
                "Interaction: a {} coefficient ({:.3f}, p = {:.3f}) indicates that the effect of relative group size on winning "
                "depends on contest location. {}. ".format(direction, stats["coef"], stats["p"], 
                                                           "This interaction is " + sig if stats["p"] < 0.05 else "This interaction is " + sig)
            )
            # Add qualitative direction meaning
            if stats["coef"] > 0:
                interpret += "Specifically, a positive interaction means the advantage of being larger increases when the focal group has a local location advantage."
            elif stats["coef"] < 0:
                interpret += "Specifically, a negative interaction means the advantage of being larger decreases when the focal group has a local location advantage."
            else:
                interpret += "No directional effect detected."
        else:
            # For main effects, convert to odds ratio wording
            interpret = (
                "Main effect: {} has coefficient = {:.3f}, SE = {:.3f}, z = {:.2f}, p = {:.3f}. ".format(v, stats["coef"], stats["bse"], stats["z"], stats["p"])
            )
            interpret += "This is {}. ".format(sig)
            interpret += "The estimated odds ratio is {:.3f} (95% CI [{:.3f}, {:.3f}]). ".format(
                stats["odds_ratio"], stats["or_ci95_lower"], stats["or_ci95_upper"])
            if v == 'log_size_ratio':
                interpret += ("Interpretation: a 1-unit increase in the natural-log size ratio (i.e., focal group being ≈e times larger than the other) "
                              "multiplies the odds of the focal group winning by {:.3f}.").format(stats["odds_ratio"])
            elif v == 'LocationAdv':
                interpret += ("Interpretation: a 1-unit increase in LocationAdv (focal group relatively closer to its home-range center) "
                              "multiplies the odds of the focal group winning by {:.3f}.").format(stats["odds_ratio"])

        lines.append(interpret)

    description = "\n".join(lines)

    return {"object": effects, "description": description}