def extract_final_answer(model_output):
    """
    Extract coefficients, SEs, p-values, 95% CIs, odds ratios (and CIs), and
    marginal effects (if available) for the key predictors that test the
    effects of relative group size and contest location.

    Returns:
      {
        "object": {
          "<variable>": {
             "coef": float or None,
             "se": float or None,
             "p_value": float or None,
             "odds_ratio": float or None,
             "or_ci_lower": float or None,
             "or_ci_upper": float or None,
             "significant_0.05": bool or None,
             "marginal_effect": dict or None   # if available (rows from summary_frame)
          },
          ...
          "nobs": int or None
        },
        "description": "Human-readable explanation of what the numbers mean and how to interpret them."
      }
    """
    import numpy as np
    import pandas as pd

    # Variables of primary interest in this study
    target_vars = ['log_size_ratio_z', 'ProximityAdvantage_z', 'size_x_location']

    results = {}
    # initialize container for extracted model-wide items
    extracted = {}

    # Try to get basic attributes from the model output
    params = None
    bse = None
    pvalues = None
    conf = None
    nobs = None

    try:
        params = getattr(model_output, 'params', None)
    except Exception:
        params = None
    try:
        bse = getattr(model_output, 'bse', None)
    except Exception:
        bse = None
    try:
        pvalues = getattr(model_output, 'pvalues', None)
    except Exception:
        pvalues = None
    try:
        # conf_int() usually returns a DataFrame with lower/upper columns
        conf = model_output.conf_int()
    except Exception:
        conf = None
    try:
        nobs = int(getattr(model_output, 'nobs', None))
    except Exception:
        # fallback to model endog length
        try:
            nobs = int(model_output.model.endog.shape[0])
        except Exception:
            nobs = None

    extracted['nobs'] = nobs

    # For each target variable, try to extract metrics; be robust to missing pieces
    for var in target_vars:
        entry = {
            "coef": None,
            "se": None,
            "p_value": None,
            "odds_ratio": None,
            "or_ci_lower": None,
            "or_ci_upper": None,
            "significant_0.05": None,
            "marginal_effect": None
        }

        # coefficient
        try:
            if params is not None and var in params.index:
                coef = float(params.loc[var])
                entry['coef'] = coef
        except Exception:
            entry['coef'] = None

        # standard error
        try:
            if bse is not None and var in bse.index:
                entry['se'] = float(bse.loc[var])
        except Exception:
            entry['se'] = None

        # p-value
        try:
            if pvalues is not None and var in pvalues.index:
                entry['p_value'] = float(pvalues.loc[var])
                entry['significant_0.05'] = (entry['p_value'] is not None) and (entry['p_value'] < 0.05)
        except Exception:
            entry['p_value'] = None
            entry['significant_0.05'] = None

        # confidence interval and odds ratio (exp of coef)
        ci_lower = ci_upper = None
        try:
            if conf is not None:
                # conf may be a DataFrame indexed by parameter names
                try:
                    row = conf.loc[var]
                    ci_lower = float(row.iloc[0])
                    ci_upper = float(row.iloc[1])
                except Exception:
                    # fallback to positional lookup if name-based fails
                    try:
                        idx = list(params.index).index(var)
                        row = conf.iloc[idx]
                        ci_lower = float(row.iloc[0])
                        ci_upper = float(row.iloc[1])
                    except Exception:
                        ci_lower = ci_upper = None
        except Exception:
            ci_lower = ci_upper = None

        try:
            if entry['coef'] is not None:
                entry['odds_ratio'] = float(np.exp(entry['coef']))
            if (ci_lower is not None) and (ci_upper is not None):
                entry['or_ci_lower'] = float(np.exp(ci_lower))
                entry['or_ci_upper'] = float(np.exp(ci_upper))
        except Exception:
            entry['odds_ratio'] = entry['or_ci_lower'] = entry['or_ci_upper'] = None

        # marginal effects if attached to the returned object by the modeling code
        me_info = None
        try:
            margeff_obj = getattr(model_output, 'margeff', None)
            if margeff_obj is not None:
                # Attempt to read a summary frame if available
                try:
                    sf = margeff_obj.summary_frame()
                    # summary_frame typically has one row per variable
                    if var in sf.index:
                        # convert the entire row to dict for completeness
                        me_info = sf.loc[var].to_dict()
                    else:
                        # if var not found in index, try positional match
                        try:
                            idx = list(params.index).index(var)
                            me_info = sf.iloc[idx].to_dict()
                        except Exception:
                            me_info = None
                except Exception:
                    # If no summary_frame, try to access .margeff attribute (array-like)
                    try:
                        me_array = getattr(margeff_obj, 'margeff', None)
                        me_var = None
                        if (me_array is not None) and (params is not None):
                            if var in params.index:
                                idx = list(params.index).index(var)
                                me_var = float(me_array[idx])
                                me_info = {"margeff": me_var}
                    except Exception:
                        me_info = None
        except Exception:
            me_info = None

        entry['marginal_effect'] = me_info
        results[var] = entry

    # Compose a human-readable description that explains what was extracted
    description_lines = [
        "Extracted coefficients, standard errors, p-values, 95% confidence intervals,",
        "and odds ratios (exp(coef)) for the main predictors testing the effects of",
        "relative group size and contest location on the probability that the focal",
        "group wins an intergroup contest. Interpretation guidance:",
        "- coef: change in log-odds of focal group winning per 1 SD increase in predictor (predictors were z-scored).",
        "- odds_ratio: multiplicative change in the odds of focal group winning per 1 SD increase (OR>1 -> higher odds).",
        "- p_value: statistical significance (commonly using alpha = 0.05).",
        "- or_ci_lower / or_ci_upper: 95% confidence interval for the odds ratio.",
        "- marginal_effect (if present): estimated change in predicted probability (dy/dx) at the sample mean.",
        "",
        "Returned object 'object' contains one entry per target variable (log_size_ratio_z,",
        "ProximityAdvantage_z, size_x_location) with numeric summaries, and 'nobs' giving sample size.",
        "Use the sign and magnitude of 'coef' (or 'odds_ratio') together with 'p_value' to judge",
        "whether a predictor increases or decreases the focal group's win probability and whether",
        "that effect is statistically significant at conventional thresholds."
    ]
    description = "\n".join(description_lines)

    return {"object": {**results, **extracted}, "description": description}