def extract_final_answer(model_output):
    """
    Extracts the estimated effect of HasChildren on the count of extramarital affairs
    from a fitted ZeroInflatedNegativeBinomial model (statsmodels wrapper).
    
    Returns:
      {
        "object": {
          "male": { "coef": ..., "se": ..., "z": ..., "p": ..., "95%_CI": (..., ...),
                    "IRR": ..., "IRR_95%_CI": (..., ...) },
          "female": { same keys as male (uses coef + interaction) },
          "notes": "Interpretation note about sign (negative => fewer affairs) and significance"
        },
        "description": "Brief explanation of what the numbers mean in context."
      }
    """
    import numpy as np
    import pandas as pd
    from math import sqrt
    from scipy.stats import norm

    # Helper to compute p-value from z
    def z_pvalue(z):
        return 2 * (1 - norm.cdf(abs(z)))

    # Attempt to obtain parameter names and values as a pandas Series
    try:
        params = pd.Series(model_output.params)
        # If the Series has integer index, try to get names from model object
        if params.index.dtype == "int":
            # try to use model parameter names if available
            if hasattr(model_output.model, "param_names"):
                params.index = list(model_output.model.param_names)
            elif hasattr(model_output, "param_names"):
                params.index = list(model_output.param_names)
    except Exception:
        # fallback: try to coerce to numpy then build Series with available names
        vals = np.asarray(model_output.params)
        names = None
        if hasattr(model_output.model, "param_names"):
            names = list(model_output.model.param_names)
        elif hasattr(model_output, "params_names"):
            names = list(model_output.params_names)
        params = pd.Series(vals, index=names)

    # Covariance matrix as DataFrame (if available)
    try:
        cov = model_output.cov_params()
        # If cov_params returns ndarray, convert to DataFrame with same index as params
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)
    except Exception:
        # If covariance not available, return partial info
        return {
            "object": None,
            "description": "Could not extract covariance matrix (cov_params) from model_output; cannot compute SEs, p-values, or CIs."
        }

    # Determine the names used for the count equation parameters.
    # Prefer model.model.exog_names (names for count equation).
    try:
        count_names = list(model_output.model.exog_names)
    except Exception:
        # Fallback: assume the count names are the subset of params before any 'inflate' or 'inflate_' prefix
        count_names = [n for n in params.index if not (str(n).startswith("inflate") or str(n).startswith("inflate_"))]

    # The two parameter names of interest in the count equation:
    child_name = None
    interaction_name = None

    # Try to find exact names for HasChildren and the interaction term.
    # Common possibilities: 'HasChildren', 'HasChildren_Female', 'HasChildren:Female', 'HasChildren*Female'
    candidates_child = ['HasChildren', 'haschildren', 'HasChildren[T.True]']
    candidates_inter = ['HasChildren_Female', 'HasChildren:Female', 'HasChildren*Female',
                        'HasChildren:Female', 'HasChildren_Female[T.1]', 'HasChildren:Female[T.1]']

    # Search among count_names (case-sensitive first, then lower-case)
    for n in count_names:
        if n in candidates_child:
            child_name = n
            break
    if child_name is None:
        for n in count_names:
            if str(n).lower() == 'haschildren'.lower():
                child_name = n
                break

    for n in count_names:
        if n in candidates_inter:
            interaction_name = n
            break
    if interaction_name is None:
        for n in count_names:
            if 'haschildren' in str(n).lower() and 'female' in str(n).lower():
                interaction_name = n
                break

    # If we still can't find names, try direct exact names in params
    if child_name is None and 'HasChildren' in params.index:
        child_name = 'HasChildren'
    if interaction_name is None and 'HasChildren_Female' in params.index:
        interaction_name = 'HasChildren_Female'

    # If still missing, we cannot compute the requested effects
    if child_name is None:
        return {
            "object": None,
            "description": "Could not locate a parameter named for 'HasChildren' in the model parameters. Available param names: "
                           + ", ".join(map(str, params.index.tolist()))
        }

    # Interaction may be absent (no moderator). If absent, assume effect is same for both sexes.
    has_interaction = interaction_name is not None and interaction_name in params.index

    # Extract coefficients
    try:
        coef_child = float(params.loc[child_name])
    except Exception:
        return {
            "object": None,
            "description": f"Found child param name '{child_name}' but could not extract its value."
        }

    coef_inter = 0.0
    if has_interaction:
        coef_inter = float(params.loc[interaction_name])

    # Compute male effect (Female=0) and female effect (Female=1)
    # Effects are on the log count scale: coef = log change in expected count.
    # We also compute IRR = exp(coef)
    effect = {}

    # For variance of linear combination, use covariance matrix entries
    def linear_combination_stats(names, coefs):
        """
        names: list of parameter names in the full param vector
        coefs: list of multipliers for those params (same length as names)
        returns (estimate, se, z, p, CI_lower, CI_upper)
        """
        est = sum(coefs[i] * float(params.loc[names[i]]) for i in range(len(names)))
        # variance = c' V c
        var = 0.0
        for i in range(len(names)):
            for j in range(len(names)):
                var += coefs[i] * coefs[j] * float(cov.loc[names[i], names[j]])
        se = sqrt(var) if var >= 0 else float('nan')
        z = est / se if se and not np.isnan(se) else float('nan')
        p = z_pvalue(z) if not np.isnan(z) else float('nan')
        ci_lo = est - norm.ppf(0.975) * se if not np.isnan(se) else (float('nan'))
        ci_hi = est + norm.ppf(0.975) * se if not np.isnan(se) else (float('nan'))
        return est, se, z, p, (ci_lo, ci_hi)

    # Male (Female=0): just HasChildren coefficient
    names_m = [child_name]
    coefs_m = [1.0]
    est_m, se_m, z_m, p_m, ci_m = linear_combination_stats(names_m, coefs_m)
    irr_m = np.exp(est_m)
    irr_ci_m = (np.exp(ci_m[0]), np.exp(ci_m[1]))

    effect['male'] = {
        "coef": est_m,
        "se": se_m,
        "z": z_m,
        "p_value": p_m,
        "95%_CI_coef": ci_m,
        "IRR": irr_m,
        "95%_CI_IRR": irr_ci_m
    }

    # Female (Female=1): HasChildren + HasChildren_Female (if interaction present)
    if has_interaction:
        names_f = [child_name, interaction_name]
        coefs_f = [1.0, 1.0]
        est_f, se_f, z_f, p_f, ci_f = linear_combination_stats(names_f, coefs_f)
        irr_f = np.exp(est_f)
        irr_ci_f = (np.exp(ci_f[0]), np.exp(ci_f[1]))
        effect['female'] = {
            "coef": est_f,
            "se": se_f,
            "z": z_f,
            "p_value": p_f,
            "95%_CI_coef": ci_f,
            "IRR": irr_f,
            "95%_CI_IRR": irr_ci_f
        }
    else:
        # No interaction: same effect as male
        effect['female'] = effect['male']
        effect['note_interaction'] = "No HasChildren x Female interaction found; effect is identical for females and males in the count equation."

    # Short interpretation about decrease/increase and significance
    def interpret(e):
        if np.isnan(e["coef"]):
            return "Estimate unavailable"
        direction = "decrease" if e["coef"] < 0 else ("increase" if e["coef"] > 0 else "no change")
        sig = "statistically significant" if (not np.isnan(e["p_value"]) and e["p_value"] < 0.05) else "not statistically significant"
        return f"Having children is associated with a {direction} in the expected count of extramarital affairs ({sig}, p={e['p_value']:.3f}). IRR={e['IRR']:.3f} (95% CI {e['95%_CI_IRR'][0]:.3f} to {e['95%_CI_IRR'][1]:.3f})."

    notes = {
        "male_interpretation": interpret(effect['male']),
        "female_interpretation": interpret(effect['female'])
    }

    return {
        "object": {
            "effects": effect,
            "notes": notes
        },
        "description": (
            "Extracted marginal effects of HasChildren from the ZINB count equation. "
            "Coefficients are on the log expected count scale. IRR = exp(coef) is the incidence rate ratio. "
            "For males (Female=0) the effect equals the HasChildren coefficient; for females (Female=1) it equals "
            "HasChildren + HasChildren_Female if the interaction term exists. Negative coef => fewer affairs; p-values indicate statistical significance."
        )
    }