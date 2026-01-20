def extract_final_answer(model_output):
    """
    Given model_output (a dict with keys like 'deaths_model' and/or 'damage_model'
    containing fitted statsmodels results objects), extract coefficients, SEs,
    p-values, and 95% CIs for the femininity-related independent variables
    (MasFem or MasFem_z and Female). Also compute the implied percent change
    in deaths (DV is log_deaths) and give a short interpretation about whether
    the results support the hypothesis that more feminine names lead to higher fatalities.

    Returns:
      {
        "object": { "<IV_name>": { "coef": ..., "se": ..., "pvalue": ...,
                                   "ci_lower": ..., "ci_upper": ...,
                                   "percent_change_in_deaths": ... }, ... },
        "description": "short interpretation string"
      }
    """
    import numpy as np

    # Prefer the deaths model (primary DV); fall back to damage_model if absent
    model = None
    if isinstance(model_output, dict):
        if 'deaths_model' in model_output and model_output['deaths_model'] is not None:
            model = model_output['deaths_model']
        elif 'damage_model' in model_output and model_output['damage_model'] is not None:
            model = model_output['damage_model']
    else:
        # If a single model object is passed directly
        model = model_output

    if model is None:
        return {
            "object": None,
            "description": "No model found in model_output (neither 'deaths_model' nor 'damage_model')."
        }

    # Helper to access attributes robustly
    params = getattr(model, "params", None)
    pvalues = getattr(model, "pvalues", None)
    bse = getattr(model, "bse", None)
    try:
        conf = model.conf_int(alpha=0.05)
    except Exception:
        conf = None

    if params is None or pvalues is None:
        return {
            "object": None,
            "description": "Model object does not expose params/pvalues; cannot extract statistics."
        }

    # Build a list of parameter names robustly
    param_names = None
    # 1) If params has an index (pandas Series)
    if hasattr(params, "index"):
        try:
            param_names = list(params.index)
        except Exception:
            param_names = None
    # 2) If model exposes param_names
    if param_names is None and hasattr(model, "param_names"):
        try:
            param_names = list(model.param_names)
        except Exception:
            param_names = None
    # 3) If model.model.exog_names exists (statsmodels results)
    if param_names is None and hasattr(model, "model") and hasattr(model.model, "exog_names"):
        try:
            param_names = list(model.model.exog_names)
        except Exception:
            param_names = None
    # 4) Fallback: generate generic param names based on length of params
    if param_names is None:
        try:
            length = len(params)
        except Exception:
            length = None
        if length is not None:
            param_names = [f"param_{i}" for i in range(length)]
        else:
            return {
                "object": None,
                "description": "Unable to determine parameter names from the model."
            }

    # Function to retrieve a statistic (params, bse, pvalues) by parameter name
    def get_stat(arr, name):
        if arr is None:
            return None
        # If arr is pandas-like Series/DataFrame with loc/index
        try:
            if hasattr(arr, "loc") and name in getattr(arr, "index", []):
                return float(arr.loc[name])
        except Exception:
            pass
        # If arr is dict-like
        try:
            if hasattr(arr, "get"):
                val = arr.get(name)
                if val is not None:
                    return float(val)
        except Exception:
            pass
        # If arr is ndarray-like, map by position
        try:
            pos = param_names.index(name)
            return float(arr[pos])
        except Exception:
            return None

    # Possible IV names used in the model code
    possible_mas = ['MasFem_z', 'MasFem']
    possible_mturk = ['MTurkMasFem_z', 'MTurkMasFem']
    possible_female = ['Female']

    ivs_found = []
    for name in possible_mas + possible_mturk + possible_female:
        if name in param_names:
            ivs_found.append(name)

    if not ivs_found:
        return {
            "object": {},
            "description": ("None of the expected femininity-related independent variables "
                            "(MasFem_z/MasFem/MTurkMasFem_z/MTurkMasFem/Female) were found in the model.")
        }

    results = {}
    for name in ivs_found:
        coef = get_stat(params, name)
        se = get_stat(bse, name)
        pval = get_stat(pvalues, name)

        # Extract confidence intervals if available
        ci_lower = ci_upper = None
        if conf is not None:
            try:
                # conf may be a DataFrame or ndarray
                if hasattr(conf, "loc") and name in getattr(conf, "index", []):
                    # DataFrame: columns 0 and 1
                    ci_lower = float(conf.loc[name, 0])
                    ci_upper = float(conf.loc[name, 1])
                else:
                    # assume ndarray in same order as param_names
                    pos = param_names.index(name)
                    ci_lower = float(conf[pos, 0])
                    ci_upper = float(conf[pos, 1])
            except Exception:
                ci_lower = ci_upper = None

        # Interpret coefficient on log_deaths: change in log outcome.
        # Convert to percent change in deaths: (exp(coef)-1)*100
        try:
            if coef is not None:
                pct_change = (np.exp(coef) - 1.0) * 100.0
            else:
                pct_change = None
        except Exception:
            pct_change = None

        results[name] = {
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "percent_change_in_deaths": pct_change
        }

    # Build a concise interpretation regarding the hypothesis:
    # Hypothesis: More feminine names -> fewer precautions -> higher deaths.
    # So we look for positive coefficients (and significance) on femininity measures
    interpretations = []
    alpha = 0.05
    for name, stats in results.items():
        pval = stats.get("pvalue")
        sig = (pval is not None and pval < alpha)
        coef = stats.get("coef")
        pct = stats.get("percent_change_in_deaths")
        pval_str = f"{pval:.3g}" if pval is not None else "NA"
        coef_str = f"{coef:.3g}" if coef is not None else "NA"
        pct_str = f"{pct:.1f}%" if pct is not None else "NA"
        if sig and coef is not None:
            if coef > 0:
                interpretations.append(
                    f"{name}: significant (p={pval_str}); coef={coef_str} -> "
                    f"associated with higher log_deaths (≈{pct_str} higher deaths). "
                    "This supports the hypothesis."
                )
            else:
                interpretations.append(
                    f"{name}: significant (p={pval_str}); coef={coef_str} -> "
                    f"associated with lower log_deaths (≈{pct_str} change). "
                    "This contradicts the hypothesis."
                )
        else:
            interpretations.append(
                f"{name}: not statistically significant (p={pval_str}); "
                "no strong evidence for an effect on log_deaths."
            )

    description = ("Extracted coefficients, standard errors, p-values, and 95% CIs for femininity-related "
                   "predictors from the chosen model (deaths_model preferred). Interpretation: " +
                   " ".join(interpretations))

    return {
        "object": results,
        "description": description
    }