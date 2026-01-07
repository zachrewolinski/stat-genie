def extract_final_answer(model_output):
    """
    Extract statistics about the effect of the applicant gender indicator ('female')
    from a fitted statsmodels GLM/Logit results object.

    Returns:
      {
        "object": {
            "param_name": "female",
            "coef": float,               # log-odds coefficient for female
            "std_err": float,
            "z_or_t": float or None,     # test statistic if available
            "p_value": float,
            "conf_int": [lower, upper],  # 95% CI for coef (log-odds)
            "odds_ratio": float,
            "odds_ratio_ci": [lower_or, upper_or],
            "avg_pred_prob_female": float, # average predicted approval if female=1
            "avg_pred_prob_male": float,   # average predicted approval if female=0
            "average_marginal_effect": float # difference (female - male)
        },
        "description": "Plain-language interpretation of the result and significance"
      }
    """
    import numpy as np
    import pandas as pd

    # Helper to raise informative error
    def _err(msg):
        raise ValueError(msg)

    # Determine parameter name/index for 'female'
    param_name = 'female'

    # Try to extract coefficient, se, p-value and conf_int in robust ways
    # Statsmodels result objects commonly have .params, .bse, .pvalues, .conf_int()
    params = None
    try:
        params = model_output.params
    except Exception:
        _err("model_output does not expose .params. Provide a statsmodels Results object.")

    if isinstance(params, (pd.Series, pd.DataFrame)) and param_name in params.index:
        coef = float(params.loc[param_name])
    else:
        # try to find parameter by name in model.exog_names (if params is ndarray)
        exog_names = getattr(getattr(model_output, "model", None), "exog_names", None)
        if exog_names and param_name in exog_names:
            idx = list(exog_names).index(param_name)
            coef_array = np.asarray(params)
            coef = float(coef_array[idx])
        else:
            _err(f"Could not find parameter '{param_name}' in model_output.params or model.exog_names.")

    # Standard error
    try:
        bse = model_output.bse
        if isinstance(bse, (pd.Series, pd.DataFrame)) and param_name in bse.index:
            std_err = float(bse.loc[param_name])
        else:
            std_err = float(np.asarray(bse)[idx]) if 'idx' in locals() else float(np.asarray(bse)[0])
    except Exception:
        std_err = None

    # p-value
    try:
        pvalues = model_output.pvalues
        if isinstance(pvalues, (pd.Series, pd.DataFrame)) and param_name in pvalues.index:
            p_value = float(pvalues.loc[param_name])
        else:
            p_value = float(np.asarray(pvalues)[idx]) if 'idx' in locals() else float(np.asarray(pvalues)[0])
    except Exception:
        p_value = None

    # conf_int (95%)
    try:
        ci = model_output.conf_int()
        if isinstance(ci, (pd.DataFrame, pd.Series)) and param_name in getattr(ci, "index", []):
            ci_lower = float(ci.loc[param_name].iloc[0])
            ci_upper = float(ci.loc[param_name].iloc[1])
        else:
            ci_arr = np.asarray(ci)
            ci_lower = float(ci_arr[idx, 0]) if 'idx' in locals() else float(ci_arr[0, 0])
            ci_upper = float(ci_arr[idx, 1]) if 'idx' in locals() else float(ci_arr[0, 1])
        conf_int = [ci_lower, ci_upper]
    except Exception:
        conf_int = [None, None]

    # Odds ratio and its CI
    try:
        odds_ratio = float(np.exp(coef))
        odds_ratio_ci = [float(np.exp(conf_int[0])) if conf_int[0] is not None else None,
                         float(np.exp(conf_int[1])) if conf_int[1] is not None else None]
    except Exception:
        odds_ratio = None
        odds_ratio_ci = [None, None]

    # Compute average predicted probability for female=1 vs female=0 (average marginal effect)
    avg_prob_female = None
    avg_prob_male = None
    ame = None
    try:
        # get design matrix (exog) and names
        model = getattr(model_output, "model", None)
        exog = getattr(model, "exog", None)
        exog_names = getattr(model, "exog_names", None)
        if exog is None or exog_names is None:
            # try results.model.exog alternative
            exog = model_output.model.exog if hasattr(model_output, "model") and hasattr(model_output.model, "exog") else None
            exog_names = model_output.model.exog_names if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names") else None

        if exog is not None and exog_names is not None and param_name in exog_names:
            female_idx = list(exog_names).index(param_name)
            exog = np.asarray(exog).copy()
            exog_female1 = exog.copy()
            exog_female1[:, female_idx] = 1.0
            exog_female0 = exog.copy()
            exog_female0[:, female_idx] = 0.0

            pred1 = model_output.predict(exog_female1)
            pred0 = model_output.predict(exog_female0)

            # ensure numpy arrays
            pred1 = np.asarray(pred1)
            pred0 = np.asarray(pred0)

            avg_prob_female = float(np.nanmean(pred1))
            avg_prob_male = float(np.nanmean(pred0))
            ame = float(avg_prob_female - avg_prob_male)
    except Exception:
        # If anything fails, leave AME fields as None (we still return coefficient-based results)
        avg_prob_female = avg_prob_female
        avg_prob_male = avg_prob_male
        ame = ame

    # Construct plain-language description
    if coef is None:
        desc = "Could not extract the 'female' coefficient from the model output."
    else:
        sign = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
        significance = ""
        if p_value is not None:
            if p_value < 0.01:
                significance = "statistically significant at p < 0.01"
            elif p_value < 0.05:
                significance = "statistically significant at p < 0.05"
            elif p_value < 0.1:
                significance = "marginally significant at p < 0.1"
            else:
                significance = "not statistically significant"
        # Compose description
        desc = (
            f"The model coefficient for 'female' is {coef:.4f} (SE={std_err:.4f}) which "
            f"corresponds to an odds ratio of {odds_ratio:.3f} (95% CI: [{odds_ratio_ci[0]:.3f}, {odds_ratio_ci[1]:.3f}]). "
        )
        if p_value is not None:
            desc += f"The p-value is {p_value:.4g}, so the effect is {significance}. "
        else:
            desc += "No p-value available to assess statistical significance. "
        if ame is not None:
            desc += (f"On average, setting female=1 vs female=0 changes the predicted approval probability by "
                     f"{ame:.4f} (female avg prob={avg_prob_female:.4f}, male avg prob={avg_prob_male:.4f}).")
        else:
            desc += "Average marginal effect could not be computed from the provided model object."

    # Build object to return (numbers)
    result_object = {
        "param_name": param_name,
        "coef": float(coef) if coef is not None else None,
        "std_err": float(std_err) if std_err is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "conf_int": [float(conf_int[0]) if conf_int[0] is not None else None,
                     float(conf_int[1]) if conf_int[1] is not None else None],
        "odds_ratio": float(odds_ratio) if odds_ratio is not None else None,
        "odds_ratio_ci": [float(odds_ratio_ci[0]) if odds_ratio_ci[0] is not None else None,
                          float(odds_ratio_ci[1]) if odds_ratio_ci[1] is not None else None],
        "avg_pred_prob_female": avg_prob_female,
        "avg_pred_prob_male": avg_prob_male,
        "average_marginal_effect": ame
    }

    return {"object": result_object, "description": desc}