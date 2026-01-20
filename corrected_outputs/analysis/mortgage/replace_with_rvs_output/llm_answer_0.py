def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator from a fitted statsmodels binary model.
    Returns a dictionary with:
      - "object": a dict containing coefficient, p-value, 95% CI, odds ratio and its 95% CI,
                  and an estimated change in predicted approval probability when female goes
                  from 0 to 1 (holding other covariates at their sample means when available).
      - "description": a short interpretation of those numbers in plain language.
    """
    import numpy as np

    res = model_output

    # Helper to get parameter index/name robustly
    try:
        param_names = list(res.params.index)
        params_is_series = True
    except Exception:
        # fallback: try to get names from model
        param_names = getattr(res.model, "exog_names", None)
        params_is_series = False
        if param_names is None:
            raise KeyError("Cannot determine parameter names from model output.")

    # Find the exact name used for the female variable (case-insensitive match)
    female_name = None
    for n in param_names:
        if str(n).lower() == "female":
            female_name = n
            break
    if female_name is None:
        raise KeyError("The model does not contain a parameter named 'female' (case-insensitive).")

    # Extract coefficient, p-value and confidence interval
    try:
        coef = float(res.params[female_name])
    except Exception:
        # if params is ndarray-like, find index
        idx = param_names.index(female_name)
        coef = float(np.asarray(res.params)[idx])

    # p-value
    try:
        p_value = float(res.pvalues[female_name])
    except Exception:
        idx = param_names.index(female_name)
        p_value = float(np.asarray(res.pvalues)[idx])

    # 95% CI for coefficient
    try:
        ci = res.conf_int().loc[female_name].astype(float)
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # conf_int may return ndarray aligned with param order
        ci_arr = np.asarray(res.conf_int())
        idx = param_names.index(female_name)
        ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Attempt to compute change in predicted probability when female: 0 -> 1,
    # holding other covariates at their sample means (requires access to exog)
    marginal_prob_change = None
    try:
        exog = np.asarray(res.model.exog)  # shape (n_obs, n_params)
        # compute mean of exog columns
        mean_exog = np.mean(exog, axis=0)
        # find index of female in model.exog_names (should match param_names)
        exog_names = list(res.model.exog_names)
        female_idx = exog_names.index(female_name)

        # prepare parameter vector as ndarray
        params_arr = np.asarray(res.params, dtype=float)

        # linear predictor at female = 0 (set female column to 0)
        x0 = mean_exog.copy()
        x0[female_idx] = 0.0
        eta0 = float(np.dot(x0, params_arr))
        p0 = 1.0 / (1.0 + np.exp(-eta0))

        # linear predictor at female = 1
        x1 = mean_exog.copy()
        x1[female_idx] = 1.0
        eta1 = float(np.dot(x1, params_arr))
        p1 = 1.0 / (1.0 + np.exp(-eta1))

        marginal_prob_change = float(p1 - p0)  # absolute change in probability (female - male)
        # also include relative change in odds as odds_ratio computed earlier
    except Exception:
        # If any of the above fails (e.g., exog not available), leave marginal_prob_change as None
        marginal_prob_change = None

    result_object = {
        "coef_logit_female": coef,
        "p_value_female": p_value,
        "95%_CI_coef": [ci_lower, ci_upper],
        "odds_ratio_female": odds_ratio,
        "95%_CI_odds_ratio": [or_ci_lower, or_ci_upper],
        "delta_prob_female_vs_male_at_means": marginal_prob_change,
        "n_obs": int(getattr(res, "nobs", res.model.endog.shape[0] if hasattr(res.model, "endog") else None))
    }

    # Plain-language description
    desc_parts = []
    desc_parts.append(
        f"The estimated log-odds coefficient for 'female' is {coef:.4f} "
        f"(p = {p_value:.4g}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}])."
    )
    desc_parts.append(
        f"This corresponds to an odds ratio of {odds_ratio:.3f} "
        f"(95% CI = [{or_ci_lower:.3f}, {or_ci_upper:.3f}])."
    )
    if marginal_prob_change is not None:
        desc_parts.append(
            f"Holding other covariates at their sample means, changing female from 0 to 1 "
            f"is associated with an absolute change in predicted approval probability of "
            f"{marginal_prob_change:.4f} (female – male)."
        )
    else:
        desc_parts.append(
            "Could not compute the change in predicted probability at covariate means (design matrix not available)."
        )

    # Interpretation guidance
    if p_value < 0.05:
        desc_parts.append("The effect is statistically significant at the 0.05 level.")
    else:
        desc_parts.append("The effect is not statistically significant at the 0.05 level.")

    description = " ".join(desc_parts)

    return {"object": result_object, "description": description}