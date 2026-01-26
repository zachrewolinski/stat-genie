def extract_final_answer(model_output):
    """
    Extracts statistics for the 'is_female' coefficient from a statsmodels
    robust Logit result (a RobustResult returned by get_robustcov_results).

    Returns a dict with keys:
      - "object": dict of numeric results (coef, se, p_value, conf_int,
                  odds_ratio, odds_ratio_ci, significant, direction,
                  predicted_probability_difference_at_means)
      - "description": human-readable interpretation of the result in context
    """
    import numpy as np

    # Basic attribute checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not have .params (not a statsmodels result object).")

    params = model_output.params  # usually a pandas Series

    # Determine parameter names (try params.index, otherwise try model.exog_names)
    if hasattr(params, "index"):
        param_names = list(params.index)
    else:
        model = getattr(model_output, "model", None)
        exog_names = getattr(model, "exog_names", None)
        if exog_names:
            param_names = list(exog_names)
        else:
            raise ValueError("Cannot determine parameter names from model_output.params or model.exog_names.")

    # Ensure 'is_female' is present
    if 'is_female' not in param_names:
        raise ValueError("'is_female' is not a parameter in the provided model output.")

    # Helper to get a parameter-like value by name, handling Series or ndarray
    def get_by_name(container, name):
        if container is None:
            return None
        if hasattr(container, "index") and name in container.index:
            return container[name]
        # fallback: treat as sequence/ndarray and use param_names index
        try:
            idx = param_names.index(name)
            arr = np.asarray(container)
            return arr[idx]
        except Exception:
            return None

    # Extract main statistics
    # coef: prefer named access if params is Series, otherwise use index lookup
    coef_val = get_by_name(params, 'is_female')
    if coef_val is None:
        raise ValueError("Could not extract 'is_female' coefficient from params.")
    coef = float(coef_val)

    bse = getattr(model_output, "bse", None)
    pvalues = getattr(model_output, "pvalues", None)
    # Confidence interval (95%): try to extract from conf_int output
    try:
        conf = model_output.conf_int()  # DataFrame/array with CI rows matching params index
    except Exception:
        conf = None

    # standard error
    se_val = get_by_name(bse, 'is_female')
    se = float(se_val) if (se_val is not None) else None

    # p-value
    pval_val = get_by_name(pvalues, 'is_female')
    pval = float(pval_val) if (pval_val is not None) else None

    # Confidence interval extraction with support for DataFrame-like or ndarray
    if conf is not None:
        try:
            if hasattr(conf, "loc"):
                ci_low = float(conf.loc['is_female', 0])
                ci_high = float(conf.loc['is_female', 1])
            else:
                idx = param_names.index('is_female')
                conf_arr = np.asarray(conf)
                ci_low = float(conf_arr[idx, 0])
                ci_high = float(conf_arr[idx, 1])
        except Exception:
            ci_low, ci_high = None, None
    else:
        ci_low, ci_high = None, None

    # Odds ratio and its CI (if CI available)
    odds_ratio = float(np.exp(coef))
    odds_ratio_ci = (float(np.exp(ci_low)), float(np.exp(ci_high))) if (ci_low is not None and ci_high is not None) else (None, None)

    # Significance and direction
    alpha = 0.05
    significant = (pval is not None) and (pval < alpha)
    if coef > 0:
        direction = "women have higher odds of approval than men (positive coefficient)"
    elif coef < 0:
        direction = "women have lower odds of approval than men (negative coefficient)"
    else:
        direction = "no difference in odds (coefficient is zero)"

    # Optional: compute predicted probability difference at sample means of covariates
    prob_diff = None
    p_male_at_means = None
    p_female_at_means = None
    try:
        model = getattr(model_output, "model", None)
        if model is not None and hasattr(model, "exog") and hasattr(model, "exog_names"):
            exog = np.asarray(model.exog)
            exog_names = list(model.exog_names)
            # mean of each column
            mean_exog = np.mean(exog, axis=0)
            # locate index of 'is_female' in exog_names
            idx = exog_names.index('is_female')
            # create two feature vectors: female=1 and female=0
            x_male = mean_exog.copy()
            x_female = mean_exog.copy()
            x_male[idx] = 0.0
            x_female[idx] = 1.0
            # Ensure ordering of params matches exog_names: build param_vals in that order if possible
            try:
                # if params supports named access
                param_vals = np.asarray([get_by_name(params, name) for name in exog_names], dtype=float)
            except Exception:
                # fallback: use params as array (assumed to be in same order)
                param_vals = np.asarray(params, dtype=float)
            lin_male = float(np.dot(x_male, param_vals))
            lin_female = float(np.dot(x_female, param_vals))
            invlogit = lambda z: 1.0 / (1.0 + np.exp(-z))
            p_male_at_means = invlogit(lin_male)
            p_female_at_means = invlogit(lin_female)
            prob_diff = float(p_female_at_means - p_male_at_means)
    except Exception:
        prob_diff = None

    # Prepare object to return
    result_object = {
        "coef": coef,
        "se": se,
        "p_value": pval,
        "conf_int": (ci_low, ci_high),
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": odds_ratio_ci,
        "significant_at_0.05": significant,
        "direction_interpretation": direction,
        "predicted_probability_difference_at_means": prob_diff,
        "predicted_prob_male_at_means": p_male_at_means,
        "predicted_prob_female_at_means": p_female_at_means
    }

    # Human-readable description
    descr_parts = []
    try:
        descr_parts.append(f"'is_female' coefficient (log-odds): {coef:.6f}")
    except Exception:
        descr_parts.append(f"'is_female' coefficient (log-odds): {coef}")
    if se is not None:
        descr_parts.append(f"SE = {se:.6f}")
    if pval is not None:
        descr_parts.append(f"p-value = {pval:.4g} ({'significant' if significant else 'not significant'} at α={alpha})")
    if (ci_low is not None) and (ci_high is not None):
        descr_parts.append(f"95% CI (log-odds) = ({ci_low:.6f}, {ci_high:.6f})")
        descr_parts.append(f"Odds ratio = {odds_ratio:.3f}, 95% CI = ({odds_ratio_ci[0]:.3f}, {odds_ratio_ci[1]:.3f})")
    else:
        descr_parts.append(f"Odds ratio = {odds_ratio:.3f} (CI not available)")

    descr_parts.append(direction)
    if prob_diff is not None:
        descr_parts.append(
            f"Holding other covariates at their sample means, predicted approval probability is "
            f"{p_female_at_means:.3f} for women vs {p_male_at_means:.3f} for men (difference = {prob_diff:.3f})."
        )

    description = " ".join(descr_parts)

    return {"object": result_object, "description": description}