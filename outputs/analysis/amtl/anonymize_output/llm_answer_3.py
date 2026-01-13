def extract_final_answer(model_output):
    """
    Extracts statistics for the 'is_human' predictor from a fitted statsmodels GLMResults/GLMResultsWrapper.
    Returns a dict with keys:
      - "object": dict of numeric results (coef, se, z, p, 95% CI, odds ratio and its 95% CI, significance boolean)
      - "description": short interpretation in the context of whether modern humans have higher AMTL

    Example return structure:
    {
      "object": {
        "coef": ...,
        "se": ...,
        "z": ...,
        "p_value": ...,
        "ci_95": [lower, upper],
        "odds_ratio": ...,
        "or_ci_95": [lower_or, upper_or],
        "significant": True/False
      },
      "description": "Interpretation text..."
    }
    """
    import numpy as np

    res = model_output

    # Ensure 'is_human' is present in the model results
    try:
        params = res.params
    except Exception as e:
        raise ValueError("The provided model_output does not appear to be a statsmodels results object.") from e

    if 'is_human' not in params.index:
        raise ValueError("'is_human' is not a parameter in the provided model output. Available params: " +
                         ", ".join(map(str, params.index.tolist())))

    # Extract core statistics
    coef = float(res.params['is_human'])
    se = float(res.bse['is_human'])
    # Wald z-statistic
    z = coef / se if se != 0 else np.nan
    p_value = float(res.pvalues['is_human'])

    # 95% confidence interval on the coefficient scale
    try:
        ci = res.conf_int().loc['is_human'].values.astype(float)
    except Exception:
        # fallback if conf_int returns numpy array
        ci = np.array(res.conf_int())[:,].astype(float)
        # try to locate index
        try:
            idx = list(res.params.index).index('is_human')
            ci = ci[idx]
        except Exception:
            raise RuntimeError("Could not extract confidence interval for 'is_human' from model output.")

    # Odds ratio and its CI (exponentiated)
    or_point = float(np.exp(coef))
    or_ci = list(np.exp(ci).astype(float))

    significant = (p_value < 0.05)

    # Directional interpretation
    if significant:
        if coef > 0:
            direction = "higher"
        elif coef < 0:
            direction = "lower"
        else:
            direction = "no difference"
    else:
        direction = "no statistically significant difference"

    description = (
        f"Controlling for age, sex, and tooth class, the 'is_human' coefficient = {coef:.4f} "
        f"(SE = {se:.4f}, z = {z:.3f}, p = {p_value:.3g}), 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]. "
        f"This corresponds to an odds ratio = {or_point:.3f} (95% CI = [{or_ci[0]:.3f}, {or_ci[1]:.3f}]). "
        f"Interpretation: modern humans have {direction} odds of antemortem tooth loss compared to the non-human primates in the sample."
    )

    result_object = {
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p_value,
        "ci_95": [float(ci[0]), float(ci[1])],
        "odds_ratio": or_point,
        "or_ci_95": [float(or_ci[0]), float(or_ci[1])],
        "significant": bool(significant)
    }

    return {"object": result_object, "description": description}