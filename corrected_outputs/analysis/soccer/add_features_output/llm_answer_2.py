def extract_final_answer(model_output):
    """
    Extracts statistics for the SkinDark coefficient from a fitted statsmodels GLM result,
    using clustered covariance if attached to the model object. Returns a dictionary with:
      - "object": dict of numeric results (coef, se_cluster, z, p, 95% CI on log scale and rate ratio scale)
      - "description": brief interpretation answering whether darker-skinned players
                       are more likely to receive red cards.
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    res = model_output

    # Try to get parameter estimates
    try:
        params = res.params.copy()
    except Exception as e:
        raise ValueError("Could not extract params from model_output: %s" % e)

    # Identify the SkinDark parameter name (exact or containing substring)
    target_name = None
    if "SkinDark" in params.index:
        target_name = "SkinDark"
    else:
        # fallback: find any param containing 'Skin' or 'skin'
        for name in params.index:
            if "Skin" in name or "skin" in name:
                target_name = name
                break

    if target_name is None:
        raise KeyError("Could not find a parameter named 'SkinDark' (or similar) in model params: %s" % list(params.index))

    # Obtain clustered covariance if available, otherwise use model covariance
    cov = None
    if hasattr(res, "cov_cluster"):
        cov = res.cov_cluster
    else:
        # try cov_params() method
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

    # Compute clustered robust standard errors (or fall back to model's bse)
    if cov is None:
        # fallback: use res.bse if available
        if hasattr(res, "bse"):
            bse_series = res.bse
            if target_name not in bse_series.index:
                raise KeyError("Target parameter %s not in res.bse index" % target_name)
            se_cluster = float(bse_series[target_name])
        else:
            raise ValueError("No covariance or bse available on model_output to compute standard errors.")
    else:
        # cov might be ndarray, DataFrame, or Series-like
        if isinstance(cov, np.ndarray):
            # assume order matches params.index
            idx = list(params.index).index(target_name)
            var = float(cov[idx, idx])
            se_cluster = float(np.sqrt(var))
        else:
            # assume cov has .loc or indexed rows/cols
            try:
                var = float(cov.loc[target_name, target_name])
                se_cluster = float(np.sqrt(var))
            except Exception:
                # try as dict-like numeric access
                try:
                    var = float(cov[target_name][target_name])
                    se_cluster = float(np.sqrt(var))
                except Exception as e:
                    raise ValueError("Could not extract variance for %s from covariance object: %s" % (target_name, e))

    # Extract coefficient and compute z, p-value, CI
    coef = float(params[target_name])
    z = coef / se_cluster if se_cluster != 0 else float("nan")
    p_two = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else float("nan")
    ci_lower = coef - 1.96 * se_cluster
    ci_upper = coef + 1.96 * se_cluster

    # Exponentiate to get rate ratio (since model is log-link NB: exp(coef) = multiplicative change in red-card rate per game)
    rate_ratio = float(np.exp(coef))
    rr_ci_lower = float(np.exp(ci_lower))
    rr_ci_upper = float(np.exp(ci_upper))

    # Decision: is there evidence that dark-skinned players receive more red cards?
    # Use two-sided p<0.05 and coef>0 to conclude "more likely".
    if np.isnan(p_two):
        conclusion = "Could not compute p-value / test statistic."
    else:
        if p_two < 0.05:
            if coef > 0:
                conclusion = "Yes: statistically significant evidence (p < 0.05) that darker-skinned players receive red cards at a higher rate."
            elif coef < 0:
                conclusion = "No (statistically significant in the opposite direction): darker-skinned players receive red cards at a lower rate."
            else:
                conclusion = "No effect (coefficient essentially zero)."
        else:
            conclusion = "No: there is no statistically significant difference in red-card rates by skin tone (p >= 0.05)."

    result_object = {
        "param_name": target_name,
        "coef_log_rate": coef,
        "se_cluster": se_cluster,
        "z_cluster": z,
        "pvalue_cluster": p_two,
        "ci_log_lower_95": ci_lower,
        "ci_log_upper_95": ci_upper,
        "rate_ratio": rate_ratio,
        "rate_ratio_ci_lower_95": rr_ci_lower,
        "rate_ratio_ci_upper_95": rr_ci_upper,
    }

    description = (
        f"Coefficient for {target_name} (log-rate scale): {coef:.4f} (SE clustered = {se_cluster:.4f}), "
        f"z = {z:.3f}, p = {p_two:.3g}. 95% CI on log scale = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"Rate ratio (exp(coef)) = {rate_ratio:.3f} with 95% CI = [{rr_ci_lower:.3f}, {rr_ci_upper:.3f}]. "
        f"Interpretation: {conclusion}  (Rate ratio >1 means higher red-card rate for darker-skinned players vs lighter-skinned players.)"
    )

    return {"object": result_object, "description": description}