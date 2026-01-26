def extract_final_answer(model_output):
    """
    Extract coefficient estimates, robust SEs, z-stats, p-values, 95% CIs,
    and odds-ratios for the primary predictors from a fitted logistic
    regression results object (possibly wrapped as in the provided model code).

    Returns a dictionary with keys:
      - "object": dict mapping predictor name -> stats dict
      - "description": plain-language interpretation for the two focal predictors
    """
    import numpy as np
    from math import exp
    try:
        from scipy.stats import norm
    except Exception:
        # minimal fallback: approximate critical value for 95% CI
        class _Norm:
            @staticmethod
            def ppf(q):
                # 97.5% quantile ~= 1.95996398454005
                if abs(q - 0.975) < 1e-6:
                    return 1.95996398454005
                raise ValueError("Only 0.975 supported in fallback")
            @staticmethod
            def cdf(x):
                # not used in fallback below
                raise NotImplementedError
        norm = _Norm()

    res = model_output  # name used in prompt

    # Extract parameter estimates (should be a pandas Series)
    try:
        params = res.params
    except Exception as e:
        raise ValueError("Could not retrieve params from model_output") from e

    # Extract clustered covariance matrix (res.cov_params() or res.cov_params)
    try:
        cov = res.cov_params()
    except TypeError:
        # some objects expose cov_params as attribute (DataFrame/ndarray)
        try:
            cov = res.cov_params
        except Exception as e:
            raise ValueError("Could not retrieve covariance matrix from model_output") from e

    # Ensure numpy arrays
    param_index = list(params.index)
    coef = np.asarray(params, dtype=float)
    cov = np.asarray(cov, dtype=float)
    se = np.sqrt(np.diag(cov))

    # z-statistics and two-sided p-values using normal approximation
    z_stats = coef / se
    pvals = 2 * (1 - norm.cdf(np.abs(z_stats)))

    # 95% CI on log-odds scale
    z95 = norm.ppf(0.975)
    ci_lo = coef - z95 * se
    ci_hi = coef + z95 * se

    # Prepare output for focal predictors of interest
    focal_predictors = ['log_size_ratio_c', 'focal_home']
    stats_out = {}
    for name in focal_predictors:
        if name not in param_index:
            stats_out[name] = {
                "error": f"Predictor '{name}' not found in model parameters. Available: {param_index}"
            }
            continue
        i = param_index.index(name)
        b = float(coef[i])
        se_i = float(se[i])
        z_i = float(z_stats[i])
        p_i = float(pvals[i])
        ci_i = (float(ci_lo[i]), float(ci_hi[i]))
        # Odds ratio and CI
        or_val = exp(b)
        or_ci = (exp(ci_i[0]), exp(ci_i[1]))

        stats_out[name] = {
            "coef_log_odds": b,
            "robust_se": se_i,
            "z": z_i,
            "p_value": p_i,
            "95ci_log_odds": ci_i,
            "odds_ratio": or_val,
            "95ci_odds_ratio": or_ci,
        }

    # Short plain-language description interpreting sign and significance
    def interpret(name, info):
        if "error" in info:
            return info["error"]
        sig = info["p_value"] < 0.05
        direction = "positive" if info["coef_log_odds"] > 0 else ("zero" if abs(info["coef_log_odds"]) < 1e-12 else "negative")
        if name == 'log_size_ratio_c':
            desc = (
                f"log_size_ratio_c (focal group larger when positive): coefficient = {info['coef_log_odds']:.3f}, "
                f"SE = {info['robust_se']:.3f}, p = {info['p_value']:.3g}. This indicates a {direction} effect on the "
                f"log-odds of winning. Odds ratio = {info['odds_ratio']:.3f} (95% CI {info['95ci_odds_ratio'][0]:.3f}–{info['95ci_odds_ratio'][1]:.3f}). "
            )
        else:  # focal_home
            desc = (
                f"focal_home (1 = contest closer to focal group's center): coefficient = {info['coef_log_odds']:.3f}, "
                f"SE = {info['robust_se']:.3f}, p = {info['p_value']:.3g}. This indicates a {direction} effect of home advantage on "
                f"the log-odds of winning. Odds ratio = {info['odds_ratio']:.3f} (95% CI {info['95ci_odds_ratio'][0]:.3f}–{info['95ci_odds_ratio'][1]:.3f}). "
            )
        if sig:
            desc += "Effect is statistically significant at alpha = 0.05."
        else:
            desc += "Effect is NOT statistically significant at alpha = 0.05."
        return desc

    descriptions = {k: interpret(k, v) for k, v in stats_out.items()}

    # Combined human-readable description
    combined_description = (
        "Extracted statistics for primary predictors. See `object` for numeric results.\n"
        + "\n".join(f"{k}: {descriptions[k]}" for k in focal_predictors)
    )

    return {"object": stats_out, "description": combined_description}