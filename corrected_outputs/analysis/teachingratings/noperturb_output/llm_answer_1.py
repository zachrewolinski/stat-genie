def extract_final_answer(model_output):
    """
    Extracts the estimated effect of instructor beauty on course evaluations
    from a fitted statsmodels OLS RegressionResultsWrapper (with cluster-robust SEs).

    Returns a dictionary with:
      - "object": a dict containing numeric results for:
            * male: effect of Beauty_z for male instructors (Gender_Female=0)
            * female: effect of Beauty_z for female instructors (Gender_Female=1)
            * interaction: coefficient on Beauty_x_GenderFemale (difference in slopes)
        Each sub-dict contains: effect, se, t, p, ci_lower, ci_upper.
      - "description": a plain-language interpretation of the numbers.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Parameter estimates and covariance matrix (cluster-robust, as fitted)
    params = res.params
    cov = res.cov_params()
    df_resid = getattr(res, "df_resid", None)

    # Helper to compute linear combination stats for any contrast given as dict {param_name: weight}
    def _linear_combination_stats(contrast):
        # Build weight vector aligned with params.index
        names = list(params.index)
        w = np.array([float(contrast.get(name, 0.0)) for name in names])
        # Compute effect (point estimate)
        param_vals = np.array([float(params[name]) for name in names])
        effect = float(w.dot(param_vals))
        # Variance and SE of linear combination
        cov_mat = np.array(cov)  # ensure ndarray
        var = float(w.dot(cov_mat).dot(w))
        se = float(np.sqrt(var)) if var >= 0 else float("nan")
        # t-stat and two-sided p-value using t distribution with residual df
        t_stat = float(effect / se) if se > 0 else float("nan")
        if df_resid is not None and np.isfinite(t_stat):
            p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
            t_crit = stats.t.ppf(1 - 0.025, df_resid)
        else:
            # fallback to normal approx if df unknown
            p_val = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if np.isfinite(t_stat) else float("nan")
            t_crit = stats.norm.ppf(1 - 0.025)
        ci_lower = float(effect - t_crit * se) if se >= 0 else float("nan")
        ci_upper = float(effect + t_crit * se) if se >= 0 else float("nan")
        return {
            "effect": effect,
            "se": se,
            "t": t_stat,
            "p": p_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    # Compute effects:
    # - For males (Gender_Female = 0): effect = coef(Beauty_z)
    male_stats = _linear_combination_stats({"Beauty_z": 1.0})
    # - Interaction term itself
    interaction_stats = _linear_combination_stats({"Beauty_x_GenderFemale": 1.0})
    # - For females (Gender_Female = 1): effect = coef(Beauty_z) + coef(Beauty_x_GenderFemale)
    female_stats = _linear_combination_stats({"Beauty_z": 1.0, "Beauty_x_GenderFemale": 1.0})

    # Prepare a concise description
    desc = (
        "Interpretation:\n"
        f"- Male instructors (Gender_Female=0): a 1 SD increase in standardized beauty (Beauty_z) "
        f"is associated with a change in course evaluation of {male_stats['effect']:.3f} points "
        f"(SE={male_stats['se']:.3f}, t={male_stats['t']:.2f}, p={male_stats['p']:.3f}, "
        f"95% CI [{male_stats['ci_lower']:.3f}, {male_stats['ci_upper']:.3f}]).\n"
        f"- Female instructors (Gender_Female=1): the corresponding change is "
        f"{female_stats['effect']:.3f} points (SE={female_stats['se']:.3f}, t={female_stats['t']:.2f}, "
        f"p={female_stats['p']:.3f}, 95% CI [{female_stats['ci_lower']:.3f}, {female_stats['ci_upper']:.3f}]).\n"
        f"- Interaction (difference in beauty effect for females vs males): "
        f"{interaction_stats['effect']:.3f} (SE={interaction_stats['se']:.3f}, t={interaction_stats['t']:.2f}, "
        f"p={interaction_stats['p']:.3f}, 95% CI [{interaction_stats['ci_lower']:.3f}, {interaction_stats['ci_upper']:.3f}]).\n\n"
        "Because Beauty_z was centered and standardized, these effects are changes in the 1-5 evaluation scale "
        "associated with a one standard-deviation increase in the panel-rated instructor attractiveness. "
        "P-values and confidence intervals use the cluster-robust covariance provided by the fitted model."
    )

    return {
        "object": {
            "male": male_stats,
            "female": female_stats,
            "interaction": interaction_stats,
        },
        "description": desc,
    }