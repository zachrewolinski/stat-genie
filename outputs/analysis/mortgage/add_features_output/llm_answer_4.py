def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of gender on mortgage acceptance from
    a fitted statsmodels GLMResultsWrapper (or Results) object.

    Returns a dict with keys:
      - "object": dict containing numeric results for:
          * female_nonblack: effect of being female for non-Black applicants
            (this is the coefficient on 'female')
          * female_black: effect of being female for Black applicants
            (this is female + female_black; computed with correct SE using the
             covariance matrix)
          * n_obs: number of observations if available
      - "description": a short explanation of the meaning of the reported numbers
    """
    import math

    # Helper to compute two-sided p-value from z using error function (no external deps)
    def z_to_p(z):
        # z is a float; compute two-sided p-value
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return 2.0 * (1.0 - cdf)

    # Access parameters and related outputs
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = getattr(model_output, "pvalues", None)
        conf_int = getattr(model_output, "conf_int")() if callable(getattr(model_output, "conf_int", None)) else getattr(model_output, "conf_int", None)
        cov = model_output.cov_params()  # covariance matrix (DataFrame or ndarray)
    except Exception as e:
        raise RuntimeError(f"Could not extract required attributes from model_output: {e}")

    # Convert conf_int to a form we can index by param name
    # conf_int may be a DataFrame or ndarray; try to use .loc if possible
    def get_conf_interval(name):
        try:
            # If conf_int is a DataFrame-like with .loc
            return tuple(conf_int.loc[name])
        except Exception:
            # If conf_int is array-like, try to match by position using params.index
            try:
                idx = list(params.index).index(name)
                return (float(conf_int[idx, 0]), float(conf_int[idx, 1]))
            except Exception:
                return (None, None)

    # Gather basic info function for a single coefficient by name
    def coef_stats(name):
        if name not in params.index:
            raise KeyError(f"Parameter '{name}' not found in model parameters.")
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else None
        z = float(coef / se) if se is not None and se != 0 else None
        # prefer p-value reported by the results object if present; otherwise compute from z
        if pvalues is not None and name in pvalues.index:
            p = float(pvalues[name])
        else:
            p = float(z_to_p(abs(z))) if z is not None else None
        ci_low, ci_high = get_conf_interval(name)
        # Odds ratio and CI (on multiplicative scale)
        try:
            or_val = math.exp(coef)
            or_ci = (math.exp(ci_low), math.exp(ci_high))
        except Exception:
            or_val = None
            or_ci = (None, None)
        return {
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "ci_95": (ci_low, ci_high),
            "odds_ratio": or_val,
            "odds_ratio_ci_95": or_ci,
            "significant_at_0.05": (p is not None and p < 0.05)
        }

    # Stats for female (this represents effect for non-Black applicants)
    female_stats = coef_stats("female")

    # If interaction female_black exists, compute effect for Black applicants as female + female_black
    female_black_stats = None
    if "female_black" in params.index:
        coef_f = float(params["female"])
        coef_fb = float(params["female_black"])
        coef_sum = coef_f + coef_fb

        # Compute variance of the sum: Var(f) + Var(fb) + 2*Cov(f, fb)
        try:
            # cov may be DataFrame-like with .loc
            var_f = float(cov.loc["female", "female"])
            var_fb = float(cov.loc["female_black", "female_black"])
            cov_f_fb = float(cov.loc["female", "female_black"])
        except Exception:
            # fallback: cov as ndarray, map indices
            idx_f = list(params.index).index("female")
            idx_fb = list(params.index).index("female_black")
            var_f = float(cov[idx_f, idx_f])
            var_fb = float(cov[idx_fb, idx_fb])
            cov_f_fb = float(cov[idx_f, idx_fb])

        var_sum = var_f + var_fb + 2.0 * cov_f_fb
        se_sum = math.sqrt(var_sum) if var_sum >= 0 else None
        z_sum = float(coef_sum / se_sum) if se_sum is not None and se_sum != 0 else None
        p_sum = float(z_to_p(abs(z_sum))) if z_sum is not None else None
        # 95% CI using normal approximation
        ci_low = coef_sum - 1.96 * se_sum if se_sum is not None else None
        ci_high = coef_sum + 1.96 * se_sum if se_sum is not None else None
        try:
            or_sum = math.exp(coef_sum)
            or_ci_sum = (math.exp(ci_low), math.exp(ci_high))
        except Exception:
            or_sum = None
            or_ci_sum = (None, None)

        female_black_stats = {
            "coef": coef_sum,
            "se": se_sum,
            "z": z_sum,
            "p_value": p_sum,
            "ci_95": (ci_low, ci_high),
            "odds_ratio": or_sum,
            "odds_ratio_ci_95": or_ci_sum,
            "significant_at_0.05": (p_sum is not None and p_sum < 0.05)
        }

    # Try to extract sample size
    n_obs = None
    if hasattr(model_output, "nobs"):
        try:
            n_obs = int(model_output.nobs)
        except Exception:
            n_obs = None

    result_object = {
        "female_nonblack": female_stats,
        "female_black": female_black_stats,
        "n_obs": n_obs
    }

    # Build a short description suitable for interpretation
    desc_lines = []
    desc_lines.append("Extracted statistics for the effect of being female on mortgage acceptance.")
    desc_lines.append("female_nonblack: effect for non-Black applicants (coefficient on 'female').")
    if female_stats["odds_ratio"] is not None:
        desc_lines.append(
            f"  - Log-odds coef = {female_stats['coef']:.4f}, SE = {female_stats['se']:.4f}, "
            f"p = {female_stats['p_value']:.4g}. Odds ratio = {female_stats['odds_ratio']:.3f} "
            f"(95% CI {female_stats['odds_ratio_ci_95'][0]:.3f}, {female_stats['odds_ratio_ci_95'][1]:.3f})."
        )
    else:
        desc_lines.append(f"  - coef = {female_stats['coef']:.6g}, p = {female_stats['p_value']!s}")

    if female_black_stats is not None:
        if female_black_stats["odds_ratio"] is not None:
            desc_lines.append(
                "female_black: effect of being female among Black applicants (female + female_black)."
            )
            desc_lines.append(
                f"  - Log-odds coef = {female_black_stats['coef']:.4f}, SE = {female_black_stats['se']:.4f}, "
                f"p = {female_black_stats['p_value']:.4g}. Odds ratio = {female_black_stats['odds_ratio']:.3f} "
                f"(95% CI {female_black_stats['odds_ratio_ci_95'][0]:.3f}, {female_black_stats['odds_ratio_ci_95'][1]:.3f})."
            )
        else:
            desc_lines.append(f"female_black coef = {female_black_stats['coef']:.6g}, p = {female_black_stats['p_value']!s}")
    else:
        desc_lines.append("No 'female_black' interaction found in the model; female effect applies to all applicants as specified.")

    desc_lines.append("Interpretation: a negative coefficient (odds ratio < 1) indicates lower odds of acceptance for females; "
                      "a positive coefficient (odds ratio > 1) indicates higher odds. Statistical significance at alpha=0.05 is indicated in the output.")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}