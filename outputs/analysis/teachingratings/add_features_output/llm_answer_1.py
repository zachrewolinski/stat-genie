def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of instructor beauty on evaluations
    from a statsmodels regression results object (ideally cluster-robust results).

    Returns:
      {
        "object": {
          "coef_beauty": float,
          "se_beauty": float,
          "t_beauty": float,
          "p_beauty": float,
          "ci95_beauty": [low, high],
          "coef_beauty_sq": float,
          "se_beauty_sq": float,
          "t_beauty_sq": float,
          "p_beauty_sq": float,
          "ci95_beauty_sq": [low, high],
          "marginal_at_mean": {
             "value": float,
             "se": float,
             "ci95": [low, high],
             "p_value": float
          },
          "marginal_at_1": { ... },
          "turning_point": {
             "value": float or None,
             "se": float or None,
             "ci95": [low, high] or None
          }
        },
        "description": str
      }
    """
    import numpy as np
    import pandas as pd
    from math import erf, sqrt

    res = model_output

    # Get params and parameter names robustly (statsmodels may return ndarray or Series)
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("model_output has no .params attribute; not a statsmodels result object")

    # Determine parameter names in order
    if hasattr(params, "index"):
        param_names = list(params.index)
    elif isinstance(params, (list, tuple)):
        param_names = list(params)
    elif isinstance(params, np.ndarray):
        # try to find names from the model or result attributes
        model = getattr(res, "model", None)
        if model is not None and hasattr(model, "exog_names"):
            param_names = list(model.exog_names)
        elif hasattr(res, "param_names"):
            param_names = list(res.param_names)
        elif hasattr(res, "names"):
            param_names = list(res.names)
        else:
            raise ValueError(
                "model_output.params is a numpy array and parameter names cannot be determined. "
                "Provide a results object with model.exog_names or res.param_names."
            )
    else:
        try:
            param_names = list(params)
        except Exception:
            raise ValueError("Could not determine parameter names from model_output.params")

    # Helper to extract a parameter-like value by name from various container types
    def get_by_name(container, name):
        if container is None:
            raise KeyError(f"Container is None; missing value for '{name}'")
        if hasattr(container, "loc") and name in container.index:
            return container.loc[name]
        if isinstance(container, dict):
            return container[name]
        if isinstance(container, np.ndarray):
            try:
                idx = param_names.index(name)
            except ValueError:
                raise KeyError(f"Parameter name '{name}' not found in parameter names")
            return container[idx]
        # pandas Series without .loc? try direct indexing
        try:
            return container[name]
        except Exception:
            # fallback: try mapping by index position
            try:
                idx = param_names.index(name)
                return list(container)[idx]
            except Exception:
                raise KeyError(f"Could not extract '{name}' from container of type {type(container)}")

    # Verify required parameter names exist in the parameter list
    if not {"Beauty", "Beauty_sq"}.issubset(set(param_names)):
        raise ValueError("model_output.params missing required coefficients 'Beauty' and/or 'Beauty_sq'")

    # Extract coefficients
    coef_beauty = float(get_by_name(params, "Beauty"))
    coef_beauty_sq = float(get_by_name(params, "Beauty_sq"))

    # Standard errors and p-values
    se_obj = getattr(res, "bse", None)
    pvals_obj = getattr(res, "pvalues", None)
    if se_obj is None or pvals_obj is None:
        raise ValueError("model_output missing .bse or .pvalues")

    se_beauty = float(get_by_name(se_obj, "Beauty"))
    se_beauty_sq = float(get_by_name(se_obj, "Beauty_sq"))
    p_beauty = float(get_by_name(pvals_obj, "Beauty"))
    p_beauty_sq = float(get_by_name(pvals_obj, "Beauty_sq"))

    # Confidence intervals (95%)
    try:
        ci = res.conf_int(alpha=0.05)
        # conf_int may be ndarray or DataFrame; handle both
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            ci_beauty = [float(ci.loc["Beauty", 0]), float(ci.loc["Beauty", 1])]
            ci_beauty_sq = [float(ci.loc["Beauty_sq", 0]), float(ci.loc["Beauty_sq", 1])]
        else:
            # ndarray, map index positions
            i_beauty = param_names.index("Beauty")
            i_beauty_sq = param_names.index("Beauty_sq")
            ci_beauty = [float(ci[i_beauty, 0]), float(ci[i_beauty, 1])]
            ci_beauty_sq = [float(ci[i_beauty_sq, 0]), float(ci[i_beauty_sq, 1])]
    except Exception:
        ci_beauty = [coef_beauty - 1.96 * se_beauty, coef_beauty + 1.96 * se_beauty]
        ci_beauty_sq = [coef_beauty_sq - 1.96 * se_beauty_sq, coef_beauty_sq + 1.96 * se_beauty_sq]

    # t-stats
    t_beauty = float(coef_beauty / se_beauty) if se_beauty != 0 else np.nan
    t_beauty_sq = float(coef_beauty_sq / se_beauty_sq) if se_beauty_sq != 0 else np.nan

    # Covariance matrix (needed for delta-method)
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        # try attribute name used by different wrappers
        if hasattr(res, "covariance"):
            cov = res.covariance
    if cov is None:
        # fallback: construct diagonal cov from se^2 (no covariances)
        # Build se array in param_names order
        se_array = np.array([float(get_by_name(se_obj, n)) for n in param_names], dtype=float)
        cov = pd.DataFrame(np.diag(se_array ** 2), index=param_names, columns=param_names)

    # Helper to safely extract covariance entries
    def cov_ij(i, j):
        if isinstance(cov, pd.DataFrame):
            return float(cov.loc[i, j])
        else:
            # ndarray: map indices
            ii = param_names.index(i)
            jj = param_names.index(j)
            return float(cov[ii, jj])

    # Marginal effect of Beauty on Eval:
    # derivative = beta_beauty + 2 * beta_beauty_sq * x
    # Compute at x = 0 (mean-centered) and at x = 1
    def marginal_at(x):
        val = coef_beauty + 2.0 * coef_beauty_sq * x
        # variance via delta method: Var(beta1 + 2 x beta2) = Var(beta1) + (2x)^2 Var(beta2) + 2*(2x)*Cov(beta1,beta2)
        var = cov_ij("Beauty", "Beauty") + (2.0 * x) ** 2 * cov_ij("Beauty_sq", "Beauty_sq") + 2.0 * (2.0 * x) * cov_ij("Beauty", "Beauty_sq")
        se_val = float(np.sqrt(var)) if var >= 0 else float(np.nan)
        ci_low = val - 1.96 * se_val
        ci_high = val + 1.96 * se_val
        # t-statistic and two-sided p using normal approx
        t_val = float(val / se_val) if se_val != 0 and not np.isnan(se_val) else None
        if t_val is None:
            p_val = None
        else:
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_val) / sqrt(2.0))))
        return {"value": float(val), "se": se_val, "ci95": [float(ci_low), float(ci_high)], "t": t_val, "p_value": float(p_val) if p_val is not None else None}

    marginal_mean = marginal_at(0.0)
    marginal_at_1 = marginal_at(1.0)

    # Turning point for quadratic (in terms of Beauty): -beta1 / (2*beta2)
    turning = {"value": None, "se": None, "ci95": None}
    if abs(coef_beauty_sq) > 0:
        tp = -coef_beauty / (2.0 * coef_beauty_sq)
        turning["value"] = float(tp)
        # delta method for variance of tp = -beta1/(2 beta2)
        # gradient g = [d/d beta1, d/d beta2] = [-1/(2 beta2), beta1/(2 beta2^2)]
        try:
            db1 = -1.0 / (2.0 * coef_beauty_sq)
            db2 = coef_beauty / (2.0 * (coef_beauty_sq ** 2))
            var_tp = (db1 ** 2) * cov_ij("Beauty", "Beauty") + (db2 ** 2) * cov_ij("Beauty_sq", "Beauty_sq") + 2.0 * db1 * db2 * cov_ij("Beauty", "Beauty_sq")
            se_tp = float(np.sqrt(var_tp)) if var_tp >= 0 else float(np.nan)
            turning["se"] = se_tp
            turning["ci95"] = [float(tp - 1.96 * se_tp), float(tp + 1.96 * se_tp)]
        except Exception:
            turning["se"] = None
            turning["ci95"] = None

    # Build output object
    out = {
        "coef_beauty": coef_beauty,
        "se_beauty": se_beauty,
        "t_beauty": t_beauty,
        "p_beauty": p_beauty,
        "ci95_beauty": ci_beauty,
        "coef_beauty_sq": coef_beauty_sq,
        "se_beauty_sq": se_beauty_sq,
        "t_beauty_sq": t_beauty_sq,
        "p_beauty_sq": p_beauty_sq,
        "ci95_beauty_sq": ci_beauty_sq,
        "marginal_at_mean": marginal_mean,
        "marginal_at_1": marginal_at_1,
        "turning_point": turning
    }

    # Compose human-readable description
    def significance(p):
        try:
            return "statistically significant (p < 0.05)" if (p is not None and p < 0.05) else "not statistically significant (p >= 0.05 or unknown)"
        except Exception:
            return "not statistically significant (p >= 0.05 or unknown)"

    desc_lines = []
    desc_lines.append(
        "The estimated linear effect of mean-centered Beauty on evaluations is "
        f"{coef_beauty:.4f} (SE = {se_beauty:.4f}, p = {p_beauty:.4f}); this is {significance(p_beauty)}."
    )
    desc_lines.append(
        "The quadratic term (Beauty_sq) is "
        f"{coef_beauty_sq:.4f} (SE = {se_beauty_sq:.4f}, p = {p_beauty_sq:.4f}); this is {significance(p_beauty_sq)}."
    )

    # Safe formatting for marginal p-values and SEs which may be None or NaN
    def fmt_val(v, fmt="{:.4f}"):
        try:
            if v is None:
                return "NA"
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return "NA"
            return fmt.format(v)
        except Exception:
            return "NA"

    desc_lines.append(
        "Because Beauty was mean-centered, the marginal effect of a one-unit increase in Beauty at the mean (Beauty=0) equals the linear coefficient: "
        f"{fmt_val(marginal_mean['value'])} (SE = {fmt_val(marginal_mean['se'])}, 95% CI = [{fmt_val(marginal_mean['ci95'][0])}, {fmt_val(marginal_mean['ci95'][1])}], p ≈ {fmt_val(marginal_mean['p_value'])})."
    )
    desc_lines.append(
        "The marginal effect at Beauty = +1 is "
        f"{fmt_val(marginal_at_1['value'])} (SE = {fmt_val(marginal_at_1['se'])}, 95% CI = [{fmt_val(marginal_at_1['ci95'][0])}, {fmt_val(marginal_at_1['ci95'][1])}])."
    )
    if turning["value"] is not None:
        desc_lines.append(
            "The quadratic turning point (where marginal effect = 0) is at Beauty = "
            f"{fmt_val(turning['value'])} (SE = {fmt_val(turning['se'])}, 95% CI = [{fmt_val(turning['ci95'][0])}, {fmt_val(turning['ci95'][1])}])."
        )
    else:
        desc_lines.append("No finite quadratic turning point could be computed (Beauty_sq ≈ 0).")

    desc_lines.append(
        "Interpretation: The linear coefficient tells how much the course evaluation score changes for a one-unit increase in mean-centered beauty at the mean beauty level. "
        "The quadratic term, if significant, indicates curvature (diminishing or increasing marginal returns)."
    )

    description = " ".join(desc_lines)

    return {"object": out, "description": description}