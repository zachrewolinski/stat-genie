def extract_final_answer(model_output):
    """
    Extract key statistics about the effects of relative group size (rel_size_z),
    contest location advantage (dist_adv_z), and their interaction on the probability
    that the focal group wins (win) from a fitted statsmodels GLM result (or a
    robust-covariance-wrapped result).

    Returns:
      {
        "object": {
           "rel_size_z": {coef, se, z, p, ci_low, ci_high, or, or_ci_low, or_ci_high},
           "dist_adv_z": {...},
           "interaction": {...}
        },
        "description": "<brief explanation of extracted numbers and how to interpret them>"
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Ensure we have parameter access
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted results object with .params")

    params = res.params
    # Ensure params is a pandas Series
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Could not coerce res.params to a pandas Series")

    # bse might be available as .bse; if not, try to extract from cov_params
    if hasattr(res, "bse"):
        bse = res.bse
        # coerce to pandas Series aligned to params.index if possible
        if not isinstance(bse, pd.Series):
            try:
                bse = pd.Series(bse, index=params.index)
            except Exception:
                # fallback: try to index by position if sizes match
                try:
                    bse = pd.Series(list(bse), index=params.index)
                except Exception:
                    raise ValueError("Could not coerce res.bse to a pandas Series")
    else:
        try:
            cov = res.cov_params()  # may fail for some wrappers
            bse_vals = np.sqrt(np.diag(cov))
            bse = pd.Series(bse_vals, index=params.index)
        except Exception:
            raise ValueError("Could not obtain standard errors from model_output")

    # p-values
    if hasattr(res, "pvalues"):
        pvalues = res.pvalues
        if not isinstance(pvalues, pd.Series):
            try:
                pvalues = pd.Series(pvalues, index=params.index)
            except Exception:
                # leave as-is and will compute later if needed
                pass
    else:
        # compute z and p from coef and bse if necessary
        z_vals = params / bse
        try:
            from scipy import stats
            pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
        except Exception:
            # if scipy missing, use normal CDF via math.erf
            import math
            pvalues = 2 * (1 - 0.5 * (1 + np.vectorize(lambda x: math.erf(x / np.sqrt(2)))(np.abs(z_vals))))

    # Ensure pvalues is Series aligned to params.index if possible
    if not isinstance(pvalues, pd.Series):
        try:
            pvalues = pd.Series(pvalues, index=params.index)
        except Exception:
            pass

    # confidence intervals
    try:
        ci = res.conf_int()  # DataFrame-like with two columns
        ci_low = ci.iloc[:, 0]
        ci_high = ci.iloc[:, 1]
        # ensure Series types
        if not isinstance(ci_low, pd.Series):
            ci_low = pd.Series(ci_low, index=params.index)
        if not isinstance(ci_high, pd.Series):
            ci_high = pd.Series(ci_high, index=params.index)
    except Exception:
        # fallback: compute normal-approx 95% CI
        z_crit = 1.959963984540054
        ci_low = params - z_crit * bse
        ci_high = params + z_crit * bse
        ci_low = pd.Series(ci_low, index=params.index)
        ci_high = pd.Series(ci_high, index=params.index)

    # Helper to safely get parameter by trying possible names
    def get_term_names():
        # possible names for interaction and main effects
        names = {
            "rel": ["rel_size_z"],
            "dist": ["dist_adv_z"],
            "int": [
                "rel_size_z:dist_adv_z",
                "dist_adv_z:rel_size_z",
                "rel_size_z*dist_adv_z",
                "C(rel_size_z)[T.1]:C(dist_adv_z)[T.1]"
            ]
        }
        return names

    term_names = get_term_names()

    def find_name(candidates):
        for c in candidates:
            if c in params.index:
                return c
        # try partial match (in case variable names were modified)
        for idx in params.index:
            for c in candidates:
                if c in str(idx):
                    return idx
        return None

    out = {}
    for key, candidates in [("rel_size_z", term_names["rel"]),
                            ("dist_adv_z", term_names["dist"]),
                            ("interaction", term_names["int"])]:
        name = find_name(candidates)
        if name is None:
            out[key] = {
                "found": False,
                "message": f"Parameter for {key} not found in model params. Available params: {list(params.index)}"
            }
            continue

        # Safely extract numeric values; wrap in try to provide informative errors
        try:
            coef = float(params[name])
        except Exception:
            raise ValueError(f"Could not extract coefficient for parameter '{name}'")

        try:
            se = float(bse[name]) if (hasattr(bse, "index") and name in bse.index) else float(bse[name])
        except Exception:
            # try positional access if name not in index but lengths match
            try:
                idx = list(params.index).index(name)
                se = float(bse.iloc[idx]) if hasattr(bse, "iloc") else float(bse[idx])
            except Exception:
                raise ValueError(f"Could not extract standard error for parameter '{name}'")

        z = coef / se if se != 0 else np.nan

        try:
            p = float(pvalues[name]) if (hasattr(pvalues, "index") and name in pvalues.index) else float(pvalues[name])
        except Exception:
            # compute from z if necessary
            try:
                from scipy import stats
                p = float(2 * (1 - stats.norm.cdf(abs(z))))
            except Exception:
                import math
                p = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2))))) if not np.isnan(z) else np.nan

        try:
            ci_l = float(ci_low[name])
            ci_h = float(ci_high[name])
        except Exception:
            # fallback using positions
            try:
                idx = list(params.index).index(name)
                ci_l = float(ci_low.iloc[idx])
                ci_h = float(ci_high.iloc[idx])
            except Exception:
                raise ValueError(f"Could not extract confidence interval for parameter '{name}'")

        # Odds ratio and CI
        or_val = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_l))
        or_ci_high = float(np.exp(ci_h))

        out[key] = {
            "found": True,
            "param_name": str(name),
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "ci_95_low": ci_l,
            "ci_95_high": ci_h,
            "odds_ratio": or_val,
            "odds_ratio_ci_95_low": or_ci_low,
            "odds_ratio_ci_95_high": or_ci_high
        }

    # Build human-readable description explaining how to interpret results
    desc_lines = []
    desc_lines.append(
        "This output gives log-odds coefficients (coef), standard errors (se), z-statistics, p-values, "
        "95% confidence intervals, and odds ratios (exp(coef)) for the focal predictors."
    )
    desc_lines.append("Interpretation guidance:")
    desc_lines.append(
        "- rel_size_z: positive coef => as the focal group is larger relative to the other group, "
        "the log-odds of winning increase (odds_ratio > 1). A statistically significant p-value (commonly < 0.05) "
        "indicates evidence that relative group size affects winning probability."
    )
    desc_lines.append(
        "- dist_adv_z: positive coef => when the focal group is relatively closer to its home-range center "
        "(location advantage), the log-odds of winning increase (odds_ratio > 1)."
    )
    desc_lines.append(
        "- interaction (rel_size_z x dist_adv_z): a significant interaction means the effect of relative group size "
        "on winning depends on location advantage (and vice versa). The sign indicates direction: "
        "positive interaction -> the positive effect of being larger is stronger when the focal has location advantage."
    )
    desc_lines.append(
        "Check the p-values and 95% CIs: if a CI for a coefficient excludes 0 (or the CI for odds ratio excludes 1), "
        "the effect is statistically significant at ~5%."
    )

    return {
        "object": out,
        "description": " ".join(desc_lines)
    }