def extract_final_answer(model_output):
    """
    Extract key statistics for the predictors of interest from a fitted statsmodels
    MixedLMResults (or MixedLMResultsWrapper) object.

    Returns:
      {
        "object": {
            "age": { "coef": ..., "se": ..., "z": ..., "p": ..., "ci_lower": ..., "ci_upper": ...,
                     "exp_coef": ..., "exp_ci_lower": ..., "exp_ci_upper": ... },
            "Sex_m": { ... },
            "Help_y": { ... }
        },
        "description": "textual interpretation summarizing each predictor's effect and significance"
      }
    """
    import numpy as np
    # Names of predictors we care about
    predictors = ['age', 'Sex_m', 'Help_y']

    # Try to get fixed-effect parameter estimates, p-values and conf int robustly
    # Different statsmodels result objects expose attributes differently; handle common cases.
    # Prefer fe_params (explicit fixed effects) if available.
    try:
        params = getattr(model_output, 'fe_params')
    except Exception:
        try:
            params = getattr(model_output, 'params')
        except Exception:
            raise ValueError("Could not find fixed-effect parameters on the model_output object.")

    # Standard errors: try bse_fe or bse (or bse if fe-specific not present)
    se = None
    try:
        se = getattr(model_output, 'bse_fe')
    except Exception:
        try:
            se = getattr(model_output, 'bse')
        except Exception:
            # try to compute se from cov_params if available
            try:
                cov = model_output.cov_params()
                se = np.sqrt(np.diag(cov))
                # convert to a Series-like mapping if params has index
                if hasattr(params, 'index'):
                    se = dict(zip(params.index, se))
            except Exception:
                raise ValueError("Could not obtain standard errors from model_output.")

    # p-values: try pvalues attribute, fallback to model_output.pvalues or compute approximate z
    pvalues = None
    try:
        pvalues = getattr(model_output, 'pvalues')
    except Exception:
        try:
            pvalues = getattr(model_output, 'pvalues')  # redundant but safe
        except Exception:
            # attempt to compute z and p from params and se
            pvalues = None

    # confidence intervals
    try:
        ci = model_output.conf_int()
        # conf_int may return DataFrame or ndarray; try to index by param names
        if hasattr(ci, 'loc') and hasattr(ci, 'columns'):
            # DataFrame-like: rows indexed by param name, two columns [lower, upper]
            def get_ci(name):
                if name in ci.index:
                    row = ci.loc[name].values
                    return float(row[0]), float(row[1])
                else:
                    return (np.nan, np.nan)
        else:
            # ndarray-like: assume order matches params.index
            if hasattr(params, 'index'):
                idxs = list(params.index)
                def get_ci(name):
                    try:
                        i = idxs.index(name)
                        return float(ci[i, 0]), float(ci[i, 1])
                    except Exception:
                        return (np.nan, np.nan)
            else:
                def get_ci(name):
                    return (np.nan, np.nan)
    except Exception:
        # fallback: try to build CI using params +/- 1.96*se if se available
        def get_ci(name):
            try:
                pv = params[name]
                se_val = se[name] if hasattr(se, '__getitem__') else np.nan
                return float(pv - 1.96 * se_val), float(pv + 1.96 * se_val)
            except Exception:
                return (np.nan, np.nan)

    # Helper to get a value from params/se/pvalues whether they're Series, dict, or ndarray
    def safe_get(mapping, key):
        if mapping is None:
            return np.nan
        try:
            return mapping[key]
        except Exception:
            # try .get for dict-like
            try:
                return mapping.get(key, np.nan)
            except Exception:
                # try if mapping is ndarray and params has index
                try:
                    if hasattr(params, 'index'):
                        i = list(params.index).index(key)
                        return mapping[i]
                except Exception:
                    return np.nan

    results = {}
    summary_lines = []
    alpha = 0.05

    for pred in predictors:
        coef = safe_get(params, pred)
        se_val = safe_get(se, pred)
        pval = safe_get(pvalues, pred) if pvalues is not None else np.nan

        # compute z if possible
        try:
            z = float(coef) / float(se_val)
        except Exception:
            z = np.nan

        # if p-value missing but z available, compute two-tailed p
        if (pval is None or (isinstance(pval, float) and np.isnan(pval))) and not np.isnan(z):
            from scipy import stats
            pval = 2 * (1 - stats.norm.cdf(abs(z)))

        ci_lower, ci_upper = get_ci(pred)
        # exponentiated coefficient and CI to interpret on original nuts/sec scale
        try:
            exp_coef = float(np.exp(coef))
            exp_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else np.nan
            exp_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else np.nan
        except Exception:
            exp_coef = exp_ci_lower = exp_ci_upper = np.nan

        significance = (not np.isnan(pval)) and (pval < alpha)
        direction = "positive" if (not np.isnan(coef) and coef > 0) else ("negative" if (not np.isnan(coef) and coef < 0) else "no clear")

        results[pred] = {
            "coef": float(coef) if not np.isnan(coef) else np.nan,
            "se": float(se_val) if not np.isnan(se_val) else np.nan,
            "z": float(z) if not np.isnan(z) else np.nan,
            "p": float(pval) if not (pval is None) and not np.isnan(pval) else np.nan,
            "ci_lower": float(ci_lower) if not np.isnan(ci_lower) else np.nan,
            "ci_upper": float(ci_upper) if not np.isnan(ci_upper) else np.nan,
            "exp_coef": exp_coef,
            "exp_ci_lower": exp_ci_lower,
            "exp_ci_upper": exp_ci_upper,
            "significant_at_0.05": bool(significance),
            "direction": direction
        }

        # Build a readable summary line
        line = (
            f"{pred}: coef={results[pred]['coef']:.4g}, se={results[pred]['se']:.4g}, "
            f"z={results[pred]['z']:.3g}, p={results[pred]['p']:.3g}; "
            f"95% CI [{results[pred]['ci_lower']:.4g}, {results[pred]['ci_upper']:.4g}]. "
            f"exp(coef)={results[pred]['exp_coef']:.4g} (95% CI [{results[pred]['exp_ci_lower']:.4g}, {results[pred]['exp_ci_upper']:.4g}]). "
            f"{'Significant' if significance else 'Not significant'} at alpha={alpha}, direction={direction}."
        )
        summary_lines.append(line)

    description = (
        "Extracted fixed-effect estimates for predictors of nut-cracking efficiency (log scale).\n"
        + "\n".join(summary_lines)
        + "\n\nInterpretation notes: coefficients are on the natural-log scale of nuts/sec. "
        "exp(coef) gives the multiplicative effect on nuts/sec per unit increase in the predictor (per year for age; relative ratio for binary predictors). "
        f"Statistical significance judged at alpha={alpha}."
    )

    return {"object": results, "description": description}