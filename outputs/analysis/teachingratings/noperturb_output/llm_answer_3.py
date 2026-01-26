def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, p-value, 95% CI, sample size, R-squared (if available),
    and a brief interpretation for the 'beauty_z' coefficient from the models returned by the
    provided modeling function.

    Expects model_output to be a dict-like with keys:
      - 'baseline_HC3'
      - 'fe_with_clustered_se'

    Returns:
      {
        "object": { "<model_key>": { "coef": float, "se": float, "pval": float,
                                     "ci_lower": float, "ci_upper": float,
                                     "nobs": int_or_None, "rsquared": float_or_None,
                                     "percent_of_scale": float,
                                     "significant_0.05": bool
                                   }, ... },
        "description": "Human-readable summary string"
      }
    """
    def _get_ci_for_param(model, param):
        # Attempt to get confidence interval robustly whether conf_int() returns DataFrame or ndarray
        ci = model.conf_int()
        try:
            # DataFrame with index
            row = ci.loc[param]
            return float(row[0]), float(row[1])
        except Exception:
            # Fallback: positional lookup
            try:
                idx = list(model.params.index).index(param)
                row = ci.iloc[idx]
                return float(row[0]), float(row[1])
            except Exception:
                return None, None

    out = {}
    summary_lines = []
    model_keys = ['baseline_HC3', 'fe_with_clustered_se']

    for key in model_keys:
        model = model_output.get(key) if isinstance(model_output, dict) else None
        if model is None:
            summary_lines.append(f"Model '{key}' not found in model_output.")
            continue

        param = 'beauty_z'
        try:
            coef = float(model.params[param])
        except Exception:
            summary_lines.append(f"Model '{key}': coefficient for '{param}' not found.")
            continue

        # Standard error and p-value (these respect cov_type used when fitting)
        try:
            se = float(model.bse[param])
        except Exception:
            se = None
        try:
            pval = float(model.pvalues[param])
        except Exception:
            pval = None

        ci_low, ci_high = _get_ci_for_param(model, param)

        # Observations and R-squared if available
        try:
            nobs = int(model.nobs)
        except Exception:
            nobs = None
        try:
            rsq = float(model.rsquared)
        except Exception:
            rsq = None

        # Interpretation metrics
        # coef is change in evaluation score (1-5 scale) per 1 SD increase in beauty_z
        # percent_of_scale: percent of the full 1-5 range (which is 4 points) represented by coef
        percent_of_scale = None
        try:
            percent_of_scale = float(coef / 4.0 * 100.0)
        except Exception:
            pass
        significant = (pval is not None) and (pval < 0.05)

        out[key] = {
            "coef": coef,
            "se": se,
            "pval": pval,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "nobs": nobs,
            "rsquared": rsq,
            "percent_of_scale": percent_of_scale,
            "significant_0.05": significant
        }

        # Build a readable summary line for this model
        sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"
        ci_text = f"[{ci_low:.3f}, {ci_high:.3f}]" if (ci_low is not None and ci_high is not None) else "CI unavailable"
        se_text = f"{se:.3f}" if se is not None else "SE unavailable"
        n_text = f"N={nobs}" if nobs is not None else "N unavailable"
        rsq_text = f"R^2={rsq:.3f}" if rsq is not None else "R^2 unavailable"
        pct_text = f"{percent_of_scale:.2f}% of the 1-5 scale" if percent_of_scale is not None else "percent of scale unavailable"

        summary_lines.append(
            f"{key}: beauty_z coef = {coef:.3f} (SE = {se_text}), 95% CI = {ci_text}, p = {pval:.3g}. "
            f"{n_text}, {rsq_text}. This means a 1 SD increase in rated attractiveness is associated with a "
            f"{coef:.3f}-point change on the 1-5 evaluation scale ({pct_text}). Effect is {sig_text}."
        )

    description = " | ".join(summary_lines) if summary_lines else "No models processed."
    return {"object": out, "description": description}