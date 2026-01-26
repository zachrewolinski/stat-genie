def extract_final_answer(model_output):
    """
    Extract coefficient, SE, t-stat, p-value, and 95% CI for 'beauty_c' from
    the 'baseline' and 'adjusted' statsmodels RegressionResultsWrapper objects
    contained in model_output (a dict with keys 'baseline' and 'adjusted').

    Returns:
      {
        "object": {
          "baseline": { "coef":..., "se":..., "t":..., "p":..., "ci_lower":..., "ci_upper":..., "significant": bool },
          "adjusted": { ... },
          "conclusion": <textual yes/no style conclusion>
        },
        "description": <brief explanation of what was extracted and how to interpret it>
      }
    """
    import numpy as np

    def _get_result_dict(res):
        # Defensive checks
        if res is None:
            return None
        # Coefficient-related values (access by name)
        name = 'beauty_c'
        params = getattr(res, 'params', None)
        if params is None:
            raise ValueError("Provided result object has no 'params' attribute")
        # get numeric entries
        try:
            coef = float(params[name])
        except Exception:
            # fallback by position (rare)
            coef = float(params.iloc[list(params.index).index(name)])
        # standard error, t, p
        try:
            se = float(res.bse[name])
            t = float(res.tvalues[name])
            p = float(res.pvalues[name])
        except Exception:
            # fallback to positional indexing if named access fails
            idx = list(params.index).index(name)
            se = float(np.asarray(res.bse)[idx])
            t = float(np.asarray(res.tvalues)[idx])
            p = float(np.asarray(res.pvalues)[idx])

        # confidence interval: handle both ndarray and DataFrame-like returns
        ci = res.conf_int()
        if isinstance(ci, np.ndarray):
            idx = list(params.index).index(name)
            ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
        else:
            # assume DataFrame-like
            try:
                ci_lower, ci_upper = float(ci.loc[name, 0]), float(ci.loc[name, 1])
            except Exception:
                # last fallback: use positional index
                idx = list(params.index).index(name)
                ci_lower, ci_upper = float(np.asarray(ci)[idx, 0]), float(np.asarray(ci)[idx, 1])

        significant = (p < 0.05)

        # round for readability
        return {
            "coef": round(coef, 4),
            "se": round(se, 4),
            "t": round(t, 4),
            "p": round(p, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "significant": bool(significant)
        }

    # Accept either a dict-like model_output or a single model (treat as adjusted)
    if isinstance(model_output, dict):
        baseline_res = model_output.get('baseline')
        adjusted_res = model_output.get('adjusted')
    else:
        baseline_res = None
        adjusted_res = model_output

    baseline_stats = _get_result_dict(baseline_res) if baseline_res is not None else None
    adjusted_stats = _get_result_dict(adjusted_res) if adjusted_res is not None else None

    # Form a concise conclusion about whether beauty predicts higher evaluations
    if adjusted_stats is not None:
        if adjusted_stats["significant"]:
            conclusion = (
                f"In the adjusted model, 'beauty_c' has a positive and statistically "
                f"significant association with course evaluations (coef={adjusted_stats['coef']}, "
                f"p={adjusted_stats['p']}). This suggests that more attractive instructors "
                f"receive higher evaluations, controlling for listed covariates."
            )
        else:
            conclusion = (
                f"In the adjusted model, 'beauty_c' is not statistically significant "
                f"(coef={adjusted_stats['coef']}, p={adjusted_stats['p']}). This suggests no "
                f"robust evidence that attractiveness predicts higher evaluations after controls."
            )
    else:
        conclusion = "No adjusted model available to form a conclusion."

    return {
        "object": {
            "baseline": baseline_stats,
            "adjusted": adjusted_stats,
            "conclusion": conclusion
        },
        "description": (
            "Extracted the OLS estimate of 'beauty_c' from both baseline (beauty only) "
            "and adjusted (beauty + controls) models. Returned coefficient, cluster-robust "
            "standard error, t-stat, p-value, 95% confidence interval, and a boolean flag "
            "for significance at alpha=0.05. The 'conclusion' field states whether beauty "
            "appears to predict higher student evaluations in the adjusted model."
        )
    }