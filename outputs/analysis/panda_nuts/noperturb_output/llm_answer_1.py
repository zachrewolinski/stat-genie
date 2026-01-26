def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, 95% confidence intervals, exponentiated effects
    (since the outcome is log-efficiency), and a short interpretation for the
    predictors of interest: 'age_c', 'sex_m', 'help_binary'.

    Returns a dict with keys:
      - "object": dict mapping each predictor -> stats dict (coef, p, ci, exp_coef, significant)
      - "description": human-readable summary interpreting the effects in context
    """
    import numpy as np
    from collections import OrderedDict

    # predictors of interest
    predictors = ['age_c', 'sex_m', 'help_binary']

    # Prepare result structure
    results = OrderedDict()

    # Basic checks
    if model_output is None:
        raise ValueError("model_output is None")
    # Attempt to access required attributes from statsmodels results
    try:
        params = model_output.params            # pandas Series
        pvalues = model_output.pvalues          # pandas Series
        conf = model_output.conf_int()          # DataFrame with two columns
    except Exception as e:
        raise ValueError(f"Provided model_output does not have expected attributes: {e}")

    # Extract stats for each predictor if present
    for pred in predictors:
        if pred in params.index:
            coef = float(params.loc[pred])
            pval = float(pvalues.loc[pred]) if pred in pvalues.index else None
            # conf_int might have columns [0,1] or named; use loc
            try:
                ci = conf.loc[pred].tolist()
                ci_lower, ci_upper = float(ci[0]), float(ci[1])
            except Exception:
                # fallback: try indexing by integer location
                ci_lower, ci_upper = (None, None)

            exp_coef = float(np.exp(coef))  # multiplicative change in nuts/sec per unit change
            significant = (pval is not None) and (pval < 0.05)

            # Interpretation text for this predictor
            if pval is None:
                interp = "No p-value available; cannot assess statistical significance."
            else:
                if significant:
                    if coef > 0:
                        direction = "associated with higher"
                    elif coef < 0:
                        direction = "associated with lower"
                    else:
                        direction = "no change in"
                    interp = (f"Statistically significant (p = {pval:.3g}). A one-unit increase in {pred} is "
                              f"{direction} nut-cracking efficiency. On the original efficiency scale, "
                              f"efficiency is multiplied by {exp_coef:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]).")
                else:
                    interp = (f"Not statistically significant (p = {pval:.3g}). Estimated effect: coef = {coef:.3f}, "
                              f"exp(coef) = {exp_coef:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]).")

            results[pred] = {
                "coefficient": coef,
                "p_value": pval,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "exp_coefficient": exp_coef,
                "significant": bool(significant),
                "interpretation": interp
            }
        else:
            results[pred] = {
                "error": f"Predictor '{pred}' not found in model parameters."
            }

    # Overall description: assemble brief summary lines
    lines = []
    for pred, info in results.items():
        if "error" in info:
            lines.append(f"{pred}: {info['error']}")
        else:
            sig_text = "significant" if info["significant"] else "not significant"
            lines.append(
                f"{pred}: coef={info['coefficient']:.3f}, exp(coef)={info['exp_coefficient']:.3f}, "
                f"95%CI=[{info['ci_lower']:.3f}, {info['ci_upper']:.3f}], p={info['p_value']:.3g} -> {sig_text}. "
                f"{info['interpretation']}"
            )

    description = ("Extracted fixed-effect estimates for predictors of interest (outcome = log(nuts/sec)). "
                   "Exponentiated coefficients give multiplicative effects on nuts-per-second. "
                   "Summary:\n- " + "\n- ".join(lines))

    return {"object": results, "description": description}