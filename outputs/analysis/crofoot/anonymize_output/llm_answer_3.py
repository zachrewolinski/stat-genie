def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels binary logit results object
    (optionally with cluster-robust covariance results) and returns a summary
    dictionary with numerical results and an interpreted description.

    Returns:
      {
        "object": {
            "nobs": int,
            "prsquared": float (if available),
            "predictors": {
                "SizeRatio": {coef, se, pvalue, conf_low, conf_high, OR, OR_CI_low, OR_CI_high},
                "LocationAdvantage": { ... },
                "SizeRatio:LocationAdvantage": { ... }
            }
        },
        "description": "Text summary interpreting the effects (sign/direction/significance)."
      }
    """
    import numpy as np

    # Helper to safely get attribute or return None
    def _safe_getattr(obj, name, default=None):
        return getattr(obj, name) if hasattr(obj, name) else default

    # Extract core arrays
    try:
        params = model_output.params.copy()
    except Exception:
        raise ValueError("model_output has no .params attribute; ensure a fitted statsmodels results object was passed.")

    try:
        bse = model_output.bse.copy()
    except Exception:
        bse = None

    try:
        pvalues = model_output.pvalues.copy()
    except Exception:
        pvalues = None

    try:
        conf = model_output.conf_int()
    except Exception:
        conf = None

    # Terms of interest (names created by 'SizeRatio * LocationAdvantage' in formula)
    terms = ['SizeRatio', 'LocationAdvantage', 'SizeRatio:LocationAdvantage']

    predictors = {}
    for term in terms:
        if term in params.index:
            coef = float(params[term])
            se = float(bse[term]) if (bse is not None and term in bse.index) else None
            pval = float(pvalues[term]) if (pvalues is not None and term in pvalues.index) else None
            if conf is not None:
                # conf_int returns a DataFrame/ndarray with two columns [lower, upper]
                try:
                    ci_low = float(conf.loc[term][0])
                    ci_high = float(conf.loc[term][1])
                except Exception:
                    # try positional indexing if conf is ndarray-like
                    try:
                        idx = list(conf.index).index(term)
                        ci_low = float(conf.iloc[idx, 0])
                        ci_high = float(conf.iloc[idx, 1])
                    except Exception:
                        ci_low = None
                        ci_high = None
            else:
                ci_low = None
                ci_high = None

            # Odds ratio and its CI
            try:
                OR = float(np.exp(coef))
                OR_CI_low = float(np.exp(ci_low)) if ci_low is not None else None
                OR_CI_high = float(np.exp(ci_high)) if ci_high is not None else None
            except Exception:
                OR = OR_CI_low = OR_CI_high = None

            predictors[term] = {
                "coef": coef,
                "se": se,
                "pvalue": pval,
                "conf_low": ci_low,
                "conf_high": ci_high,
                "OR": OR,
                "OR_CI_low": OR_CI_low,
                "OR_CI_high": OR_CI_high,
            }
        else:
            predictors[term] = None  # term not in the model

    # Additional model-level info
    nobs = None
    try:
        nobs = int(model_output.nobs)
    except Exception:
        try:
            # fallback: length of model endog
            nobs = int(len(model_output.model.endog))
        except Exception:
            nobs = None

    prsquared = _safe_getattr(model_output, 'prsquared', None)
    llf = _safe_getattr(model_output, 'llf', None)

    # Build a human-readable description interpreting results
    lines = []
    lines.append("Summary of effects on probability that the focal group wins (focal perspective):")
    if nobs is not None:
        lines.append(f"- Number of observations: {nobs}")
    if prsquared is not None:
        lines.append(f"- McFadden's pseudo-R^2: {prsquared:.3f}")

    # Interpret each predictor
    for term in terms:
        info = predictors.get(term)
        if info is None:
            lines.append(f"- {term}: not included in the fitted model.")
            continue

        coef = info["coef"]
        p = info["pvalue"]
        OR = info["OR"]
        ci_low = info["conf_low"]
        ci_high = info["conf_high"]
        OR_ci_low = info["OR_CI_low"]
        OR_ci_high = info["OR_CI_high"]

        if p is None:
            sig_text = "p-value unavailable"
        else:
            sig_text = "statistically significant (p < 0.05)" if p < 0.05 else f"not statistically significant (p = {p:.3f})"

        # Direction
        if coef > 0:
            direction = "positive — higher values increase the log-odds (OR > 1)"
        elif coef < 0:
            direction = "negative — higher values decrease the log-odds (OR < 1)"
        else:
            direction = "no effect (coef = 0)"

        # Compose line
        line = f"- {term}: coef = {coef:.3f}, se = {info['se']:.3f} " if info['se'] is not None else f"- {term}: coef = {coef:.3f} "
        if p is not None:
            line += f", p = {p:.3f}; {sig_text}; {direction}."
        else:
            line += f"; {direction}."
        if OR is not None:
            line += f" Odds ratio = {OR:.3f}"
            if OR_ci_low is not None and OR_ci_high is not None:
                line += f" (95% CI {OR_ci_low:.3f}–{OR_ci_high:.3f})."
            else:
                line += "."
        lines.append(line)

    # Specific interpretation for interaction term
    interaction_info = predictors.get('SizeRatio:LocationAdvantage')
    if interaction_info is not None:
        p_int = interaction_info["pvalue"]
        coef_int = interaction_info["coef"]
        if p_int is not None and p_int < 0.05:
            if coef_int > 0:
                lines.append(
                    "- Interaction interpretation: The positive and significant interaction means the numerical advantage "
                    "(SizeRatio) has a stronger positive effect on focal group's chance of winning when the focal group is "
                    "closer to its home-range center (higher LocationAdvantage, i.e., on focal's home turf)."
                )
            else:
                lines.append(
                    "- Interaction interpretation: The negative and significant interaction means the numerical advantage "
                    "is less beneficial (or may even reverse) when the focal group is on its home turf; conversely, SizeRatio "
                    "benefits focal wins more when the contest location favors the other group."
                )
        else:
            lines.append(
                "- Interaction interpretation: No evidence that the effect of numerical advantage (SizeRatio) depends on contest location; "
                "interpret main effects directly (if significant)."
            )

    description = " ".join(lines)

    result_object = {
        "nobs": nobs,
        "prsquared": float(prsquared) if prsquared is not None else None,
        "llf": float(llf) if llf is not None else None,
        "predictors": predictors,
    }

    return {"object": result_object, "description": description}