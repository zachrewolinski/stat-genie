def extract_final_answer(model_output):
    """
    Extract key statistics for the hypothesis test from the supplied model output.
    Expects either:
      - a dict-like object with keys 'model_primary', optionally 'model_winsor', 'model_deaths'
        each mapping to a statsmodels RegressionResultsWrapper, or
      - a single statsmodels RegressionResultsWrapper (treated as the primary model).
    Returns a dictionary with:
      - "object": nested dict of numeric summaries for masfem_z and FemaleName for each available model
      - "description": a plain-language interpretation of the primary-model results relative to the hypothesis
    """
    import numpy as np
    import pandas as pd

    def summarize_result(res, varname):
        """Return summary stats for a variable in a statsmodels result object."""
        out = {"present": False}
        if res is None:
            return out
        params = res.params
        if varname not in params.index:
            return out
        out["present"] = True
        coef = float(params[varname])
        se = float(res.bse[varname]) if varname in res.bse.index else None
        t = float(res.tvalues[varname]) if varname in res.tvalues.index else None
        p = float(res.pvalues[varname]) if varname in res.pvalues.index else None
        # confidence interval (robust to conf_int returning ndarray or DataFrame)
        ci = res.conf_int()
        try:
            lower, upper = ci.loc[varname].astype(float).tolist()
        except Exception:
            # fallback: find position
            try:
                idx = list(params.index).index(varname)
                lower, upper = float(ci[idx, 0]), float(ci[idx, 1])
            except Exception:
                lower, upper = None, None
        # approximate percent change on original (1 + damage) scale:
        try:
            pct_change = (np.expm1(coef)) * 100.0  # percent
        except Exception:
            pct_change = None

        out.update({
            "coef": coef,
            "se": se,
            "t": t,
            "p": p,
            "ci_lower": lower,
            "ci_upper": upper,
            "approx_pct_change_in_1_plus_damage": pct_change
        })
        return out

    # Normalize model_output to a dict of models
    models = {}
    if model_output is None:
        raise ValueError("model_output is None")
    # If it's a dict-like with model_primary key, use that
    if isinstance(model_output, dict):
        # copy only expected keys
        for k in ["model_primary", "model_winsor", "model_deaths"]:
            models[k] = model_output.get(k, None)
    else:
        # assume it's a single statsmodels result -> treat as primary
        models["model_primary"] = model_output
        models["model_winsor"] = None
        models["model_deaths"] = None

    # Variables of interest
    vars_of_interest = ["masfem_z", "FemaleName"]

    summary = {}
    for key, res in models.items():
        if res is None:
            summary[key] = None
            continue
        summary[key] = {}
        for v in vars_of_interest:
            summary[key][v] = summarize_result(res, v)

    # Build a human-readable description focusing on the primary model
    prim = summary.get("model_primary")
    if prim is None:
        description = "No primary model found in model_output."
    else:
        # masfem_z interpretation
        m = prim.get("masfem_z", {})
        f = prim.get("FemaleName", {})
        lines = []
        if not m or not m.get("present", False):
            lines.append("Primary model: 'masfem_z' not present in model output.")
        else:
            coef = m["coef"]
            p = m["p"]
            se = m["se"]
            ci_lo = m["ci_lower"]
            ci_hi = m["ci_upper"]
            pct = m["approx_pct_change_in_1_plus_damage"]
            sig = ("statistically significant (p < 0.05)"
                   if (p is not None and p < 0.05) else
                   ("marginally significant (p < 0.10)" if (p is not None and p < 0.10) else "not statistically significant"))
            lines.append(
                "Primary model — masfem_z: coef = {coef:.4f}, SE = {se:.4f}, p = {p:.4g}, 95% CI [{lo:.4f}, {hi:.4f}]. "
                .format(coef=coef, se=se, p=p, lo=ci_lo, hi=ci_hi)
            )
            if pct is not None:
                lines.append(
                    "This implies approximately a {pct:.2f}% change in (1 + damage) per 1 SD increase in name femininity (exp(coef)-1 approximation). "
                    .format(pct=pct)
                )
            lines.append("Inference: " + sig + ".")
        # FemaleName interpretation
        if not f or not f.get("present", False):
            lines.append("Primary model: 'FemaleName' not present in model output.")
        else:
            coef = f["coef"]
            p = f["p"]
            se = f["se"]
            ci_lo = f["ci_lower"]
            ci_hi = f["ci_upper"]
            pct = f["approx_pct_change_in_1_plus_damage"]
            sig = ("statistically significant (p < 0.05)"
                   if (p is not None and p < 0.05) else
                   ("marginally significant (p < 0.10)" if (p is not None and p < 0.10) else "not statistically significant"))
            lines.append(
                "Primary model — FemaleName (binary): coef = {coef:.4f}, SE = {se:.4f}, p = {p:.4g}, 95% CI [{lo:.4f}, {hi:.4f}]. "
                .format(coef=coef, se=se, p=p, lo=ci_lo, hi=ci_hi)
            )
            if pct is not None:
                lines.append(
                    "This implies approximately a {pct:.2f}% difference in (1 + damage) for female vs male names (exp(coef)-1 approximation). "
                    .format(pct=pct)
                )
            lines.append("Inference: " + sig + ".")

        # Conclude relative to hypothesis
        # If masfem_z coef > 0 and significant -> supports hypothesis; if not -> does not support.
        if m and m.get("present", False) and (m.get("p") is not None):
            if (m["coef"] > 0) and (m["p"] < 0.05):
                lines.append("Net conclusion: The primary model provides statistical evidence consistent with the hypothesis (more feminine names → greater damage).")
            elif (m["coef"] > 0) and (m["p"] >= 0.05):
                lines.append("Net conclusion: The coefficient is positive (consistent with the hypothesis) but not statistically significant at conventional levels.")
            elif (m["coef"] < 0) and (m["p"] < 0.05):
                lines.append("Net conclusion: The primary model finds a statistically significant effect in the direction opposite the hypothesis (more feminine names → lower damage).")
            else:
                lines.append("Net conclusion: No statistically significant evidence supporting the hypothesis in the primary model.")
        description = " ".join(lines)

    return {"object": summary, "description": description}