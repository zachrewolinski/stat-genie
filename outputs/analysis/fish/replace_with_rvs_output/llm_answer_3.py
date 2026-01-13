def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (NegativeBinomial or Poisson).
    Returns a dict with:
      - "object": dict with coefficient table (coef, se, p, 95% CI), exponentiated coefficients (rate ratios)
                  baseline rate per hour (exp(intercept)), mean predicted rate per hour (if available),
                  and simple diagnostics (if attached to the results).
      - "description": short human-readable interpretation focusing on livebait and camper effects
                       (whether they significantly change fish-per-hour and by how much).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic parameter table
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("Provided model_output does not appear to be a statsmodels results object with .params")

    bse = getattr(res, "bse", None)
    pvalues = getattr(res, "pvalues", None)
    conf = None
    try:
        conf = res.conf_int()
    except Exception:
        # If conf_int fails, leave as None and handle below
        conf = None

    # Exponentiated coefficients (rate ratios) and exponentiated CIs if available
    try:
        rr = np.exp(params)
    except Exception:
        rr = None

    if conf is not None:
        conf_exp = np.exp(conf)
    else:
        conf_exp = None

    # Variables of interest (expected to be in the model)
    vars_of_interest = []
    # include intercept if present
    if "const" in params.index:
        vars_of_interest.append("const")
    # common covariates requested
    for v in ("livebait", "camper", "persons", "child"):
        if v in params.index:
            vars_of_interest.append(v)

    coef_table = {}
    for v in vars_of_interest:
        entry = {
            "coef": float(params[v]) if v in params.index else None,
            "se": float(bse[v]) if (bse is not None and v in bse.index) else None,
            "p_value": float(pvalues[v]) if (pvalues is not None and v in pvalues.index) else None,
            "ci_lower": float(conf.loc[v, 0]) if (conf is not None and v in conf.index) else None,
            "ci_upper": float(conf.loc[v, 1]) if (conf is not None and v in conf.index) else None,
            "rate_ratio": float(rr[v]) if (rr is not None and v in rr.index) else None,
            "rr_ci_lower": float(conf_exp.loc[v, 0]) if (conf_exp is not None and v in conf_exp.index) else None,
            "rr_ci_upper": float(conf_exp.loc[v, 1]) if (conf_exp is not None and v in conf_exp.index) else None,
        }
        coef_table[v] = entry

    # Baseline rate per hour implied by intercept (exp(intercept)).
    baseline_rate_per_hour = None
    if "const" in params.index:
        try:
            baseline_rate_per_hour = float(np.exp(params["const"]))
        except Exception:
            baseline_rate_per_hour = None

    # Mean predicted rate per hour from attached predictions if present
    mean_predicted_rate_per_hour = None
    try:
        preds = getattr(res, "predictions", None)
        if preds is not None and "predicted_rate_per_hour" in preds.columns:
            mean_predicted_rate_per_hour = float(preds["predicted_rate_per_hour"].mean())
    except Exception:
        mean_predicted_rate_per_hour = None

    # Diagnostics if attached to results
    diagnostics = getattr(res, "diagnostics", None)

    # Short interpretation focused on livebait and camper
    def interpret_var(v):
        if v not in coef_table:
            return f"{v}: not included in model."
        row = coef_table[v]
        if row["rate_ratio"] is None:
            return f"{v}: coefficient = {row['coef']:.4f} (p={row['p_value']:.4g})"
        p = row["p_value"]
        rr_val = row["rate_ratio"]
        ci_lo = row["rr_ci_lower"]
        ci_hi = row["rr_ci_upper"]
        sig = ("statistically significant" if (p is not None and p < 0.05) else "not statistically significant")
        direction = "increase" if rr_val > 1 else ("decrease" if rr_val < 1 else "no change")
        pct_change = (rr_val - 1) * 100.0
        ci_text = f" (95% CI for RR: {ci_lo:.3f}–{ci_hi:.3f})" if (ci_lo is not None and ci_hi is not None) else ""
        return (f"{v}: rate ratio = {rr_val:.3f}{ci_text}, p = {p:.3g} → {sig}. "
                f"Interpreted as a {pct_change:.1f}% {direction} in fish-per-hour when {v} = 1 vs 0.")

    desc_lines = []
    # Give a compact summary for livebait and camper
    for v in ("livebait", "camper"):
        if v in coef_table:
            desc_lines.append(interpret_var(v))

    # Add baseline and average predicted rate info
    if baseline_rate_per_hour is not None:
        desc_lines.append(f"Baseline (reference) rate implied by intercept: {baseline_rate_per_hour:.3f} fish per hour "
                          "(this is for covariates set to zero: livebait=0, camper=0, persons=0, child=0).")
    if mean_predicted_rate_per_hour is not None:
        desc_lines.append(f"Mean model-predicted fish-per-hour across observed trips: {mean_predicted_rate_per_hour:.3f} fish/hour.")

    # Add a note about diagnostics/overdispersion if available
    if diagnostics is not None and isinstance(diagnostics, dict):
        od_ratio = diagnostics.get("overdispersion_ratio", None)
        if od_ratio is not None:
            desc_lines.append(f"Data diagnostics: mean count = {diagnostics.get('mean_count')}, var = {diagnostics.get('var_count')}, "
                              f"variance/mean = {od_ratio:.3f} (values >>1 suggest overdispersion). Observations = {diagnostics.get('n_obs')}.")
        else:
            desc_lines.append("Model diagnostics attached to results.")

    description = " ".join(desc_lines) if desc_lines else "No interpretable summary could be constructed from the model output."

    # Construct the returned object
    return {
        "object": {
            "coef_table": coef_table,
            "baseline_rate_per_hour": baseline_rate_per_hour,
            "mean_predicted_rate_per_hour": mean_predicted_rate_per_hour,
            "diagnostics": diagnostics
        },
        "description": description
    }