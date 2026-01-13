def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM/GLMResultsWrapper object
    that modeled fish_caught with a log link and an offset (log_hours).

    Returns:
      {
        "object": dict -> mapping variable -> {
            "coef": float (log rate ratio),
            "se": float,
            "z": float (coef / se),
            "pvalue": float,
            "ci_lower": float,
            "ci_upper": float,
            "rate_ratio": float (exp(coef)),
            "rr_ci_lower": float,
            "rr_ci_upper": float
        },
        "description": str -> short interpretation of the results
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Validate expected attributes
    for attr in ("params", "bse", "pvalues", "conf_int"):
        if not hasattr(res, attr):
            raise ValueError(f"Model output missing required attribute: {attr}")

    params = res.params.copy()
    bse = res.bse.copy()
    pvalues = res.pvalues.copy()
    # conf_int may be a method or attribute depending on statsmodels version
    try:
        conf = res.conf_int()  # DataFrame: [[lower, upper], ...]
    except TypeError:
        conf = res.conf_int  # fallback if it's stored differently

    # Ensure conf is DataFrame-like with numeric columns
    conf = pd.DataFrame(conf, index=params.index)
    if conf.shape[1] < 2:
        raise ValueError("conf_int() did not return two columns (lower, upper).")

    ci_lower = conf.iloc[:, 0]
    ci_upper = conf.iloc[:, 1]

    # Build result table
    summary = {}
    for name in params.index:
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else float("nan")
        z = coef / se if se != 0 and not np.isnan(se) else float("nan")
        p = float(pvalues[name]) if name in pvalues.index else float("nan")
        cl = float(ci_lower[name]) if name in ci_lower.index else float("nan")
        cu = float(ci_upper[name]) if name in ci_upper.index else float("nan")
        rr = float(np.exp(coef))
        rr_cl = float(np.exp(cl)) if not np.isnan(cl) else float("nan")
        rr_cu = float(np.exp(cu)) if not np.isnan(cu) else float("nan")

        summary[name] = {
            "coef": coef,
            "se": se,
            "z": z,
            "pvalue": p,
            "ci_lower": cl,
            "ci_upper": cu,
            "rate_ratio": rr,
            "rr_ci_lower": rr_cl,
            "rr_ci_upper": rr_cu,
        }

    # Interpret notable results succinctly
    # Identify predictors (exclude intercept/const)
    intercept_names = set(["const", "Intercept", "intercept"])
    predictors = [n for n in params.index if n not in intercept_names]

    sig_preds = []
    weak_preds = []
    for p in predictors:
        pv = summary[p]["pvalue"]
        if np.isfinite(pv):
            if pv < 0.05:
                sig_preds.append(p)
            elif pv < 0.10:
                weak_preds.append(p)

    # Baseline rate per hour if intercept present
    baseline_txt = ""
    for iname in intercept_names:
        if iname in params.index:
            baseline_rate = float(np.exp(params[iname]))
            baseline_txt = (
                f"Baseline estimated catch rate (all predictors = 0): "
                f"{baseline_rate:.3f} fish/hour (exp(intercept)). "
            )
            break

    # Compose description
    desc_lines = []
    desc_lines.append(
        "Extracted coefficients (log rate ratios) and their exponentiated values (rate ratios = fish-per-hour multiplicative effects)."
    )
    if baseline_txt:
        desc_lines.append(baseline_txt)
    if sig_preds:
        desc_lines.append(
            "Statistically significant predictors at p < 0.05: "
            + ", ".join(sig_preds)
            + "."
        )
    if weak_preds:
        desc_lines.append(
            "Predictors with 0.05 <= p < 0.10 (weak evidence): "
            + ", ".join(weak_preds)
            + "."
        )
    if not sig_preds and not weak_preds:
        desc_lines.append("No predictors showed evidence of association at p < 0.10.")
    desc_lines.append(
        "For each predictor, 'rate_ratio' = exp(coef). A rate_ratio > 1 means higher fish-per-hour; < 1 means lower."
    )

    description = " ".join(desc_lines)

    return {"object": summary, "description": description}