def extract_final_answer(model_output):
    """
    Extracts the Is_Human effect from model_output and returns a structured answer.

    Returns a dict with:
      - "object": dict with extracted statistics (coef, se, z, p, ci, odds_ratio, odds_ratio_ci, n_obs, n_specimens, conclusion)
      - "description": short human-readable interpretation of the result in the context of the question
    """
    import math
    import numpy as np

    out = {
        "object": None,
        "description": None
    }

    # Helper to safely fetch items
    def safe_get(d, key, default=None):
        try:
            return d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)
        except Exception:
            return default

    summary = safe_get(model_output, "summary_Is_Human", None)
    res = safe_get(model_output, "model_fit", None)
    n_obs = safe_get(model_output, "n_obs", None)
    n_specimens = safe_get(model_output, "n_specimens", None)

    coef = se = z = pval = ci_lower = ci_upper = or_est = or_ci = None

    # Prefer the provided precomputed summary if available
    if isinstance(summary, dict):
        coef = summary.get("coef_Is_Human")
        se = summary.get("se_Is_Human")
        z = summary.get("z_Is_Human")
        pval = summary.get("p_Is_Human")
        ci = summary.get("ci95_Is_Human")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        or_est = summary.get("odds_ratio_Is_Human") or summary.get("odds_ratio_Is_Human")
        or_ci = summary.get("odds_ratio_ci95")
    else:
        # Try to extract from the model result object
        try:
            params = getattr(res, "params", None)
            if params is not None and "Is_Human" in params:
                coef = float(params["Is_Human"])
        except Exception:
            coef = None

        try:
            # bse may be a Series or dict-like
            bse = getattr(res, "bse", None)
            if bse is not None and "Is_Human" in bse:
                se = float(bse["Is_Human"])
        except Exception:
            se = None

        try:
            pvals = getattr(res, "pvalues", None)
            if pvals is not None and "Is_Human" in pvals:
                pval = float(pvals["Is_Human"])
        except Exception:
            pval = None

        try:
            # conf_int might be a method or attribute
            conf = res.conf_int() if callable(getattr(res, "conf_int", None)) else getattr(res, "conf_int", None)
            if conf is not None:
                # conf could be a DataFrame-like indexable by label
                try:
                    row = conf.loc["Is_Human"]
                    ci_lower, ci_upper = float(row[0]), float(row[1])
                except Exception:
                    # fallback if conf is array-like and params order unknown, skip
                    pass
        except Exception:
            pass

        # compute z if possible
        try:
            if coef is not None and se not in (None, 0):
                z = float(coef / se)
                # if pval missing, approximate using normal dist
                if pval is None:
                    from scipy import stats
                    pval = float(2 * stats.norm.sf(abs(z)))
        except Exception:
            pass

        # odds ratio and its CI if we have coef and CI
        try:
            if coef is not None:
                or_est = float(math.exp(coef))
            if ci_lower is not None and ci_upper is not None:
                or_ci = [float(math.exp(ci_lower)), float(math.exp(ci_upper))]
        except Exception:
            pass

    # Final sanity: if we have coefficient and CI but not odds ratio, compute
    try:
        if or_est is None and coef is not None:
            or_est = float(np.exp(coef))
        if or_ci is None and ci_lower is not None and ci_upper is not None:
            or_ci = [float(np.exp(ci_lower)), float(np.exp(ci_upper))]
    except Exception:
        pass

    # Decide conclusion: do modern humans have higher AMTL?
    conclusion = None
    conclusion_text = None
    if coef is not None:
        # positive coef means higher log-odds (more AMTL) in humans
        direction = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
        if pval is not None:
            sig = pval < 0.05
            conclusion = bool(sig and coef > 0)
            if sig:
                conclusion_text = (
                    f"Yes. Modern humans have statistically significantly {direction} AMTL "
                    f"relative to non-human primates (coef={coef:.3f}, se={se:.3f}, z={z:.3f}, p={pval:.3g})."
                )
            else:
                conclusion_text = (
                    f"No statistically significant difference: estimated effect is {direction} but not significant "
                    f"(coef={coef:.3f}, se={se:.3f}, z={z:.3f}, p={pval:.3g})."
                )
        else:
            # no p-value available; rely on CI if present
            if ci_lower is not None and ci_upper is not None:
                # if CI on log-odds excludes 0 -> significant
                excludes_zero = (ci_lower > 0) or (ci_upper < 0)
                conclusion = bool(excludes_zero and coef > 0)
                if excludes_zero:
                    conclusion_text = (
                        f"Yes. Modern humans have {direction} AMTL; 95% CI for coef = [{ci_lower:.3f}, {ci_upper:.3f}] "
                        "does not include 0."
                    )
                else:
                    conclusion_text = (
                        f"No statistically significant difference based on 95% CI for coef = [{ci_lower:.3f}, {ci_upper:.3f}]."
                    )
            else:
                conclusion_text = "Cannot determine statistical significance: p-value and confidence interval not available."

    else:
        conclusion_text = "Model did not provide an Is_Human coefficient; cannot draw conclusion."

    # Build object result
    result_obj = {
        "coef_log_odds": None if coef is None else float(coef),
        "se": None if se is None else float(se),
        "z": None if z is None else float(z),
        "p_value": None if pval is None else float(pval),
        "ci95_log_odds": None if (ci_lower is None or ci_upper is None) else [float(ci_lower), float(ci_upper)],
        "odds_ratio": None if or_est is None else float(or_est),
        "odds_ratio_ci95": None if or_ci is None else [float(or_ci[0]), float(or_ci[1])],
        "n_observations": n_obs,
        "n_specimens_clustered": n_specimens,
        "conclusion_modern_humans_higher_AMTL": conclusion,
    }

    out["object"] = result_obj

    # Compose a concise description
    desc_parts = []
    if conclusion_text:
        desc_parts.append(conclusion_text)
    if result_obj["odds_ratio"] is not None and result_obj["odds_ratio_ci95"] is not None:
        desc_parts.append(
            f"Estimated odds of AMTL in modern humans are {result_obj['odds_ratio']:.2f} "
            f"(95% CI: {result_obj['odds_ratio_ci95'][0]:.2f}–{result_obj['odds_ratio_ci95'][1]:.2f})."
        )
    if n_obs is not None and n_specimens is not None:
        desc_parts.append(f"Model used {n_obs} tooth-class observations clustered by {n_specimens} specimens.")
    desc = " ".join(desc_parts) if desc_parts else "No additional information available."

    out["description"] = desc

    return out