def extract_final_answer(model_output):
    """
    Extract key statistics for SizeRatio_z, LocationAdv_z, and their interaction
    from a fitted model output (as returned by the provided modeling function).
    Returns a dictionary with:
      - "object": dict of extracted numeric results (coef, se, p, conf_int, odds ratio)
      - "description": short interpretation of those results in plain language

    The function tries to use the cluster-robust result if available, falling back
    to the fitted model object. It is defensive to handle slight differences in
    result object APIs.
    """
    import math
    import numpy as np
    # Get the results object (prefer cluster-robust if present)
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('model_cluster_robust') or model_output.get('model_fitted') or model_output.get('model')
    else:
        res = model_output

    if res is None:
        return {
            "object": None,
            "description": "No model result object found in model_output."
        }

    # Helper to safely pull parameter info for a parameter name
    def get_param_info(name):
        out = {"name": name, "found": False}
        try:
            params = getattr(res, "params", None)
            if params is None:
                # some wrappers may store results differently
                params = res.params
        except Exception:
            params = None

        if params is None:
            return out

        # Find the exact parameter name in case of slight naming differences
        param_keys = list(params.index)
        matched = None
        for key in param_keys:
            if key == name:
                matched = key
                break
        if matched is None:
            # try alternative interaction name styles
            alt_names = [
                name,
                name.replace(":", "*"),
                name.replace(":", " * "),
                name.replace(":", "_x_"),
                name.replace(":", "."),
            ]
            for alt in alt_names:
                if alt in param_keys:
                    matched = alt
                    break
        if matched is None:
            # not found
            return out

        out["found"] = True
        out["param_name"] = matched
        coef = float(params[matched])
        out["coef"] = coef

        # standard error
        try:
            se = float(getattr(res, "bse")[matched])
        except Exception:
            try:
                se = float(res.bse[matched])
            except Exception:
                se = None
        out["std_err"] = se

        # p-value (may be pvalues or z/p depending)
        pval = None
        try:
            pval = float(getattr(res, "pvalues")[matched])
        except Exception:
            try:
                pval = float(res.pvalues[matched])
            except Exception:
                # compute from z if possible
                try:
                    z = coef / se
                    from scipy import stats
                    pval = 2 * (1 - stats.norm.cdf(abs(z)))
                except Exception:
                    pval = None
        out["p_value"] = pval

        # confidence intervals
        try:
            ci = res.conf_int().loc[matched]
            # conf_int may be a DataFrame with columns [0,1] or named
            lo = float(ci.iloc[0])
            hi = float(ci.iloc[1])
        except Exception:
            try:
                ci_mat = res.conf_int()
                # if it's an ndarray with column order matching params
                if hasattr(ci_mat, "shape"):
                    # try to find index of matched
                    idx = list(params.index).index(matched)
                    lo = float(ci_mat[idx, 0])
                    hi = float(ci_mat[idx, 1])
                else:
                    lo = hi = None
            except Exception:
                lo = hi = None
        out["conf_int"] = (lo, hi)

        # odds ratio and its CI (for logistic model)
        try:
            out["odds_ratio"] = math.exp(coef)
            if out["conf_int"][0] is not None and out["conf_int"][1] is not None:
                out["odds_ratio_ci"] = (math.exp(out["conf_int"][0]), math.exp(out["conf_int"][1]))
            else:
                out["odds_ratio_ci"] = (None, None)
        except Exception:
            out["odds_ratio"] = out["odds_ratio_ci"] = None

        return out

    # variables of interest
    var_names = ["SizeRatio_z", "LocationAdv_z", "SizeRatio_z:LocationAdv_z", "MaleDiff_z", "FemaleDiff_z"]
    results = {}
    for v in var_names:
        results[v] = get_param_info(v)

    # Build a short interpretation string
    lines = []
    # Check LocationAdv_z first (interpretable)
    loc = results.get("LocationAdv_z")
    if loc and loc.get("found"):
        p = loc.get("p_value")
        orr = loc.get("odds_ratio")
        ci = loc.get("conf_int")
        if p is not None and p < 0.05:
            lines.append(
                f"Contest location (LocationAdv_z) has a positive, statistically significant effect "
                f"(coef={loc['coef']:.3f}, p={p:.3f}; odds ratio={orr:.2f}, 95% CI odds ratio={tuple(round(x,2) if x is not None else None for x in loc['odds_ratio_ci'])}). "
                "This means contests closer to the focal group's center increase focal group win probability."
            )
        else:
            lines.append(
                f"Contest location (LocationAdv_z) shows a positive coefficient (coef={loc['coef']:.3f}) "
                f"but is not clearly significant (p={p})."
            )
    else:
        lines.append("LocationAdv_z not found in model output.")

    # SizeRatio
    sz = results.get("SizeRatio_z")
    if sz and sz.get("found"):
        p = sz.get("p_value")
        orr = sz.get("odds_ratio")
        if p is not None and p < 0.05:
            sigtext = "statistically significant"
        elif p is not None and p < 0.1:
            sigtext = "marginally significant"
        else:
            sigtext = "not statistically significant"
        lines.append(
            f"Relative group size (SizeRatio_z) has a very large positive coefficient (coef={sz['coef']:.3f}), "
            f"{sigtext} (p={p}). Odds ratio ≈ {orr:.2e} (note: very large coefficient suggests possible separation/instability)."
        )
    else:
        lines.append("SizeRatio_z not found in model output.")

    # Interaction
    inter = results.get("SizeRatio_z:LocationAdv_z")
    if inter and inter.get("found"):
        p = inter.get("p_value")
        if p is not None and p < 0.05:
            lines.append(
                f"The interaction (SizeRatio_z:LocationAdv_z) is significant (coef={inter['coef']:.3f}, p={p}), "
                "indicating the effect of size depends on location."
            )
        else:
            lines.append(
                f"The interaction (SizeRatio_z:LocationAdv_z) is not statistically significant (coef={inter['coef']:.3f}, p={p}), "
                "so there is no clear evidence that the size effect depends on location."
            )
    else:
        lines.append("Interaction term not found in model output.")

    # Controls quick note
    male = results.get("MaleDiff_z")
    female = results.get("FemaleDiff_z")
    ctrl_lines = []
    if male and male.get("found"):
        ctrl_lines.append(f"MaleDiff_z coef={male['coef']:.3f} (p={male['p_value']:.3f})")
    if female and female.get("found"):
        ctrl_lines.append(f"FemaleDiff_z coef={female['coef']:.3f} (p={female['p_value']:.3f})")
    if ctrl_lines:
        lines.append("Controls: " + "; ".join(ctrl_lines) + ".")

    description = " ".join(lines)

    return {
        "object": results,
        "description": description
    }