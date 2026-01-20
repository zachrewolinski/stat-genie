def extract_final_answer(model_output):
    """
    Extract statistics for the effect of beauty (beauty_z) from the model output.

    Parameters
    ----------
    model_output : dict-like
        Expected to contain at least:
        - 'baseline': statsmodels RegressionResultsWrapper (eval ~ beauty_z)
        - 'full_cluster': statsmodels RegressionResultsWrapper (eval ~ beauty_z + controls,
                          clustered SEs by prof)

    Returns
    -------
    dict with keys:
    - "object": dict with extracted numeric results for each model (coefficient,
                standard error, t-value, p-value, 95% CI, nobs, R-squared)
    - "description": human-readable summary interpreting the coefficient(s)
    """
    results_summary = {}
    descriptions = []

    def _extract_from_result(res, name):
        # Safely extract stats for 'beauty_z'
        out = {
            "model_name": name,
            "coefficient": None,
            "std_error": None,
            "t_value": None,
            "p_value": None,
            "ci_lower": None,
            "ci_upper": None,
            "nobs": None,
            "r_squared": None,
            "note": None
        }
        try:
            params = res.params
            if 'beauty_z' not in params.index:
                out["note"] = "beauty_z not in model parameters"
                return out
            # index position of beauty_z
            idx = list(params.index).index('beauty_z')

            out["coefficient"] = float(params.loc['beauty_z'])
            # bse, tvalues, pvalues should be available (they respect cov_type used in fitting)
            try:
                out["std_error"] = float(res.bse.loc['beauty_z'])
            except Exception:
                # fallback by index
                out["std_error"] = float(res.bse[idx])
            try:
                out["t_value"] = float(res.tvalues.loc['beauty_z'])
            except Exception:
                out["t_value"] = float(res.tvalues[idx])
            try:
                out["p_value"] = float(res.pvalues.loc['beauty_z'])
            except Exception:
                out["p_value"] = float(res.pvalues[idx])
            # confidence interval
            try:
                ci = res.conf_int()
                # ci might be DataFrame or ndarray
                try:
                    out["ci_lower"] = float(ci.loc['beauty_z'][0])
                    out["ci_upper"] = float(ci.loc['beauty_z'][1])
                except Exception:
                    # assume ndarray with same order as params
                    out["ci_lower"] = float(ci[idx, 0])
                    out["ci_upper"] = float(ci[idx, 1])
            except Exception:
                out["ci_lower"], out["ci_upper"] = (None, None)

            # nobs and R^2 if present
            try:
                out["nobs"] = int(res.nobs)
            except Exception:
                out["nobs"] = None
            try:
                out["r_squared"] = float(res.rsquared)
            except Exception:
                out["r_squared"] = None

        except Exception as e:
            out["note"] = f"error extracting info: {e}"
        return out

    # Expecting dict-like input
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output expected to be a dictionary with keys 'baseline' and 'full_cluster'."
        }

    # Extract for baseline and full_cluster when available
    for key in ['baseline', 'full_cluster']:
        if key in model_output and model_output[key] is not None:
            res = model_output[key]
            results_summary[key] = _extract_from_result(res, key)
        else:
            results_summary[key] = {"note": f"'{key}' not found in model_output."}

    # Build a concise interpretation string for beauty_z
    def interpret(entry):
        if entry is None or entry.get("coefficient") is None:
            return f"{entry.get('model_name', '')}: no estimate available."
        coef = entry["coefficient"]
        p = entry["p_value"]
        se = entry["std_error"]
        ci_l = entry["ci_lower"]
        ci_u = entry["ci_upper"]
        n = entry["nobs"]
        r2 = entry["r_squared"]

        signif = None
        if p is None:
            signif = "p-value unavailable"
        else:
            signif = "statistically significant (p < .05)" if p < 0.05 else "not statistically significant (p >= .05)"

        interp = (f"{entry['model_name']}: beauty_z coef = {coef:.3f}, SE = {se:.3f}, "
                  f"t = {entry['t_value']:.3f}, p = {p:.3f}. 95% CI [{ci_l:.3f}, {ci_u:.3f}]. "
                  f"N = {n}, R^2 = {r2:.3f}. This means a one standard-deviation increase in instructor "
                  f"beauty is associated with a {coef:.3f}-point change in teaching evaluation "
                  f"(scale 1–5). The effect is {signif}.")
        return interp

    # Create interpretations for available models
    for key in ['baseline', 'full_cluster']:
        entry = results_summary.get(key)
        if entry is None:
            descriptions.append(f"{key}: no results extracted.")
        elif entry.get("coefficient") is None:
            descriptions.append(f"{key}: {entry.get('note', 'no coefficient found')}")
        else:
            descriptions.append(interpret(entry))

    description_text = " | ".join(descriptions)

    return {
        "object": results_summary,
        "description": description_text
    }