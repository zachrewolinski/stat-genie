def extract_final_answer(model_output):
    """
    Extract coefficient, uncertainty, test stats, and a short interpretation
    for the 'beauty_z' predictor from a statsmodels RegressionResultsWrapper.

    Returns a dict with keys:
      - "object": dict with numeric results {coef, std_err, t_stat, p_value, ci_lower, ci_upper, n_obs, df_resid, significance}
      - "description": human-readable interpretation of the effect on the 1-5 teaching_eval scale
    """
    out = {"object": None, "description": ""}

    try:
        res = model_output

        # Basic checks
        if not hasattr(res, "params"):
            raise ValueError("model_output has no .params attribute (not a fitted statsmodels result).")

        param_names = list(res.params.index)
        if "beauty_z" not in param_names:
            raise ValueError("'beauty_z' not found among model parameters: " + ", ".join(param_names))

        # Extract estimates
        coef = float(res.params["beauty_z"])
        se = float(res.bse["beauty_z"]) if hasattr(res, "bse") and "beauty_z" in res.bse.index else None
        t_stat = float(res.tvalues["beauty_z"]) if hasattr(res, "tvalues") and "beauty_z" in res.tvalues.index else None
        p_value = float(res.pvalues["beauty_z"]) if hasattr(res, "pvalues") and "beauty_z" in res.pvalues.index else None

        # Confidence interval (robust to different return types)
        ci_lower = ci_upper = None
        if hasattr(res, "conf_int"):
            try:
                ci = res.conf_int()
                # ci can be a DataFrame/ndarray; handle both
                if hasattr(ci, "loc") and "beauty_z" in ci.index:
                    row = ci.loc["beauty_z"]
                    ci_lower, ci_upper = float(row[0]), float(row[1])
                else:
                    # assume rows align with params order
                    idx = param_names.index("beauty_z")
                    row = ci[idx]
                    ci_lower, ci_upper = float(row[0]), float(row[1])
            except Exception:
                ci_lower = ci_upper = None

        # Sample size and degrees of freedom
        n_obs = int(res.nobs) if hasattr(res, "nobs") else None
        df_resid = float(res.df_resid) if hasattr(res, "df_resid") else None

        # Simple significance label
        significance = None
        if p_value is not None:
            if p_value < 0.001:
                significance = "p < 0.001"
            elif p_value < 0.01:
                significance = "p < 0.01"
            elif p_value < 0.05:
                significance = "p < 0.05"
            else:
                significance = "not statistically significant (p >= 0.05)"

        obj = {
            "coef": coef,
            "std_err": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "df_resid": df_resid,
            "significance": significance,
        }

        # Interpretation: beauty_z is standardized, so coef is change in teaching_eval (1-5) per 1 SD increase in beauty
        if ci_lower is not None and ci_upper is not None and p_value is not None:
            descr = (
                f"The estimated effect of a one-standard-deviation increase in instructor attractiveness (beauty_z) "
                f"on the course teaching evaluation is {coef:.3f} points (SE = {se:.3f}, t = {t_stat:.2f}, p = {p_value:.3f}). "
                f"The 95% confidence interval is [{ci_lower:.3f}, {ci_upper:.3f}]. "
                f"This means a 1 SD higher beauty rating is associated with a {coef:.3f}-point change on the 1–5 evaluation scale. "
                f"Based on the p-value, this effect is {significance}."
            )
        else:
            descr = (
                f"Coefficient for beauty_z = {coef}. Additional statistics could not be fully extracted. "
                f"Details available in the 'object' field."
            )

        out["object"] = obj
        out["description"] = descr
        return out

    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to extract results for 'beauty_z': {e}"
        }