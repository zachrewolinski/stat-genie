def extract_final_answer(model_output):
    """
    Extract statistics about the StudentTeacherRatio coefficient from a fitted statsmodels OLS result.

    Returns a dict with keys:
      - "object": a dictionary of numerical outputs (coef, se, t, p-value, 95% CI, standardized beta if computable,
                  nobs, r_squared, significance boolean, and a short categorical conclusion).
      - "description": a human-readable interpretation of the coefficient in context (direction and significance).
    """
    import numpy as np
    import pandas as pd

    # Basic validation
    if not hasattr(model_output, "params"):
        return {
            "object": None,
            "description": "Input does not appear to be a statsmodels RegressionResults object (missing .params)."
        }

    params = model_output.params

    # Find the parameter name corresponding to StudentTeacherRatio (case-insensitive, allowance for slight naming differences)
    target_lower = "studentteacherratio"
    matches = [name for name in params.index if name.lower() == target_lower]
    if not matches:
        # try looser match: contains both 'student' and 'ratio'
        matches = [name for name in params.index if ("student" in name.lower() and "ratio" in name.lower())]
    if not matches:
        return {
            "object": None,
            "description": "The fitted model does not include a parameter matching 'StudentTeacherRatio'."
        }

    name = matches[0]

    # Extract core statistics, guarding against missing attributes
    try:
        coef = float(params[name])
    except Exception:
        coef = None

    def safe_get(attr, key):
        try:
            val = getattr(model_output, attr)
            return float(val[key])
        except Exception:
            return None

    se = safe_get("bse", name)
    tval = safe_get("tvalues", name)
    pval = safe_get("pvalues", name)

    # 95% CI
    try:
        ci_df = model_output.conf_int()
        # conf_int() can return ndarray or DataFrame; handle both
        if isinstance(ci_df, (list, tuple, np.ndarray)):
            # convert to DataFrame with param names if possible
            ci_df = pd.DataFrame(ci_df, index=model_output.params.index)
        ci_lower = float(ci_df.loc[name, 0])
        ci_upper = float(ci_df.loc[name, 1])
        ci = [ci_lower, ci_upper]
    except Exception:
        ci = None

    # Sample size and R-squared if available
    nobs = int(model_output.nobs) if hasattr(model_output, "nobs") else None
    r_squared = float(model_output.rsquared) if hasattr(model_output, "rsquared") else None

    # Standardized (beta) coefficient if exog/endog accessible
    std_beta = None
    try:
        exog = model_output.model.exog
        exog_names = list(model_output.model.exog_names)
        if name in exog_names:
            idx = exog_names.index(name)
            x_std = float(np.std(exog[:, idx], ddof=0))
            y = model_output.model.endog
            y_std = float(np.std(y, ddof=0))
            if y_std != 0:
                std_beta = float(coef * x_std / y_std)
    except Exception:
        std_beta = None

    # Significance judgement at alpha = 0.05
    significant = (pval is not None and pval < 0.05)

    # Interpretation: recall StudentTeacherRatio higher => more students per teacher.
    if coef is None:
        interpretation_direction = "Coefficient unavailable."
    else:
        if coef < 0:
            interpretation_direction = (
                "Negative coefficient: higher student-teacher ratio (more students per teacher) is associated with LOWER AvgTestScore; "
                "consequently, a LOWER student-teacher ratio (fewer students per teacher) is associated with HIGHER AvgTestScore."
            )
        elif coef > 0:
            interpretation_direction = (
                "Positive coefficient: higher student-teacher ratio (more students per teacher) is associated with HIGHER AvgTestScore; "
                "consequently, a LOWER student-teacher ratio would be associated with LOWER AvgTestScore (opposite of the hypothesized direction)."
            )
        else:
            interpretation_direction = "Coefficient is zero (no association)."

    sig_text = (
        "This association is statistically significant at alpha=0.05."
        if significant else
        "This association is NOT statistically significant at alpha=0.05."
    )

    # Build description
    desc_parts = []
    desc_parts.append(f"Parameter examined: '{name}'.")
    if coef is not None:
        desc_parts.append(f"Estimate = {coef:.4f} (SE = {se:.4f}, t = {tval:.3f}, p = {pval:.4g}).")
    else:
        desc_parts.append("Estimate and standard errors could not be retrieved.")
    if ci is not None:
        desc_parts.append(f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}].")
    if std_beta is not None:
        desc_parts.append(f"Standardized (beta) = {std_beta:.4f}.")
    if nobs is not None:
        desc_parts.append(f"Sample size n = {nobs}.")
    if r_squared is not None:
        desc_parts.append(f"R-squared = {r_squared:.3f}.")

    desc_parts.append(interpretation_direction)
    desc_parts.append(sig_text)

    description = " ".join(desc_parts)

    result_object = {
        "param_name": name,
        "coef": coef,
        "se": se,
        "t_value": tval,
        "p_value": pval,
        "ci_95": ci,
        "standardized_beta": std_beta,
        "nobs": nobs,
        "r_squared": r_squared,
        "significant_0.05": significant,
        # short categorical conclusion helpful for programmatic checks:
        "conclusion": (
            "lower_ratio_associated_with_higher_performance"
            if (coef is not None and coef < 0 and significant)
            else ("opposite_direction_significant" if (coef is not None and coef > 0 and significant) else "no_statistical_evidence")
        )
    }

    return {"object": result_object, "description": description}