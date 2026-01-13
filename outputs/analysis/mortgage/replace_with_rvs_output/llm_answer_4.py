def extract_final_answer(model_output):
    """
    Extracts statistics related to the 'female' coefficient from a fitted statsmodels Logit results object.

    Returns a dictionary with:
      - "object": a dict containing numeric estimates (coefficient, se, p-value, 95% CI,
                  odds ratio and its 95% CI, and average marginal effect if available).
      - "description": a short human-readable interpretation of the result in context,
                       which states whether the effect is statistically significant
                       (at the 5% level) and its direction.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic sanity checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (missing .params)")

    # Find the parameter name for female
    if "female" in res.params.index:
        varname = "female"
    else:
        # try to find a close match containing 'female'
        matches = [idx for idx in res.params.index if "female" in str(idx).lower()]
        if matches:
            varname = matches[0]
        else:
            raise KeyError("No coefficient named 'female' found in model_output.params")

    # Extract coefficient, se, p-value
    coef = float(res.params[varname])
    se = float(res.bse[varname]) if hasattr(res, "bse") else None
    pvalue = float(res.pvalues[varname]) if hasattr(res, "pvalues") else None

    # 95% confidence interval for coefficient
    try:
        ci_df = res.conf_int()
        ci_lower = float(ci_df.loc[varname, 0])
        ci_upper = float(ci_df.loc[varname, 1])
    except Exception:
        # fallback: compute using coef +/- 1.96*se if se available
        if se is not None:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            ci_lower = ci_upper = None

    # Odds ratio and its CI
    try:
        odds_ratio = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        odds_ratio = or_ci_lower = or_ci_upper = None

    # Try to compute average marginal effect (AME) for 'female' if available
    ame = ame_se = ame_p = None
    try:
        # get_margeff returns a MarginsResults; summary_frame is a DataFrame with dy/dx, Std.Err, P>|z|
        margeff = res.get_margeff()
        ame_df = margeff.summary_frame()
        # Index may contain varname; otherwise try to match similarly
        if varname in ame_df.index:
            ame = float(ame_df.loc[varname, "dy/dx"])
            # column label for std err may vary in versions; try common names
            if "Std. Err." in ame_df.columns:
                ame_se = float(ame_df.loc[varname, "Std. Err."])
            elif "Std. Err" in ame_df.columns:
                ame_se = float(ame_df.loc[varname, "Std. Err"])
            elif "std err" in ame_df.columns:
                ame_se = float(ame_df.loc[varname, "std err"])
            elif "Std. Error" in ame_df.columns:
                ame_se = float(ame_df.loc[varname, "Std. Error"])
            # p-value columns
            if "P>|z|" in ame_df.columns:
                ame_p = float(ame_df.loc[varname, "P>|z|"])
            elif "pvalue" in ame_df.columns:
                ame_p = float(ame_df.loc[varname, "pvalue"])
        else:
            # try case-insensitive match
            lowered = [idx.lower() for idx in ame_df.index.astype(str)]
            if varname.lower() in lowered:
                matched = ame_df.index[lowered.index(varname.lower())]
                ame = float(ame_df.loc[matched, "dy/dx"])
                # attempt to extract se and p similarly as above
                if "P>|z|" in ame_df.columns:
                    ame_p = float(ame_df.loc[matched, "P>|z|"])
    except Exception:
        # margin effects may not be available; ignore silently
        pass

    # Build numeric output object
    numeric_result = {
        "variable": varname,
        "coef": coef,
        "se": se,
        "p_value": pvalue,
        "95%_CI_coef": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "95%_CI_odds_ratio": [or_ci_lower, or_ci_upper],
        "average_marginal_effect": ame,
        "ame_se": ame_se,
        "ame_p_value": ame_p
    }

    # Create a short interpretation string based on p-value and sign
    def fmt(x):
        try:
            return f"{x:.4f}"
        except Exception:
            return str(x)

    if pvalue is None:
        significance_text = "p-value unavailable; cannot assess statistical significance."
    else:
        if pvalue < 0.01:
            sig_level = "p < 0.01"
        elif pvalue < 0.05:
            sig_level = "p < 0.05"
        elif pvalue < 0.1:
            sig_level = "p < 0.10"
        else:
            sig_level = None

        if sig_level is not None:
            direction = "increase" if coef > 0 else "decrease"
            significance_text = (
                f"The effect is statistically significant ({sig_level}). "
                f"Being female is associated with a {direction} in the log-odds of mortgage acceptance."
            )
        else:
            significance_text = (
                "The effect is not statistically significant at conventional levels (p >= 0.10). "
                "We cannot reject the null hypothesis of no effect of gender on mortgage acceptance."
            )

    # Add an interpretable magnitude statement using odds ratio and/or AME if available
    magnitude_parts = []
    if odds_ratio is not None:
        if odds_ratio > 1:
            magnitude_parts.append(
                f"Odds ratio = {fmt(odds_ratio)} (95% CI: [{fmt(or_ci_lower) if or_ci_lower is not None else None}, {fmt(or_ci_upper) if or_ci_upper is not None else None}])"
            )
        else:
            magnitude_parts.append(
                f"Odds ratio = {fmt(odds_ratio)} (95% CI: [{fmt(or_ci_lower) if or_ci_lower is not None else None}, {fmt(or_ci_upper) if or_ci_upper is not None else None}])"
            )
    if ame is not None:
        # AME is in probability units (change in probability)
        magnitude_parts.append(f"Average marginal effect = {fmt(ame)} (approx. {float(ame)*100:.2f} percentage points change in probability); ame p-value = {fmt(ame_p) if ame_p is not None else 'N/A'}")

    magnitude_text = " ".join(magnitude_parts) if magnitude_parts else "Magnitude metrics (odds ratio / AME) not available."

    description = (
        f"'female' coefficient = {fmt(coef)} (SE = {fmt(se) if se is not None else 'N/A'}, p = {fmt(pvalue) if pvalue is not None else 'N/A'}). "
        f"95% CI for coefficient = [{fmt(ci_lower) if ci_lower is not None else 'N/A'}, {fmt(ci_upper) if ci_upper is not None else 'N/A'}]. "
        f"{significance_text} {magnitude_text}"
    )

    return {"object": numeric_result, "description": description}