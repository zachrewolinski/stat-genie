def extract_final_answer(model_output):
    """
    Extracts statistics for the 'student_teacher_ratio' coefficient from a fitted
    statsmodels RegressionResultsWrapper and returns an interpretable summary.

    Returns a dict with keys:
      - "object": dict containing numeric outputs (coef, se, t, p, 95% CI, standardized effect, nobs)
      - "description": plain-English interpretation focused on whether a lower
                       student-teacher ratio is associated with higher academic performance.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Preferred variable name
    var_name = 'student_teacher_ratio'

    # Try to locate the variable name robustly
    params_index = list(res.params.index)
    if var_name not in params_index:
        # try to find a matching name that contains the substring
        matches = [n for n in params_index if var_name in n]
        if len(matches) == 1:
            var_name = matches[0]
        elif len(matches) > 1:
            # pick the exact match if present, else first match
            exact = [n for n in matches if n == 'student_teacher_ratio']
            var_name = exact[0] if exact else matches[0]
        else:
            raise KeyError(f"Variable '{var_name}' not found in model params. Available params: {params_index}")

    coef = float(res.params[var_name])
    se = float(res.bse[var_name]) if hasattr(res, 'bse') else None
    tstat = float(res.tvalues[var_name]) if hasattr(res, 'tvalues') else None
    pval = float(res.pvalues[var_name]) if hasattr(res, 'pvalues') else None

    # Confidence interval (95%)
    try:
        ci_low, ci_high = res.conf_int().loc[var_name].astype(float).tolist()
    except Exception:
        ci = res.conf_int()
        # fallback if conf_int doesn't have labeled index
        try:
            idx = params_index.index(var_name)
            ci_low, ci_high = float(ci.iloc[idx, 0]), float(ci.iloc[idx, 1])
        except Exception:
            ci_low, ci_high = None, None

    # Number of observations
    try:
        nobs = int(res.nobs)
    except Exception:
        # fallback to length of endog if available
        try:
            nobs = int(getattr(res.model, 'endog').shape[0])
        except Exception:
            nobs = None

    # Standardized effect: coef * (sd_x / sd_y)
    std_effect = None
    try:
        # Try to get original data columns from model.data.frame if available
        df = None
        if hasattr(res.model, 'data') and getattr(res.model.data, 'frame', None) is not None:
            df = res.model.data.frame
        elif hasattr(res.model, 'data') and getattr(res.model.data, 'orig_endog', None) is not None:
            # as fallback attempt to reconstruct from endog/exog
            df = None

        if df is not None and var_name in df.columns and res.model.endog is not None:
            x = df[var_name].astype(float).values
            y = np.asarray(res.model.endog, dtype=float)
            sd_x = np.std(x, ddof=1)
            sd_y = np.std(y, ddof=1)
            if sd_x > 0 and sd_y > 0:
                std_effect = float(coef * (sd_x / sd_y))
        else:
            # fallback: use model.exog columns if names available
            exog_names = getattr(res.model, 'exog_names', None)
            exog = getattr(res.model, 'exog', None)
            endog = getattr(res.model, 'endog', None)
            if exog is not None and exog_names is not None and endog is not None:
                if var_name in exog_names:
                    ix = exog_names.index(var_name)
                else:
                    # try substring match
                    matches = [i for i, n in enumerate(exog_names) if var_name in n]
                    ix = matches[0] if matches else None
                if ix is not None:
                    x = np.asarray(exog[:, ix], dtype=float)
                    y = np.asarray(endog, dtype=float)
                    sd_x = np.std(x, ddof=1)
                    sd_y = np.std(y, ddof=1)
                    if sd_x > 0 and sd_y > 0:
                        std_effect = float(coef * (sd_x / sd_y))
    except Exception:
        std_effect = None

    # Interpretation about direction: lower ratio => fewer students per teacher.
    # If coef < 0 then higher ratio -> lower grades, thus lower ratio -> higher grades.
    significance = None
    if pval is not None:
        significance = pval < 0.05

    if coef < 0 and significance:
        verdict = (
            "Yes — statistically significant: the estimated coefficient is negative "
            "(coef = {coef:.4f}, p = {pval:.3g}), which implies that a lower "
            "student-teacher ratio (fewer students per teacher) is associated with "
            "higher average reading scores. 95% CI = [{ci_low:.4f}, {ci_high:.4f}]."
        ).format(coef=coef, pval=pval, ci_low=ci_low if ci_low is not None else float('nan'),
                 ci_high=ci_high if ci_high is not None else float('nan'))
    elif coef < 0 and (significance is False):
        verdict = (
            "No statistically significant evidence: the estimated coefficient is negative "
            "(coef = {coef:.4f}) suggesting lower student-teacher ratio is associated with higher "
            "reading scores, but this effect is not statistically significant (p = {pval:.3g}). "
            "95% CI = [{ci_low:.4f}, {ci_high:.4f}]."
        ).format(coef=coef, pval=pval, ci_low=ci_low if ci_low is not None else float('nan'),
                 ci_high=ci_high if ci_high is not None else float('nan'))
    elif coef > 0 and significance:
        verdict = (
            "Yes — statistically significant but in the opposite direction: the estimated coefficient is positive "
            "(coef = {coef:.4f}, p = {pval:.3g}), which implies that a lower student-teacher ratio (fewer students per teacher) "
            "is associated with lower average reading scores. 95% CI = [{ci_low:.4f}, {ci_high:.4f}]."
        ).format(coef=coef, pval=pval, ci_low=ci_low if ci_low is not None else float('nan'),
                 ci_high=ci_high if ci_high is not None else float('nan'))
    else:
        # coef == 0 or pval is None
        verdict = (
            "No clear evidence of an association: estimated coefficient = {coef:.4f}, p = {pval}. "
            "95% CI = [{ci_low}, {ci_high}]."
        ).format(coef=coef, pval=(f"{pval:.3g}" if pval is not None else "NA"),
                 ci_low=(f"{ci_low:.4f}" if ci_low is not None else "NA"),
                 ci_high=(f"{ci_high:.4f}" if ci_high is not None else "NA"))

    # Build the returned object
    result_object = {
        "variable": var_name,
        "coef": coef,
        "std_error": se,
        "t_stat": tstat,
        "p_value": pval,
        "95%_CI": [ci_low, ci_high],
        "standardized_effect": std_effect,
        "nobs": nobs,
        "significant_at_0.05": significance,
    }

    description_lines = [
        "Extracted coefficient and inference for 'student_teacher_ratio'.",
        verdict,
    ]
    if std_effect is not None:
        description_lines.append(
            f"Standardized effect (coef * sd_x / sd_y) = {std_effect:.4f} (interpretable as SD change in grades per 1 SD change in ratio)."
        )
    description_lines.append("Interpretation caveat: this is an observational association from an OLS model controlling for listed covariates; it does not by itself establish causation.")

    return {"object": result_object, "description": " ".join(description_lines)}