def extract_final_answer(model_output):
    """
    Extracts the SkinTone effect (Dark vs Light) from a fitted statsmodels GLM/GLMResultsWrapper object
    (optionally with clustered robust covariance applied). Returns a dictionary with a numeric object
    (detailed stats) and a short human-readable description/interpretation.

    Returned dictionary:
      {
        "object": {
          "param_name": str,
          "coef": float,             # log rate ratio (coefficient)
          "se": float,               # standard error (based on covariance in model_output)
          "pvalue": float,
          "ci_lower": float,         # 95% CI on coefficient scale
          "ci_upper": float,
          "irr": float,              # incidence rate ratio = exp(coef)
          "irr_ci_lower": float,
          "irr_ci_upper": float,
          "comparison": str,         # e.g., "Dark vs Light" or best-effort description
          "significant": bool,
          "alpha": 0.05
        },
        "description": str           # short plain-English summary/conclusion
      }
    """
    import re
    import numpy as np
    from scipy import stats

    # Prepare a friendly error return
    def error_return(msg):
        return {
            "object": None,
            "description": f"Could not extract SkinTone effect: {msg}"
        }

    # Try to obtain parameter names and coefficients
    try:
        params = model_output.params
    except Exception as e:
        return error_return(f"model_output has no .params attribute ({e})")

    # params is usually a pandas Series
    try:
        param_names = list(params.index)
    except Exception:
        # fallback: convert to list of strings
        param_names = [str(p) for p in params]

    # Find parameter(s) related to SkinTone (case-insensitive)
    skin_idx = [i for i, n in enumerate(param_names) if 'skintone' in n.lower() or 'skin_tone' in n.lower() or 'skin' in n.lower() and 'tone' in n.lower()]
    # If that failed, try more generic 'C(SkinTone)' pattern or exact 'SkinTone'
    if not skin_idx:
        skin_idx = [i for i, n in enumerate(param_names) if 'C(SkinTone)' in n or 'SkinTone' == n or n.startswith('SkinTone')]

    if not skin_idx:
        return error_return("No parameter name matching 'SkinTone' found in model parameters. Available parameters: "
                            + ", ".join(param_names))

    # Choose the first matching SkinTone parameter (typical case: one indicator)
    idx = skin_idx[0]
    name = param_names[idx]
    coef = float(params.iloc[idx])

    # Try to get p-value, standard error, and confidence interval from model_output
    try:
        pvalues = model_output.pvalues
        pval = float(pvalues[name])
    except Exception:
        # compute from covariance matrix if pvalues not present
        try:
            cov = model_output.cov_params()
            se = float(np.sqrt(np.diag(cov))[idx])
            z = coef / se if se != 0 else np.nan
            pval = float(2 * (1 - stats.norm.cdf(abs(z))))
        except Exception as e:
            return error_return(f"Could not obtain p-value or covariance matrix ({e})")

    # Get standard error
    try:
        se = float(model_output.bse[name])
    except Exception:
        try:
            cov = model_output.cov_params()
            se = float(np.sqrt(np.diag(cov))[idx])
        except Exception as e:
            return error_return(f"Could not obtain standard error ({e})")

    # Confidence interval (95%)
    alpha = 0.05
    try:
        ci = model_output.conf_int(alpha=alpha)
        # conf_int may be a DataFrame
        ci_lower = float(ci.loc[name][0])
        ci_upper = float(ci.loc[name][1])
    except Exception:
        # fallback using normal approximation
        zcrit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = coef - zcrit * se
        ci_upper = coef + zcrit * se

    # Incidence rate ratio (IRR) and CI on IRR scale
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Attempt to determine which level is compared to which (e.g., "T.Dark" => Dark vs reference)
    comparison = name
    # common statsmodels naming: C(SkinTone)[T.Dark]
    m = re.search(r'T\.([^\]\s]+)', name)
    if m:
        level = m.group(1)
        # try to find reference level from the original data if available
        ref = None
        try:
            df = model_output.model.data.frame
            if 'SkinTone' in df.columns:
                levels = list(pd.Series(df['SkinTone']).dropna().unique())
                # if two levels found, reference is the one not equal to level (best-effort)
                if len(levels) == 2:
                    other = [l for l in levels if str(l) != level]
                    if other:
                        ref = other[0]
        except Exception:
            ref = None

        if ref is not None:
            comparison = f"{level} vs {ref}"
        else:
            comparison = f"{level} vs reference"
    else:
        # fallback: if name contains 'Dark' or 'Light' use that info
        if 'dark' in name.lower():
            comparison = "Dark vs reference"
        elif 'light' in name.lower():
            comparison = "Light vs reference"
        else:
            comparison = name

    # Build result object
    result_obj = {
        "param_name": name,
        "coef": coef,
        "se": se,
        "pvalue": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "comparison": comparison,
        "significant": (pval < alpha),
        "alpha": alpha
    }

    # Form a concise interpretation answering the task question
    # Determine whether the parameter compares Dark to Light specifically
    conclusion = ""
    # If comparison explicitly names Dark vs Light, build direct conclusion
    if ('Dark' in comparison) and ('Light' in comparison or 'light' in comparison or 'reference' in comparison):
        # decide direction
        if 'Dark' in comparison:
            if irr > 1 and pval < alpha:
                conclusion = (f"Yes — estimated IRR for Dark players vs Light = {irr:.3f} "
                              f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}], p = {pval:.3g}). "
                              "Dark-skin players receive red cards at a statistically significantly higher rate.")
            elif irr > 1:
                conclusion = (f"No statistically significant evidence — estimated IRR for Dark vs Light = {irr:.3f} "
                              f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}], p = {pval:.3g}). "
                              "Point estimate suggests higher rate for Dark players but not statistically significant.")
            elif irr < 1 and pval < alpha:
                conclusion = (f"No — estimated IRR for Dark vs Light = {irr:.3f} "
                              f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}], p = {pval:.3g}). "
                              "Dark-skin players receive red cards at a statistically significantly lower rate.")
            else:
                conclusion = (f"No statistically significant evidence — estimated IRR for Dark vs Light = {irr:.3f} "
                              f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}], p = {pval:.3g}).")
    else:
        # General wording if we couldn't conclusively map the comparison to Dark vs Light
        direction = "higher" if irr > 1 else "lower"
        sig_text = "statistically significant" if pval < alpha else "not statistically significant"
        conclusion = (f"The model estimate for '{name}' is coef = {coef:.4f} (IRR = {irr:.3f}, "
                      f"95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}], p = {pval:.3g}). "
                      f"This indicates the compared level has a {direction} rate of red cards vs the reference; "
                      f"the effect is {sig_text} at α = {alpha}.")

    description = conclusion

    return {"object": result_obj, "description": description}