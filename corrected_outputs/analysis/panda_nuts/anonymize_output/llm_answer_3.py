def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, and 95% confidence intervals for the
    focal terms: Age_c, Sex_male, Help, Age_c:Help, Sex_male:Help from a statsmodels
    fitted model object (MixedLMResultsWrapper or RegressionResultsWrapper).

    Returns a dictionary:
      {
        "object": { term_name: { "param_name": str,
                                 "coef": float or None,
                                 "se": float or None,
                                 "pvalue": float or None,
                                 "ci_lower": float or None,
                                 "ci_upper": float or None,
                                 "significant": bool or None
                               }, ... },
        "description": "Readable summary interpreting each effect"
      }
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Focal terms (we will match these flexibly because statsmodels may name interactions
    # in either order like 'Age_c:Help' or 'Help:Age_c')
    focal_terms = [
        ("Age_c", ["Age_c"]),
        ("Sex_male", ["Sex_male"]),
        ("Help", ["Help"]),
        ("Age_c:Help", ["Age_c", "Help"]),
        ("Sex_male:Help", ["Sex_male", "Help"])
    ]

    # Retrieve parameter estimates (fixed effects) robustly for MixedLM or OLS
    # Try typical attribute names in order of preference
    if hasattr(model_output, "fe_params"):
        params = pd.Series(model_output.fe_params)
    elif hasattr(model_output, "params"):
        params = pd.Series(model_output.params)
    else:
        raise ValueError("Cannot locate parameter estimates on the provided model object.")

    # Standard errors
    if hasattr(model_output, "bse_fe"):
        bse = pd.Series(model_output.bse_fe)
    elif hasattr(model_output, "bse"):
        bse = pd.Series(model_output.bse)
    else:
        # If no bse available, create NaN series with same index
        bse = pd.Series(index=params.index, dtype=float)

    # p-values (may not exist for some MixedLM results); try multiple attribute names
    pvalues = None
    if hasattr(model_output, "pvalues"):
        try:
            pvalues = pd.Series(model_output.pvalues)
        except Exception:
            pvalues = None
    if pvalues is None and hasattr(model_output, "pvalues_fe"):
        try:
            pvalues = pd.Series(model_output.pvalues_fe)
        except Exception:
            pvalues = None

    # Confidence intervals
    try:
        ci = model_output.conf_int()
        # conf_int returns DataFrame-like with two columns; ensure it's a DataFrame
        ci = pd.DataFrame(ci)
    except Exception:
        ci = None

    # If p-values are missing but we have params and bse and df_resid, approximate p-values using t-distribution
    if (pvalues is None or pvalues.empty) and (not bse.isnull().all()):
        try:
            df_resid = float(getattr(model_output, "df_resid", np.nan))
            t_stats = params / bse
            if np.isfinite(df_resid):
                p_approx = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))
                pvalues = pd.Series(p_approx, index=params.index)
            else:
                # fallback using normal approx
                p_approx = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
                pvalues = pd.Series(p_approx, index=params.index)
        except Exception:
            pvalues = pd.Series(index=params.index, dtype=float)

    # Helper to find parameter name in params index that matches all tokens (order-insensitive)
    def find_param_index(tokens):
        tokens = [t for t in tokens if t]  # filter empties
        # direct exact match first
        for name in params.index:
            if name in ("Intercept", "const"):
                continue
            if name == ":".join(tokens) or name == tokens[0]:
                return name
        # search for a param name that contains all tokens (in any order), separated by non-alphanumeric characters
        for name in params.index:
            lname = str(name)
            if all(tok in lname for tok in tokens):
                return name
        return None

    results = {}
    alpha = 0.05
    for label, tokens in focal_terms:
        matched = find_param_index(tokens)
        if matched is None:
            # not found; report None
            results[label] = {
                "param_name": None,
                "coef": None,
                "se": None,
                "pvalue": None,
                "ci_lower": None,
                "ci_upper": None,
                "significant": None,
                "note": f"No parameter matching tokens {tokens} found in model."
            }
            continue

        coef = float(params.get(matched, np.nan)) if matched in params.index else None
        se = float(bse.get(matched, np.nan)) if matched in bse.index else None
        pval = float(pvalues.get(matched, np.nan)) if (pvalues is not None and matched in pvalues.index) else None

        if ci is not None:
            # conf_int DataFrame may have row indexed by parameter name
            try:
                if matched in ci.index:
                    ci_lower = float(ci.loc[matched].iloc[0])
                    ci_upper = float(ci.loc[matched].iloc[1])
                else:
                    # sometimes conf_int uses integer index matching params order; try position-based
                    # fallback: compute using coef +/- 1.96 * se
                    ci_lower = float(coef - 1.96 * se) if (coef is not None and se is not None) else None
                    ci_upper = float(coef + 1.96 * se) if (coef is not None and se is not None) else None
            except Exception:
                ci_lower = float(coef - 1.96 * se) if (coef is not None and se is not None) else None
                ci_upper = float(coef + 1.96 * se) if (coef is not None and se is not None) else None
        else:
            ci_lower = float(coef - 1.96 * se) if (coef is not None and se is not None) else None
            ci_upper = float(coef + 1.96 * se) if (coef is not None and se is not None) else None

        significant = None
        if pval is not None and not np.isnan(pval):
            significant = bool(pval < alpha)
        elif ci_lower is not None and ci_upper is not None:
            # if CI does not include 0, treat as significant
            significant = not (ci_lower <= 0 <= ci_upper)
        else:
            significant = None

        results[label] = {
            "param_name": str(matched),
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": significant
        }

    # Build a brief textual description interpreting each effect
    descr_lines = []
    for label in ["Age_c", "Sex_male", "Help", "Age_c:Help", "Sex_male:Help"]:
        entry = results[label]
        if entry.get("param_name") is None:
            descr_lines.append(f"{label}: parameter not found in model.")
            continue
        coef = entry["coef"]
        pval = entry["pvalue"]
        sig = entry["significant"]
        # interpret direction if coef available
        if coef is None:
            descr = f"{label} ({entry['param_name']}): estimate unavailable."
        else:
            direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
            # short meaning depending on the term
            if label == "Age_c":
                meaning = f"A 1-year increase in age (centered) is associated with a {coef:.3f} nuts/min {direction}."
            elif label == "Sex_male":
                meaning = f"Males (vs. females) are estimated to differ by {coef:.3f} nuts/min ({'higher' if coef>0 else 'lower'} if significant)."
            elif label == "Help":
                meaning = f"Receiving help (vs. not) is associated with a {coef:.3f} nuts/min {direction}."
            elif label == "Age_c:Help":
                meaning = f"The Age x Help interaction ({coef:.3f}) indicates how the effect of help changes with age: positive means help benefits older individuals more."
            elif label == "Sex_male:Help":
                meaning = f"The Sex x Help interaction ({coef:.3f}) indicates whether the effect of help differs between males and females (positive means males benefit more)."
            else:
                meaning = f"{label}: coef={coef:.3f}."

            signif_str = ("statistically significant (p={:.3g})".format(pval) if (pval is not None and not np.isnan(pval)) else
                          ("significant by CI" if sig else "not significant" if sig is False else "significance unknown"))
            descr = f"{label} ({entry['param_name']}): {meaning} Estimated SE={entry['se']:.3f}." \
                    f" This effect is {signif_str}."
        descr_lines.append(descr)

    description = " | ".join(descr_lines)

    return {"object": results, "description": description}