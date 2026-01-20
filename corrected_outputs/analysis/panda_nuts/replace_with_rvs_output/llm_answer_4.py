def extract_final_answer(model_output):
    """
    Extract key fixed-effect coefficients, standard errors, p-values, and 95% CIs
    from a fitted statsmodels MixedLMResults (or wrapper) object, focusing on:
      - age_c (main effect)
      - sex (male vs female contrast)
      - help (yes vs no contrast)
      - interaction age_c:help

    Returns:
      {
        "object": {
          "coeffs": { term_name: {coef, se, pvalue, ci_low, ci_high} , ...},
          "model_stats": { "aic": ..., "bic": ..., "llf": ... },
          "random_effects": { "re_variance": ..., "resid_variance (scale)": ... }
        },
        "description": "Interpretation text describing how to read the stats in context."
      }
    """
    import numpy as np
    from scipy.stats import norm
    import pandas as pd

    res = model_output  # expected statsmodels MixedLMResultsWrapper

    # Try to obtain parameter estimates and standard errors
    try:
        params = pd.Series(res.params)
    except Exception:
        params = pd.Series(res.params if hasattr(res, "params") else {})
    try:
        bse = pd.Series(res.bse)
    except Exception:
        bse = pd.Series(res.bse if hasattr(res, "bse") else {})

    # Compute or obtain p-values
    if hasattr(res, "pvalues") and res.pvalues is not None:
        try:
            pvals = pd.Series(res.pvalues)
        except Exception:
            pvals = pd.Series(res.pvalues)
    else:
        # Normal (Wald) approximation: z = coef / se
        with np.errstate(divide="ignore", invalid="ignore"):
            z = params.values / bse.values
            pvals_arr = 2 * (1 - norm.cdf(np.abs(z)))
            pvals = pd.Series(pvals_arr, index=params.index)

    # Confidence intervals: try res.conf_int(), otherwise use ±1.96*se
    try:
        ci = res.conf_int()
        # conf_int may be a DataFrame with two columns; ensure index alignment
        ci_df = pd.DataFrame(ci)
        if ci_df.shape[1] == 2:
            ci_df.columns = ["ci_low", "ci_high"]
        else:
            # fallback: compute from se
            raise ValueError("Unexpected conf_int shape")
    except Exception:
        ci_low = params - 1.96 * bse
        ci_high = params + 1.96 * bse
        ci_df = pd.DataFrame({"ci_low": ci_low, "ci_high": ci_high})

    # Helper to find parameter names robustly
    param_index = list(params.index)

    def find_param(patterns):
        """
        patterns: list of substrings; return first param name that contains all substrings
        """
        for name in param_index:
            if all(p in name for p in patterns):
                return name
        return None

    # Target terms (robust search)
    term_age = find_param(["age_c"])  # main effect
    term_sex = find_param(["C(sex)"]) or find_param(["sex"])  # e.g., 'C(sex)[T.m]'
    term_help = find_param(["C(help)"]) or find_param(["help"])  # e.g., 'C(help)[T.yes]'
    term_inter = None
    # look for an interaction containing both age_c and help
    for name in param_index:
        if "age_c" in name and ("help" in name or "C(help)" in name):
            term_inter = name
            break
    # as fallback, try pattern with ':' combining both
    if term_inter is None:
        for name in param_index:
            if ":" in name and "age_c" in name:
                term_inter = name
                break

    # Collect stats for each target term
    def collect_stats(term_name):
        if term_name is None:
            return None
        return {
            "term": term_name,
            "coef": float(params.get(term_name, np.nan)),
            "se": float(bse.get(term_name, np.nan)),
            "pvalue": float(pvals.get(term_name, np.nan)),
            "ci_low": float(ci_df.loc[term_name, "ci_low"]) if term_name in ci_df.index else float(np.nan),
            "ci_high": float(ci_df.loc[term_name, "ci_high"]) if term_name in ci_df.index else float(np.nan),
        }

    stats = {
        "age_c": collect_stats(term_age),
        "sex_contrast": collect_stats(term_sex),
        "help_contrast": collect_stats(term_help),
        "age_by_help_interaction": collect_stats(term_inter),
    }

    # Model-level stats
    model_stats = {}
    for attr in ("aic", "bic", "llf"):
        model_stats[attr] = float(getattr(res, attr)) if hasattr(res, attr) else None

    # Random effect / residual variance if available
    random_effects = {}
    # Try to extract random effect covariance (variance) and scale (residual variance)
    try:
        # covariance of random effects (may be DataFrame or array)
        cov_re = res.cov_re
        random_effects["cov_re"] = cov_re if cov_re is None else np.array(cov_re).tolist()
    except Exception:
        random_effects["cov_re"] = None
    try:
        random_effects["scale"] = float(res.scale)  # residual variance (sigma^2)
    except Exception:
        random_effects["scale"] = None

    output_object = {
        "coeffs": stats,
        "model_stats": model_stats,
        "random_effects": random_effects,
    }

    # Build a brief interpretation template (numbers are in output_object["coeffs"])
    description_lines = []
    description_lines.append(
        "Extracted fixed-effect estimates (coef, se, p-value, 95% CI) for age, sex, help, and their interaction."
    )
    description_lines.append(
        "Interpretation guide: positive coefficient => higher log(nuts/sec); negative => lower log(nuts/sec)."
    )
    description_lines.append(
        "Significance: p-value < 0.05 suggests evidence the effect differs from zero (Wald test)."
    )
    description_lines.append(
        "age_c: main effect of centered age (effect of increasing age on log nuts/sec when help is at reference level)."
    )
    description_lines.append(
        "sex_contrast: effect of the non-reference sex level vs reference (e.g., 'm' vs 'f'). Check term name to see coding."
    )
    description_lines.append(
        "help_contrast: effect of receiving help ('yes') vs not ('no') on log nuts/sec (main effect)."
    )
    description_lines.append(
        "age_by_help_interaction: whether the slope of age differs depending on receiving help (positive => help increases age-related gain)."
    )

    description = " ".join(description_lines)

    return {"object": output_object, "description": description}