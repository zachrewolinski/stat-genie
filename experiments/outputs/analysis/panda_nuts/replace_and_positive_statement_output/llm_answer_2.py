def extract_final_answer(model_output):
    """
    Extracts coefficients, SE, t/z, p-values, 95% CIs and exponentiated effects for the
    predictors of interest (age_c, sex, help_bin) from a statsmodels MixedLMResults-like object.

    Returns a dictionary with:
      - "object": dict mapping each target predictor to a dict of statistics
      - "description": human-readable summary of the direction and significance of effects

    The function is written to be robust to slight differences in attribute names across
    statsmodels versions (e.g., fe_params, bse_fe, pvalues).
    """
    import numpy as np
    from math import exp
    from scipy import stats

    # Helper to get attributes robustly
    def get_attr(obj, *names, default=None):
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return default

    # Fixed-effect parameter estimates
    fe_params = get_attr(model_output, "fe_params", "params")
    if fe_params is None:
        raise ValueError("Could not find fixed-effect parameters on model_output.")

    # Standard errors for fixed effects
    bse = get_attr(model_output, "bse_fe", "bse", default=None)
    # p-values (may exist)
    pvalues = get_attr(model_output, "pvalues", default=None)
    # tvalues / z-values (may exist)
    tvalues = get_attr(model_output, "tvalues", "zvalues", default=None)

    # Confidence intervals
    try:
        ci = model_output.conf_int()
    except Exception:
        ci = None

    # Ensure indices/names are available
    param_index = fe_params.index if hasattr(fe_params, "index") else None
    if param_index is None:
        # try converting to pandas Series
        try:
            import pandas as pd
            fe_params = pd.Series(fe_params)
            param_index = fe_params.index
        except Exception:
            raise ValueError("Could not determine parameter names/index.")

    # Convert bse, pvalues, tvalues to Series aligned to param_index if possible
    def to_series(x, name):
        if x is None:
            return None
        try:
            import pandas as pd
            if hasattr(x, "index") and list(x.index) == list(param_index):
                return x
            else:
                # try to coerce
                return pd.Series(x, index=param_index, name=name)
        except Exception:
            return None

    import pandas as pd
    fe_params = pd.Series(fe_params, index=param_index, name="coef")
    bse = to_series(bse, "bse")
    if pvalues is not None:
        pvalues = to_series(pvalues, "pvalue")
    if tvalues is not None:
        tvalues = to_series(tvalues, "t")
    # conf_int to DataFrame with same index if possible
    if ci is not None:
        try:
            ci_df = pd.DataFrame(ci, index=param_index, columns=["ci_lower", "ci_upper"])
        except Exception:
            # if ci has no matching index, try to coerce
            try:
                ci_df = pd.DataFrame(ci, columns=["ci_lower", "ci_upper"])
                ci_df.index = param_index
            except Exception:
                ci_df = None
    else:
        ci_df = None

    # Variables of interest: age_c, sex (any parameter name containing 'sex'), help_bin
    names = list(param_index)

    def find_names(containing):
        return [n for n in names if containing in n]

    target_map = {
        "age_c": find_names("age_c"),
        "sex": [n for n in names if ("sex" in n or "C(sex)" in n or "C(sex)" in n) and "C(sex)" not in n or "sex" in n],
        # Above line is permissive; we'll further filter below
        "help_bin": find_names("help_bin")
    }

    # More careful sex matching: prefer names that contain 'sex' but exclude 'sex' in 'C(sex)' formatting quirks handled above
    sex_names = [n for n in names if "sex" in n]
    # Remove 'C(' if weird; keep all that mention sex except the intercept
    target_map["sex"] = sex_names

    # Build results for each variable (if multiple names found for a predictor, include all)
    results = {}
    for key in ["age_c", "sex", "help_bin"]:
        param_names = target_map.get(key, [])
        if not param_names:
            results[key] = {
                "found": False,
                "message": f"No parameter found matching '{key}'. Available params: {names}"
            }
            continue

        items = {}
        for pname in param_names:
            coef = fe_params.get(pname, np.nan)
            se = bse.get(pname, np.nan) if bse is not None else np.nan
            t_or_z = tvalues.get(pname, np.nan) if tvalues is not None else (coef / se if se and not np.isnan(se) else np.nan)
            pval = pvalues.get(pname, np.nan) if pvalues is not None else np.nan
            # If p-values not present, approximate using normal distribution on z = coef/se
            if (pval is None or (isinstance(pval, float) and np.isnan(pval))) and not (se is None or np.isnan(se)):
                z = coef / se
                pval = 2 * (1 - stats.norm.cdf(abs(z)))
                t_or_z = z
            # Confidence interval
            if ci_df is not None and pname in ci_df.index:
                ci_low = ci_df.loc[pname, "ci_lower"]
                ci_high = ci_df.loc[pname, "ci_upper"]
            else:
                # Approximate 95% CI from coef +/- 1.96*se if se available
                if not (se is None or np.isnan(se)):
                    ci_low = coef - 1.96 * se
                    ci_high = coef + 1.96 * se
                else:
                    ci_low = ci_high = np.nan

            # Exponentiated coefficient (multiplicative effect on nuts_per_min + 1e-6)
            try:
                exp_coef = float(np.exp(coef))
                pct_change = (exp_coef - 1.0) * 100.0
            except Exception:
                exp_coef = np.nan
                pct_change = np.nan

            items[pname] = {
                "coef": float(coef) if not np.isnan(coef) else np.nan,
                "se": float(se) if not (se is None or np.isnan(se)) else np.nan,
                "t_or_z": float(t_or_z) if not (t_or_z is None or np.isnan(t_or_z)) else np.nan,
                "p_value": float(pval) if not (pval is None or np.isnan(pval)) else np.nan,
                "ci_95": [float(ci_low) if not np.isnan(ci_low) else np.nan,
                          float(ci_high) if not np.isnan(ci_high) else np.nan],
                "exp_coef": exp_coef,
                "pct_change_approx": pct_change
            }

        results[key] = {
            "found": True,
            "parameters": items
        }

    # Form a concise human-readable description summarizing direction and significance
    def sig_label(p):
        if p is None or np.isnan(p):
            return "p=?"
        if p < 0.001:
            return "p<0.001"
        return f"p={p:.3f}"

    desc_lines = []
    # Age
    if results["age_c"]["found"]:
        # If multiple age params (unlikely), summarize the first
        pname, stats_age = next(iter(results["age_c"]["parameters"].items()))
        coef = stats_age["coef"]
        pval = stats_age["p_value"]
        expc = stats_age["exp_coef"]
        pct = stats_age["pct_change_approx"]
        ci = stats_age["ci_95"]
        sig = sig_label(pval)
        direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
        desc_lines.append(
            f"Age (per centered year) [{pname}]: coef={coef:.3f}, SE={stats_age['se']:.3f}, {sig}, 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]. "
            f"On the original rate scale this corresponds to multiplicative factor {expc:.3f} (~{pct:.1f}% {direction} in nuts/min per year)."
        )
    else:
        desc_lines.append(results["age_c"]["message"])

    # Sex
    if results["sex"]["found"]:
        for pname, stats_sex in results["sex"]["parameters"].items():
            coef = stats_sex["coef"]
            pval = stats_sex["p_value"]
            expc = stats_sex["exp_coef"]
            pct = stats_sex["pct_change_approx"]
            ci = stats_sex["ci_95"]
            sig = sig_label(pval)
            # Determine comparison label from parameter name if possible
            label = pname
            desc_lines.append(
                f"Sex [{label}]: coef={coef:.3f}, SE={stats_sex['se']:.3f}, {sig}, 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]. "
                f"On the original scale: multiplicative factor {expc:.3f} (~{pct:.1f}% change in nuts/min for this sex level vs reference)."
            )
    else:
        desc_lines.append(results["sex"]["message"])

    # Help
    if results["help_bin"]["found"]:
        pname, stats_help = next(iter(results["help_bin"]["parameters"].items()))
        coef = stats_help["coef"]
        pval = stats_help["p_value"]
        expc = stats_help["exp_coef"]
        pct = stats_help["pct_change_approx"]
        ci = stats_help["ci_95"]
        sig = sig_label(pval)
        direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
        desc_lines.append(
            f"Receiving help [{pname}]: coef={coef:.3f}, SE={stats_help['se']:.3f}, {sig}, 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]. "
            f"On the original rate scale: multiplicative factor {expc:.3f} (~{pct:.1f}% {direction} in nuts/min when helped vs not helped)."
        )
    else:
        desc_lines.append(results["help_bin"]["message"])

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }