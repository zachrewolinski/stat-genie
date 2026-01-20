def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and multiplicative
    (percent) effects on fish_per_hour from a statsmodels OLS fit on log(fish_per_hour).
    
    Returns a dict with:
      - "object": a dictionary containing a coefficients table and model fit stats
      - "description": a human-readable interpretation of the key results
    
    Expects model_output to be a statsmodels RegressionResults (RegressionResultsWrapper).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic validation
    required_attrs = ['params', 'bse', 'pvalues', 'conf_int', 'rsquared', 'nobs']
    for a in required_attrs:
        if not hasattr(res, a):
            raise ValueError(f"Provided model_output is missing required attribute: {a}")

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int(alpha=0.05)  # DataFrame with two columns (lower, upper)

    # Variables of interest (include constant if present)
    vars_of_interest = list(params.index)

    rows = []
    for v in vars_of_interest:
        coef = float(params[v])
        se = float(bse[v]) if v in bse.index else None
        pval = float(pvalues[v]) if v in pvalues.index else None
        ci_lower = float(conf.loc[v, 0]) if v in conf.index else None
        ci_upper = float(conf.loc[v, 1]) if v in conf.index else None

        # For a model on log(rate), coefficients approximate log-multiplicative effects.
        # Convert to percent change on the original fish_per_hour scale:
        pct_change = (np.exp(coef) - 1) * 100.0
        pct_lower = (np.exp(ci_lower) - 1) * 100.0 if ci_lower is not None else None
        pct_upper = (np.exp(ci_upper) - 1) * 100.0 if ci_upper is not None else None

        signif = (pval is not None) and (pval < 0.05)

        rows.append({
            "term": v,
            "coef_log": coef,
            "se": se,
            "pvalue": pval,
            "ci_log_low": ci_lower,
            "ci_log_high": ci_upper,
            "pct_change": pct_change,
            "pct_change_ci_low": pct_lower,
            "pct_change_ci_high": pct_upper,
            "significant_p_lt_0.05": bool(signif)
        })

    coef_table = pd.DataFrame(rows).set_index("term")

    # Model-level metrics
    model_info = {
        "nobs": int(res.nobs),
        "r_squared": float(res.rsquared) if hasattr(res, "rsquared") else None,
        "adj_r_squared": float(res.rsquared_adj) if hasattr(res, "rsquared_adj") else None
    }

    # Build a concise human-readable description focusing on substantive effects.
    desc_lines = []
    desc_lines.append(
        f"Model fit: n = {model_info['nobs']}, R^2 = {model_info['r_squared']:.3f}, "
        f"adj. R^2 = {model_info['adj_r_squared']:.3f}."
    )

    # For each non-constant term produce an interpretation sentence.
    for term, row in coef_table.iterrows():
        if term.lower() in ['const', 'constant']:
            continue

        pct = row["pct_change"]
        low = row["pct_change_ci_low"]
        high = row["pct_change_ci_high"]
        pval = row["pvalue"]
        signif = row["significant_p_lt_0.05"]

        # Interpret standardized variables (persons_s, camper_s) explicitly.
        if term == "persons_s":
            unit_desc = "one standard-deviation increase in group size"
        elif term == "camper_s":
            unit_desc = "one standard-deviation increase in number of campers"
        elif term == "livebait":
            unit_desc = "using live bait (vs not using live bait)"
        elif term == "child":
            unit_desc = "presence of a child in the group (vs none)"
        elif term == "livebait_persons":
            unit_desc = "interaction: livebait × standardized group size (interpret with main terms)"
        else:
            unit_desc = f"a one-unit increase in {term}"

        # Format percent and CI nicely
        pct_str = f"{pct:.1f}%"
        ci_str = f"[{low:.1f}%, {high:.1f}%]" if (low is not None and high is not None) else "CI unavailable"
        sig_str = "statistically significant (p < 0.05)" if signif else f"not statistically significant (p = {pval:.3f})"

        line = (
            f"Holding other variables constant, {unit_desc} is associated with an estimated "
            f"{pct_str} change in fish-per-hour ({ci_str}), {sig_str}."
        )
        desc_lines.append(line)

    description = " ".join(desc_lines)

    # Prepare object to return (JSON-serializable)
    result_object = {
        "coeff_table": coef_table.round(4).to_dict(orient="index"),
        "model_info": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in model_info.items()}
    }

    return {"object": result_object, "description": description}