def extract_final_answer(model_output):
    """
    Extract statistics relevant to how children's reliance on majority preference
    develops with age across sites from a fitted statsmodels GLM (logistic).

    Returns a dict with:
      - "object": a dict containing:
          - "coef_table": pandas.DataFrame of model coefficients, SEs, p-values, 95% CI
          - "site_age_slopes": pandas.DataFrame giving, for each Site level,
                * slope (log-odds change per 1 unit of Age_centered),
                * SE of slope, z, p (two-sided),
                * odds ratio (OR) per year and 95% CI for the OR
          These are the numeric objects you can inspect programmatically.
      - "description": a short plain-language interpretation of what those numbers mean
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    res = model_output

    # Basic coefficient table
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        conf = res.conf_int()
    except Exception as e:
        raise ValueError("model_output does not look like a statsmodels results object: %s" % e)

    coef_table = pd.DataFrame({
        "coef": params,
        "se": bse,
        "pvalue": pvalues,
        "ci_lower": conf[0],
        "ci_upper": conf[1]
    })

    # Attempt to get Site levels from the original data used to fit the model
    site_levels = None
    try:
        df = res.model.data.frame  # available when formula and DataFrame were used
        if "Site" in df.columns:
            # preserve categorical ordering if provided
            site_levels = pd.Categorical(df["Site"]).categories.tolist()
    except Exception:
        # fallback: try to infer site names from parameter names (works only if C(Site)[T.<level>] present)
        site_levels = None

    # If we couldn't get site levels from data, infer from param names (excluding baseline)
    if site_levels is None:
        site_tokens = []
        for name in params.index:
            if name.startswith("C(Site)[T."):
                # extract between 'C(Site)[T.' and the closing ']'
                start = name.find("C(Site)[T.") + len("C(Site)[T.")
                end = name.find("]", start)
                token = name[start:end]
                site_tokens.append(token)
        # We can't reliably know the baseline name; treat baseline as "baseline_ref"
        # and the others as parsed tokens. Put baseline first.
        if site_tokens:
            site_levels = ["(baseline)"] + site_tokens
        else:
            # no site terms at all: treat it as single-site case
            site_levels = ["(only_site)"]

    # Compute age slopes (log-odds change per 1 unit of Age_centered) for each site:
    # For baseline: slope = coef['Age_centered']
    # For other sites: slope = coef['Age_centered'] + coef['Age_centered:C(Site)[T.<site>]']
    cov = res.cov_params()  # covariance matrix of coefficients

    # Ensure 'Age_centered' exists
    if "Age_centered" not in params.index:
        raise ValueError("Model does not contain 'Age_centered' term. Check model formula.")

    age_coef = params["Age_centered"]

    rows = []
    for i, site in enumerate(site_levels):
        if i == 0 and site != "(baseline)":
            # If we have a concrete baseline name (from categories), we need to find the correct
            # parameter name for interactions for other sites. When categories were found from the data,
            # the baseline is site_levels[0].
            baseline_name = site_levels[0]
        # Determine the interaction parameter name used by statsmodels for this site (if any)
        if site == "(baseline)" or site == "(only_site)":
            interaction_name = None
        else:
            interaction_name = f"Age_centered:C(Site)[T.{site}]"

        # Get interaction coef if present
        inter_coef = params.get(interaction_name, 0.0)
        slope = age_coef + inter_coef

        # Variance of sum: var(age) + var(inter) + 2*cov(age,inter)
        var_age = cov.loc["Age_centered", "Age_centered"]
        if interaction_name is not None and interaction_name in cov.index:
            var_inter = cov.loc[interaction_name, interaction_name]
            cov_age_inter = cov.loc["Age_centered", interaction_name]
        else:
            var_inter = 0.0
            cov_age_inter = 0.0

        var_slope = var_age + var_inter + 2.0 * cov_age_inter
        se_slope = np.sqrt(var_slope) if var_slope > 0 else 0.0

        # z and two-sided p-value
        if se_slope > 0:
            z = slope / se_slope
            p_two = 2.0 * (1.0 - norm.cdf(abs(z)))
        else:
            z = np.nan
            p_two = np.nan

        # Convert to odds ratio per year and 95% CI
        or_per_year = np.exp(slope)
        ci_low = np.exp(slope - 1.96 * se_slope)
        ci_high = np.exp(slope + 1.96 * se_slope)

        rows.append({
            "Site": site,
            "slope_log_odds_per_year": slope,
            "se_slope": se_slope,
            "z": z,
            "p_value_slope": p_two,
            "OR_per_year": or_per_year,
            "OR_95CI_lower": ci_low,
            "OR_95CI_upper": ci_high
        })

    site_age_slopes = pd.DataFrame(rows)

    # Also report the quadratic (Age_centered_sq) term summary because it indicates nonlinearity
    quad_summary = None
    if "Age_centered_sq" in params.index:
        quad_coef = params["Age_centered_sq"]
        quad_se = bse["Age_centered_sq"]
        quad_p = pvalues["Age_centered_sq"]
        quad_ci = conf.loc["Age_centered_sq"].tolist()
        quad_summary = {
            "coef": quad_coef,
            "se": quad_se,
            "pvalue": quad_p,
            "ci_lower": quad_ci[0],
            "ci_upper": quad_ci[1],
            "interpretation": (
                "A statistically significant quadratic term indicates a nonlinear (curved) "
                "developmental trajectory in log-odds of choosing the majority option with age."
            )
        }

    # Build the object to return
    result_object = {
        "coef_table": coef_table,
        "site_age_slopes": site_age_slopes,
        "age_quadratic_summary": quad_summary
    }

    # Build a short description of how to interpret the numbers
    description_lines = [
        "Extracted coefficients and tests relevant to developmental effects of age on choosing the majority option.",
        "For each Site, 'slope_log_odds_per_year' is the estimated change in log-odds of choosing the majority per 1 unit increase in Age_centered.",
        ("An OR_per_year > 1 indicates increasing reliance on the majority with age in that Site; "
         "OR_per_year < 1 indicates decreasing reliance."),
        "p_value_slope tests whether the slope (log-odds change per year) differs from zero for that Site.",
        "Also included: the full coefficient table for the model and a summary for the quadratic age term (if present).",
        "To conclude whether developmental trajectories differ across cultural contexts, inspect the p-values "
        "of the age-by-site interaction coefficients (these are implicitly used when computing the site-specific slopes)."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}