def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, confidence intervals, and site-specific age slopes
    from a statsmodels Logit model output dictionary as produced by the provided
    modeling function.

    Returns:
      {
        "object": {
            "age_coef": float,
            "age_pvalue": float,
            "age_ci": [low, high],
            "age_sq_coef": float,
            "age_sq_pvalue": float,
            "age_sq_ci": [low, high],
            "age_marginal_effect_overall": float or None,
            "age_marginal_effect_se": float or None,
            "age_slopes_by_site": { site_name: { "slope": float, "se": float, "z": float, "p": float } , ... },
            "interaction_terms_pvalues": { interaction_param_name: pvalue, ... }
        },
        "description": "brief human-readable interpretation"
      }
    """
    import numpy as np
    import pandas as pd

    # Try import stats for z->p conversion; if missing, leave p as None when needed
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = {"object": None, "description": None}

    # Unpack
    model_fit = model_output.get('model_fit', None)
    marg_eff = model_output.get('marginal_effects', None)

    if model_fit is None:
        res['description'] = "No fitted model found in model_output['model_fit']."
        return res

    params = getattr(model_fit, "params", pd.Series(dtype=float))
    pvalues = getattr(model_fit, "pvalues", pd.Series(dtype=float))
    # conf_int may be a method
    try:
        conf = model_fit.conf_int()
    except Exception:
        # empty DataFrame if unavailable
        conf = pd.DataFrame(index=params.index, columns=[0, 1])

    try:
        cov = model_fit.cov_params()
    except Exception:
        cov = pd.DataFrame()

    # Helper to safely get param/pvalue/conf
    def get_param(name):
        # params and pvalues may be pandas Series
        val = params.get(name, np.nan) if hasattr(params, "get") else params[name] if name in params else np.nan
        pv = pvalues.get(name, np.nan) if hasattr(pvalues, "get") else pvalues[name] if name in pvalues else np.nan
        if name in getattr(conf, "index", []):
            try:
                ci_low, ci_high = conf.loc[name].tolist()
            except Exception:
                ci_low, ci_high = (np.nan, np.nan)
        else:
            ci_low, ci_high = (np.nan, np.nan)
        return val, pv, (ci_low, ci_high)

    # small helper to convert to float or None
    def maybe_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, (float, int, np.floating, np.integer)):
                if np.isnan(x):
                    return None
                return float(x)
            # pandas NA handling
            if pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    # Extract main Age effects
    age_name = 'Age_c'
    age_sq_name = 'Age_sq'

    age_coef, age_p, age_ci = get_param(age_name)
    age_sq_coef, age_sq_p, age_sq_ci = get_param(age_sq_name)

    # Try to get average marginal effect for Age_c if marginal_effects available
    age_margeff = None
    age_margeff_se = None
    if marg_eff is not None:
        # Attempt multiple safe strategies to extract marginal effect for Age_c
        try:
            if hasattr(marg_eff, "summary_frame"):
                try:
                    sf = marg_eff.summary_frame()
                    # Ensure index are strings for matching
                    sf_index_str = [str(i) for i in sf.index]
                    # possible column names for effect and se
                    eff_cols = [c for c in sf.columns if c.lower().replace(".", "").replace(" ", "") in ("dydx", "dy/dx", "dy_dx", "effect")]
                    se_cols = [c for c in sf.columns if c.lower().replace(".", "").replace(" ", "") in ("stderr", "std.err", "stderr", "std_err", "se")]
                    eff_col = eff_cols[0] if eff_cols else None
                    se_col = se_cols[0] if se_cols else None

                    # try exact match first
                    matched_idx = None
                    if age_name in sf.index:
                        matched_idx = age_name
                    else:
                        # find row where age_name appears in its string representation
                        for i, s in zip(sf.index, sf_index_str):
                            if age_name in s:
                                matched_idx = i
                                break

                    if matched_idx is not None and eff_col is not None:
                        try:
                            age_margeff = maybe_float(sf.loc[matched_idx, eff_col])
                        except Exception:
                            age_margeff = None
                        if se_col is not None:
                            try:
                                age_margeff_se = maybe_float(sf.loc[matched_idx, se_col])
                            except Exception:
                                age_margeff_se = None
                    else:
                        # as fallback, if the summary frame is a single-row summary for each observation,
                        # try to find a column named exactly 'Age_c' (some versions can present wide output)
                        if age_name in sf.columns:
                            age_margeff = maybe_float(sf[age_name].iat[0]) if len(sf) > 0 else None
                except Exception:
                    # fall through to other methods
                    age_margeff = None
                    age_margeff_se = None
            # other marg_eff attributes
            if age_margeff is None:
                # try marg_eff.margeff(s)
                if hasattr(marg_eff, "margeff"):
                    try:
                        me = marg_eff.margeff
                        if isinstance(me, (list, tuple, np.ndarray)):
                            age_margeff = maybe_float(me[0]) if len(me) > 0 else None
                        else:
                            age_margeff = maybe_float(me)
                    except Exception:
                        age_margeff = None
                elif hasattr(marg_eff, "margeffs"):
                    try:
                        me = marg_eff.margeffs
                        if isinstance(me, (list, tuple, np.ndarray)):
                            age_margeff = maybe_float(me[0]) if len(me) > 0 else None
                        else:
                            age_margeff = maybe_float(me)
                    except Exception:
                        age_margeff = None

            # try to find a standard error attribute if available
            if age_margeff_se is None:
                for attr in ("margeff_se", "margeff_se_mean", "margeff_se_ave", "se"):
                    if hasattr(marg_eff, attr):
                        try:
                            se_val = getattr(marg_eff, attr)
                            if isinstance(se_val, (list, tuple, np.ndarray)):
                                age_margeff_se = maybe_float(se_val[0]) if len(se_val) > 0 else None
                            else:
                                age_margeff_se = maybe_float(se_val)
                            break
                        except Exception:
                            continue
        except Exception:
            age_margeff = None
            age_margeff_se = None

    # Determine site levels from the model data frame if available
    site_levels = []
    try:
        df = model_fit.model.data.frame
        if df is not None and 'Site' in df.columns:
            # If categorical, get categories, else unique values
            try:
                if pd.api.types.is_categorical_dtype(df['Site']):
                    site_levels = list(df['Site'].cat.categories)
                else:
                    site_levels = sorted(pd.unique(df['Site']).tolist())
            except Exception:
                site_levels = sorted(df['Site'].dropna().unique().tolist())
    except Exception:
        site_levels = []

    # Build site-specific slopes for Age_c
    slopes_by_site = {}
    # collect interaction p-values for reporting
    interaction_pvalues = {}

    # We'll need se for the Age_c parameter (base)
    age_var = np.nan
    try:
        if (hasattr(cov, "loc")) and (age_name in cov.index):
            age_var = cov.loc[age_name, age_name]
    except Exception:
        age_var = np.nan
    age_se = np.sqrt(age_var) if (not pd.isna(age_var)) else np.nan

    # The reference site is the first category in site_levels if available.
    ref_site = site_levels[0] if site_levels else None

    # helper to find interaction param name for a given site label
    def find_interaction_param_for_site(site_label):
        candidates = []
        for pname in params.index:
            ps = str(pname)
            if 'Age_c' in ps and 'C(Site)' in ps and str(site_label) in ps:
                candidates.append(pname)
        # Also consider reversed order or different coding
        for pname in params.index:
            ps = str(pname)
            if 'Age_c' in ps and 'Site' in ps and str(site_label) in ps and pname not in candidates:
                candidates.append(pname)
        return candidates[0] if candidates else None

    if site_levels:
        for site in site_levels:
            if site == ref_site:
                slope = age_coef
                se = age_se
            else:
                inter_name = find_interaction_param_for_site(site)
                if inter_name:
                    inter_coef = params.get(inter_name, 0.0)
                    slope = (age_coef if not pd.isna(age_coef) else 0.0) + (inter_coef if not pd.isna(inter_coef) else 0.0)
                    # compute SE of sum: var(a) + var(b) + 2 cov(a,b)
                    try:
                        var_a = cov.loc[age_name, age_name]
                        var_b = cov.loc[inter_name, inter_name]
                        cov_ab = cov.loc[age_name, inter_name]
                        se = np.sqrt(var_a + var_b + 2 * cov_ab)
                    except Exception:
                        se = np.nan
                    interaction_pvalues[str(inter_name)] = maybe_float(pvalues.get(inter_name, np.nan))
                else:
                    # no interaction term present (maybe site was reference or coding different)
                    slope = age_coef
                    se = age_se
            # compute z and p for slope
            if (se is not None) and (not pd.isna(se)) and (se != 0) and (slope is not None) and stats is not None:
                try:
                    z = (slope) / se
                    p = float(2 * (1 - stats.norm.cdf(abs(z))))
                except Exception:
                    z = None
                    p = None
            else:
                z = None
                p = None
            slopes_by_site[str(site)] = {
                "slope": maybe_float(slope),
                "se": maybe_float(se),
                "z": maybe_float(z),
                "p": maybe_float(p)
            }
    else:
        # If site levels unknown, still try to find all interaction params from param names
        for pname in params.index:
            ps = str(pname)
            if 'Age_c' in ps and 'C(Site)' in ps:
                inter_name = pname
                # try to parse site label from pname
                # common forms: 'Age_c:C(Site)[T.site_label]' or 'C(Site)[T.site_label]:Age_c'
                site_label = ps
                if ']' in ps:
                    parts = ps.split(']')
                    site_label = parts[-1].strip(':').strip()
                    if site_label == '':
                        site_label = parts[-1]
                inter_coef = params.get(inter_name, 0.0)
                slope = (age_coef if not pd.isna(age_coef) else 0.0) + (inter_coef if not pd.isna(inter_coef) else 0.0)
                try:
                    var_a = cov.loc[age_name, age_name]
                    var_b = cov.loc[inter_name, inter_name]
                    cov_ab = cov.loc[age_name, inter_name]
                    se = np.sqrt(var_a + var_b + 2 * cov_ab)
                except Exception:
                    se = np.nan
                if (se is not None) and (not pd.isna(se)) and (se != 0) and stats is not None:
                    try:
                        z = slope / se
                        p = float(2 * (1 - stats.norm.cdf(abs(z))))
                    except Exception:
                        z = None
                        p = None
                else:
                    z = None
                    p = None
                slopes_by_site[str(site_label)] = {
                    "slope": maybe_float(slope),
                    "se": maybe_float(se),
                    "z": maybe_float(z),
                    "p": maybe_float(p)
                }
                interaction_pvalues[str(inter_name)] = maybe_float(pvalues.get(inter_name, np.nan))

    # Package the object to return
    out_object = {
        "age_coef": maybe_float(age_coef),
        "age_pvalue": maybe_float(age_p),
        "age_ci": [maybe_float(age_ci[0]), maybe_float(age_ci[1])],
        "age_sq_coef": maybe_float(age_sq_coef),
        "age_sq_pvalue": maybe_float(age_sq_p),
        "age_sq_ci": [maybe_float(age_sq_ci[0]), maybe_float(age_sq_ci[1])],
        "age_marginal_effect_overall": maybe_float(age_margeff),
        "age_marginal_effect_se": maybe_float(age_margeff_se),
        "age_slopes_by_site": slopes_by_site,
        "interaction_terms_pvalues": interaction_pvalues,
        "model_aic": maybe_float(getattr(model_fit, 'aic', None)),
        "model_bic": maybe_float(getattr(model_fit, 'bic', None))
    }

    # Build a concise description interpreting the main results
    desc_lines = []

    # Helper for safe formatted printing
    def fmt(val, digits=4):
        return f"{val:.{digits}f}" if (val is not None) else "NA"

    # Main effect interpretation
    if out_object["age_pvalue"] is not None:
        sig = "statistically significant" if out_object["age_pvalue"] < 0.05 else "not statistically significant"
        desc_lines.append(
            f"The linear age term (Age_c) has coefficient {fmt(out_object['age_coef'])} (p = {out_object['age_pvalue']:.3g}), which is {sig}."
        )
    else:
        desc_lines.append("Could not retrieve p-value for the linear age term (Age_c).")

    if out_object["age_sq_pvalue"] is not None:
        sig2 = "statistically significant" if out_object["age_sq_pvalue"] < 0.05 else "not statistically significant"
        nonlinear_text = "a nonlinear" if out_object['age_sq_pvalue'] < 0.05 else "no clear nonlinear"
        desc_lines.append(
            f"The quadratic age term (Age_sq) has coefficient {fmt(out_object['age_sq_coef'])} (p = {out_object['age_sq_pvalue']:.3g}), which is {sig2}, indicating {nonlinear_text} age trajectory."
        )
    else:
        desc_lines.append("Could not retrieve p-value for the quadratic age term (Age_sq).")

    # Marginal effect
    if out_object["age_marginal_effect_overall"] is not None:
        me_val = fmt(out_object["age_marginal_effect_overall"], digits=4)
        me_se = fmt(out_object["age_marginal_effect_se"], digits=4) if out_object["age_marginal_effect_se"] is not None else "NA"
        desc_lines.append(
            f"The average marginal effect of Age_c on the probability of choosing the majority (overall) is approximately {me_val} (SE {me_se})."
        )
    else:
        desc_lines.append("Average marginal effect for Age_c was not available in the output or could not be extracted.")

    # Site-specific slopes summary: list sites where slope differs significantly from zero
    sig_sites = []
    nonsig_sites = []
    for site, info in out_object['age_slopes_by_site'].items():
        p = info.get('p', None)
        slope = info.get('slope', None)
        if p is None:
            nonsig_sites.append(site)
        else:
            if p < 0.05:
                sig_sites.append((site, slope, p))
            else:
                nonsig_sites.append(site)
    if sig_sites:
        sig_text = "; ".join([f"{s[0]} (slope={fmt(s[1])}, p={s[2]:.3g})" for s in sig_sites])
        desc_lines.append("Sites showing significant age-related change in majority choice: " + sig_text + ".")
    else:
        desc_lines.append("No site shows a clearly significant age-related change in majority choice based on site-specific slope tests (individual p-values).")

    # Interactions overall
    if out_object['interaction_terms_pvalues']:
        n_inter = len(out_object['interaction_terms_pvalues'])
        n_signif = sum(1 for p in out_object['interaction_terms_pvalues'].values() if (p is not None and p < 0.05))
        desc_lines.append(f"{n_signif} of {n_inter} Age-by-Site interaction parameters are individually significant (p < 0.05), suggesting potential variation in developmental trajectories across sites.")
    else:
        desc_lines.append("No Age-by-Site interaction parameters were found or extractable; this limits inference about cross-cultural variation in developmental slopes.")

    description = " ".join(desc_lines)

    res['object'] = out_object
    res['description'] = description
    return res