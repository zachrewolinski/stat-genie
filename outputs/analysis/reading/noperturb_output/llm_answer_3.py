def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View on reading speed for:
      - non-dyslexic readers (main effect of reader_view)
      - dyslexic readers (main effect + interaction)

    Assumes model_output is a statsmodels MixedLMResults (or its wrapper) from the
    model:
      log_speed ~ reader_view * dyslexia_bin + ...  (with random intercepts)

    Returns a dictionary with keys:
      - "object": dict with numeric results (betas, SEs, 95% CIs, p-values,
                  percent change on original speed scale)
      - "description": textual interpretation answering whether Reader View
                       improves reading speed for individuals with dyslexia.
    """
    import numpy as np
    import scipy.stats as st

    res = model_output

    # Fixed-effect parameter estimates
    try:
        fe = res.fe_params  # pandas Series
    except Exception:
        # fallback to params if fe_params not available
        fe = res.params

    param_names = list(fe.index)

    # Find names for reader_view, interaction
    def find_param(containing):
        for name in param_names:
            if all(token in name for token in containing):
                return name
        return None

    # Try common possibilities
    reader_name = find_param(['reader_view'])
    dys_name = find_param(['dyslexia_bin'])
    interaction_name = find_param(['reader_view', 'dyslexia_bin'])

    if reader_name is None:
        raise ValueError("Could not find a fixed-effect parameter name containing 'reader_view' in model's fixed effects.")
    if dys_name is None:
        # dyslexia main effect might not be needed for calculation, but warn
        pass
    if interaction_name is None:
        # If no interaction term found, we'll treat as no interaction (i.e., same effect)
        interaction_present = False
    else:
        interaction_present = True

    # Obtain covariance matrix for fixed effects
    # cov_params() may return a DataFrame or ndarray; try to index by fe.index
    try:
        cov_all = res.cov_params()
        # Use only rows/cols that correspond to fixed effects if possible
        try:
            cov_fe = cov_all.loc[param_names, param_names]
        except Exception:
            # cov_all might be ndarray; try to convert using param_names order
            cov_fe = np.asarray(cov_all)
            # If shapes mismatch, fallback to diag from bse_fe if available
            if cov_fe.shape[0] != len(param_names):
                cov_fe = None
    except Exception:
        cov_fe = None

    # If we couldn't build cov_fe, try using bse for SEs (less ideal: no covariance)
    if cov_fe is None:
        # Try to get standard errors for fixed effects
        try:
            bse = res.bse_fe
            # convert to Series aligned with param_names if needed
            if hasattr(bse, 'index'):
                bse_ser = bse.reindex(param_names)
                var_ser = (bse_ser ** 2).to_dict()
            else:
                bse_ser = np.asarray(bse)
                var_ser = dict(zip(param_names, bse_ser ** 2))
            cov_fe = np.zeros((len(param_names), len(param_names)))
            for i, name in enumerate(param_names):
                cov_fe[i, i] = var_ser.get(name, np.nan)
            cov_fe = np.asarray(cov_fe)
        except Exception:
            raise RuntimeError("Could not extract covariance matrix or standard errors for fixed effects from model output.")

    # Helper to extract beta and variance for a given parameter name
    def get_beta_var(name):
        if name is None:
            return 0.0, 0.0
        beta = float(fe[name])
        try:
            # cov_fe might be DataFrame or ndarray
            if hasattr(cov_fe, 'loc'):
                var = float(cov_fe.loc[name, name])
            else:
                idx = param_names.index(name)
                var = float(cov_fe[idx, idx])
        except Exception:
            # fallback: try bse_fe if available
            try:
                bse = res.bse_fe[name]
                var = float(bse ** 2)
            except Exception:
                var = np.nan
        return beta, var

    # Non-dyslexic effect: coefficient on reader_view
    beta_rv, var_rv = get_beta_var(reader_name)
    se_rv = np.sqrt(var_rv) if var_rv >= 0 else np.nan
    z_rv = beta_rv / se_rv if se_rv and not np.isnan(se_rv) else np.nan
    p_rv = 2 * (1 - st.norm.cdf(abs(z_rv))) if not np.isnan(z_rv) else np.nan
    ci_rv = (beta_rv - 1.96 * se_rv, beta_rv + 1.96 * se_rv) if not np.isnan(se_rv) else (np.nan, np.nan)

    # Dyslexic effect: sum of reader_view + interaction
    if interaction_present:
        beta_int, var_int = get_beta_var(interaction_name)
        # variance of sum = var_rv + var_int + 2*cov(r, int)
        try:
            if hasattr(cov_fe, 'loc'):
                cov_r_int = float(cov_fe.loc[reader_name, interaction_name])
            else:
                i = param_names.index(reader_name)
                j = param_names.index(interaction_name)
                cov_r_int = float(cov_fe[i, j])
        except Exception:
            cov_r_int = 0.0
        beta_dys = beta_rv + beta_int
        var_dys = var_rv + var_int + 2 * cov_r_int
    else:
        # No interaction term: effect is same as main effect
        beta_int = 0.0
        cov_r_int = 0.0
        beta_dys = beta_rv
        var_dys = var_rv

    se_dys = np.sqrt(var_dys) if var_dys >= 0 else np.nan
    z_dys = beta_dys / se_dys if se_dys and not np.isnan(se_dys) else np.nan
    p_dys = 2 * (1 - st.norm.cdf(abs(z_dys))) if not np.isnan(z_dys) else np.nan
    ci_dys = (beta_dys - 1.96 * se_dys, beta_dys + 1.96 * se_dys) if not np.isnan(se_dys) else (np.nan, np.nan)

    # Convert log-scale effects to multiplicative percent change in speed
    pct_rv = (np.exp(beta_rv) - 1) * 100 if not np.isnan(beta_rv) else np.nan
    pct_dys = (np.exp(beta_dys) - 1) * 100 if not np.isnan(beta_dys) else np.nan
    # CIs on multiplicative scale
    ci_rv_pct = (np.exp(ci_rv[0]) - 1) * 100, (np.exp(ci_rv[1]) - 1) * 100 if not np.isnan(ci_rv[0]) else (np.nan, np.nan)
    ci_dys_pct = (np.exp(ci_dys[0]) - 1) * 100, (np.exp(ci_dys[1]) - 1) * 100 if not np.isnan(ci_dys[0]) else (np.nan, np.nan)

    # Build object to return
    result_obj = {
        "non_dyslexic": {
            "param_name": reader_name,
            "beta_log_speed": beta_rv,
            "se": se_rv,
            "95%_CI_log": ci_rv,
            "p_value": p_rv,
            "percent_change_speed": pct_rv,
            "95%_CI_percent_change": ci_rv_pct,
        },
        "dyslexic": {
            "param_name": f"{reader_name} + {interaction_name}" if interaction_present else reader_name,
            "beta_log_speed": beta_dys,
            "se": se_dys,
            "95%_CI_log": ci_dys,
            "p_value": p_dys,
            "percent_change_speed": pct_dys,
            "95%_CI_percent_change": ci_dys_pct,
        },
        "notes": {
            "reader_param": reader_name,
            "interaction_param": interaction_name if interaction_present else None,
            "covariance_reader_interaction": cov_r_int if interaction_present else None
        }
    }

    # Simple textual conclusion for dyslexic readers
    alpha = 0.05
    if not np.isnan(p_dys):
        if p_dys < alpha:
            if beta_dys > 0:
                conclusion = (
                    "Reader View increases reading speed for individuals with dyslexia: "
                    f"estimated effect = {pct_dys:.2f}% faster (95% CI {ci_dys_pct[0]:.2f}%, {ci_dys_pct[1]:.2f}%), "
                    f"p = {p_dys:.3g}."
                )
            else:
                conclusion = (
                    "Reader View decreases reading speed for individuals with dyslexia: "
                    f"estimated effect = {pct_dys:.2f}% (negative) (95% CI {ci_dys_pct[0]:.2f}%, {ci_dys_pct[1]:.2f}%), "
                    f"p = {p_dys:.3g}."
                )
        else:
            conclusion = (
                "No statistically significant effect of Reader View on reading speed for individuals with dyslexia "
                f"(estimated {pct_dys:.2f}% change, 95% CI {ci_dys_pct[0]:.2f}% to {ci_dys_pct[1]:.2f}%, p = {p_dys:.3g})."
            )
    else:
        conclusion = "Could not compute p-value for dyslexic effect."

    description = (
        "This output gives the estimated effect of turning Reader View on (relative to off) on log(reading speed), "
        "for non-dyslexic readers (the main reader_view coefficient) and for dyslexic readers (main + interaction). "
        "Results are provided as: beta on log-speed, standard error, 95% CI on log scale, two-sided p-value, "
        "and percent change on the original speed scale with 95% CI. "
        "Conclusion: " + conclusion
    )

    return {"object": result_obj, "description": description}