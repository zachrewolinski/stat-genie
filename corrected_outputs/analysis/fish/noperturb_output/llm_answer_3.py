def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResults (NegativeBinomial with exposure)
    and return a summary focused on rates per hour.

    Returns a dict with:
      - "object": dict containing coefficient table (coef, se, p, ci),
                  incidence rate ratios (IRR) and their CIs,
                  baseline rate (per hour) and rate for mean covariates, and
                  overdispersion diagnostic if available.
      - "description": human-readable interpretation of the main results.

    Notes:
      - For a GLM with exposure, the model is: log(mu) = log(hours) + X * beta.
        Therefore exp(beta_j) is the multiplicative effect on the expected fish-catching
        rate per hour (incidence rate ratio, IRR) for a one-unit increase in predictor j.
      - The baseline rate per hour (when all predictors = 0) = exp(intercept).
    """
    import numpy as np

    res = model_output  # alias

    # Prepare outputs
    out = {}
    coef_table = {}

    try:
        params = res.params  # pandas Series
        bse = res.bse
        pvals = res.pvalues
        ci = res.conf_int()  # DataFrame/array with two columns: lower, upper (on linear scale)
        var_names = list(params.index)
    except Exception as e:
        raise ValueError("Model output does not expose expected attributes (.params, .bse, .pvalues, .conf_int()).") from e

    # Fill coefficient table and compute IRR (exp(coef)) and CI on IRR scale
    for name in var_names:
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else None
        p = float(pvals[name]) if name in pvals.index else None
        ci_lower = float(ci.loc[name, 0]) if name in ci.index else float(ci[name][0]) if hasattr(ci, "__getitem__") else None
        ci_upper = float(ci.loc[name, 1]) if name in ci.index else float(ci[name][1]) if hasattr(ci, "__getitem__") else None

        irr = float(np.exp(coef)) if coef is not None else None
        irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

        coef_table[name] = {
            "coef": coef,
            "std_error": se,
            "p_value": p,
            "ci_lower (coef scale, 95%)": ci_lower,
            "ci_upper (coef scale, 95%)": ci_upper,
            "IRR (exp(coef))": irr,
            "IRR_CI_lower (95%)": irr_ci_lower,
            "IRR_CI_upper (95%)": irr_ci_upper,
        }

    out['coef_table'] = coef_table

    # Baseline rate per hour (when predictors = 0) is exp(intercept)
    intercept_name = None
    for n in var_names:
        if n.lower() in ['const', 'intercept', 'constant']:
            intercept_name = n
            break

    if intercept_name is not None:
        intercept = params[intercept_name]
        intercept_ci = ci.loc[intercept_name].values if intercept_name in ci.index else None
        baseline_rate_per_hour = float(np.exp(intercept))
        baseline_rate_ci = (float(np.exp(intercept_ci[0])), float(np.exp(intercept_ci[1]))) if intercept_ci is not None else (None, None)
        out['baseline_rate_per_hour'] = baseline_rate_per_hour
        out['baseline_rate_per_hour_CI95'] = baseline_rate_ci
    else:
        out['baseline_rate_per_hour'] = None
        out['baseline_rate_per_hour_CI95'] = (None, None)

    # Expected rate per hour at mean covariate values (use model design matrix means)
    try:
        # model.exog gives the design matrix used for X (including const)
        exog = res.model.exog
        exog_names = list(res.model.exog_names)
        exog_means = np.asarray(exog).mean(axis=0)
        # Create vector aligned with params ordering
        # res.params is typically indexed by exog_names, so dot product with exog_means in same order should work.
        # But to be safe, align by names:
        params_array = np.asarray([params[name] for name in exog_names])
        linpred_mean = float(np.dot(exog_means, params_array))
        rate_mean_per_hour = float(np.exp(linpred_mean))
        # CI for linear predictor at mean covariates:
        covp = res.cov_params()
        # ensure covp is numpy array aligned in exog_names order
        covp_array = np.asarray(covp.loc[exog_names, exog_names]) if hasattr(covp, "loc") else np.asarray(covp)
        se_linpred = float(np.sqrt(np.dot(exog_means, np.dot(covp_array, exog_means))))
        z = 1.96
        lp_ci_lower = linpred_mean - z * se_linpred
        lp_ci_upper = linpred_mean + z * se_linpred
        rate_mean_CI = (float(np.exp(lp_ci_lower)), float(np.exp(lp_ci_upper)))

        out['mean_covariate_rate_per_hour'] = rate_mean_per_hour
        out['mean_covariate_rate_per_hour_CI95'] = rate_mean_CI
        out['mean_covariates_used'] = {name: float(exog_means[i]) for i, name in enumerate(exog_names)}
    except Exception:
        # If anything fails, skip mean-rate computation
        out['mean_covariate_rate_per_hour'] = None
        out['mean_covariate_rate_per_hour_CI95'] = (None, None)
        out['mean_covariates_used'] = None

    # Overdispersion check (Pearson chi2 / df), if attached by the fitting routine
    pearson_per_df = getattr(res, 'pearson_chi2_per_df', None)
    pearson = getattr(res, 'pearson_chi2', None)
    if pearson_per_df is not None:
        out['pearson_chi2'] = float(pearson) if pearson is not None else None
        out['pearson_chi2_per_df'] = float(pearson_per_df)
    else:
        # try to compute if resid_pearson available
        try:
            pearson_chi2 = float((res.resid_pearson ** 2).sum())
            df_resid = float(res.df_resid) if hasattr(res, 'df_resid') else None
            out['pearson_chi2'] = pearson_chi2
            out['pearson_chi2_per_df'] = (pearson_chi2 / df_resid) if df_resid not in (None, 0) else None
        except Exception:
            out['pearson_chi2'] = None
            out['pearson_chi2_per_df'] = None

    # Build human-readable description summarizing the key interpretations
    desc_lines = []
    desc_lines.append("Summary of effects on fish caught PER HOUR (Negative Binomial GLM with log(hours) exposure):")
    desc_lines.append("- For each predictor, IRR = exp(coefficient) is the multiplicative change in expected fish caught per hour for a one-unit increase in that predictor, holding others constant.")
    # Add short statements for key predictors if present
    for var in ['livebait', 'camper', 'total_people']:
        if var in coef_table:
            irr = coef_table[var]['IRR (exp(coef))']
            p = coef_table[var]['p_value']
            ci_l = coef_table[var]['IRR_CI_lower (95%)']
            ci_u = coef_table[var]['IRR_CI_upper (95%)']
            if irr is not None:
                sig = "statistically significant" if (p is not None and p < 0.05) else "not statistically significant"
                desc_lines.append(f"  - {var}: IRR = {irr:.3f} (95% CI {ci_l:.3f}–{ci_u:.3f}), p = {p:.3g} → {sig}.")
    # Baseline rate
    if out['baseline_rate_per_hour'] is not None:
        br = out['baseline_rate_per_hour']
        br_ci = out['baseline_rate_per_hour_CI95']
        desc_lines.append(f"- Baseline expected catch rate (all predictors = 0) = {br:.3f} fish/hour (95% CI {br_ci[0]:.3f}–{br_ci[1]:.3f}).")
    if out['mean_covariate_rate_per_hour'] is not None:
        mr = out['mean_covariate_rate_per_hour']
        mr_ci = out['mean_covariate_rate_per_hour_CI95']
        desc_lines.append(f"- Expected catch rate at the mean covariate values = {mr:.3f} fish/hour (95% CI {mr_ci[0]:.3f}–{mr_ci[1]:.3f}).")
    if out['pearson_chi2_per_df'] is not None:
        desc_lines.append(f"- Overdispersion diagnostic: Pearson chi2 / df = {out['pearson_chi2_per_df']:.3f} (values >>1 suggest overdispersion).")

    description = "\n".join(desc_lines)

    return {"object": out, "description": description}