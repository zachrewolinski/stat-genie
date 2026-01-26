def extract_final_answer(model_output):
    """
    Extracts fixed-effect estimates and inferential statistics from a fitted
    statsmodels MixedLMResults (or MixedLMResultsWrapper) object.

    Returns a dictionary with:
      - "object": a pandas.DataFrame (index = fixed-effect names) with columns:
            coef, se, z, p_value, ci_lower, ci_upper, pct_change (approx)
        where pct_change = (exp(coef)-1)*100, i.e. approximate percent change
        in Efficiency_per_min for a one-unit change in the predictor (or for
        the category vs reference for categorical predictors), because the
        model was fit on log(Efficiency_per_min + 1e-6).
      - "description": a short explanation of what the table contains and how
        to interpret the results in the context of the task (effects of age,
        sex, and help on nut-cracking efficiency).

    The function is robust to different statsmodels versions (falls back to
    computing p-values and CIs if they are not provided).
    """
    import numpy as np
    import pandas as pd
    try:
        from scipy.stats import norm
    except Exception:
        # if scipy is not available, use approximation from numpy (less ideal)
        def _norm_cdf(x):
            return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))
        norm = type("norm", (), {"cdf": staticmethod(_norm_cdf)})

    # Attempt to get fixed-effect parameter names and estimates
    # Prefer fe_params if available (works for MixedLMResults), otherwise use model.exog_names
    if hasattr(model_output, "fe_params"):
        fe_params = model_output.fe_params
    else:
        # fallback: select params corresponding to model.exog_names
        exog_names = None
        if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            exog_names = list(model_output.model.exog_names)
        if exog_names is None:
            # last resort: take all params
            fe_params = getattr(model_output, "params")
        else:
            params = getattr(model_output, "params")
            try:
                fe_params = params.loc[exog_names]
            except Exception:
                # if params is ndarray or ordering differs, attempt to coerce to pandas Series
                try:
                    params_series = pd.Series(params, index=getattr(params, "index", None))
                    fe_params = params_series.loc[exog_names]
                except Exception:
                    # fallback to taking first len(exog_names) elements
                    fe_params = pd.Series(np.asarray(params)[: len(exog_names)], index=exog_names)

    # Ensure fe_params is a pandas Series with index
    if not hasattr(fe_params, "index"):
        try:
            fe_params = pd.Series(fe_params)
        except Exception:
            fe_params = pd.Series(list(fe_params))

    fe_names = list(fe_params.index)
    coefs = np.asarray(fe_params, dtype=float)

    # Standard errors: try common attributes, otherwise compute from cov_params
    se_vals = None
    if hasattr(model_output, "bse_fe"):
        try:
            se_vals = np.asarray(model_output.bse_fe.loc[fe_names], dtype=float)
        except Exception:
            se_vals = np.asarray(model_output.bse_fe, dtype=float)
    elif hasattr(model_output, "bse"):
        # bse might contain all params; select fixed-effect names if present
        bse_all = getattr(model_output, "bse")
        try:
            se_vals = np.asarray(bse_all.loc[fe_names], dtype=float)
        except Exception:
            # if bse is an array-like in same order as params, attempt to match by index
            try:
                se_vals = np.asarray(bse_all[: len(coefs)], dtype=float)
            except Exception:
                se_vals = None

    if se_vals is None:
        # compute from covariance matrix if possible
        if hasattr(model_output, "cov_params"):
            try:
                cov = model_output.cov_params()
                try:
                    # cov may be DataFrame: select rows/cols corresponding to fe_names
                    cov_fe = cov.loc[fe_names, fe_names]
                    se_vals = np.sqrt(np.diag(cov_fe)).astype(float)
                except Exception:
                    # if cov is ndarray and in same order:
                    cov_arr = np.asarray(cov)
                    se_vals = np.sqrt(np.diag(cov_arr)[: len(coefs)]).astype(float)
            except Exception:
                se_vals = None
        else:
            se_vals = None

    if se_vals is None:
        # as a last resort, set NA
        se_vals = np.array([np.nan] * len(coefs), dtype=float)

    # z-statistics and p-values
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stats = coefs / se_vals
        # p-values: try model_output.pvalues if available
        p_vals = None
        if hasattr(model_output, "pvalues"):
            try:
                p_all = model_output.pvalues
                # select fixed effects p-values if possible
                try:
                    p_vals = np.asarray(p_all.loc[fe_names], dtype=float)
                except Exception:
                    # maybe p_all is array-like
                    p_vals = np.asarray(p_all[: len(coefs)], dtype=float)
            except Exception:
                p_vals = None
        if p_vals is None:
            # compute two-sided p-values from z
            p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_stats)))

    # Confidence intervals: try conf_int()
    ci_lower = np.empty(len(coefs), dtype=float)
    ci_upper = np.empty(len(coefs), dtype=float)
    if hasattr(model_output, "conf_int"):
        try:
            ci = model_output.conf_int()
            # conf_int may return array or DataFrame
            if isinstance(ci, pd.DataFrame):
                try:
                    ci_sel = ci.loc[fe_names]
                    ci_lower = np.asarray(ci_sel.iloc[:, 0], dtype=float)
                    ci_upper = np.asarray(ci_sel.iloc[:, 1], dtype=float)
                except Exception:
                    # fallback to matching order if possible
                    ci_arr = np.asarray(ci)
                    ci_lower = ci_arr[: len(coefs), 0].astype(float)
                    ci_upper = ci_arr[: len(coefs), 1].astype(float)
            else:
                # assume numpy array in same order
                ci_arr = np.asarray(ci)
                if ci_arr.ndim == 2 and ci_arr.shape[0] >= len(coefs):
                    ci_lower = ci_arr[: len(coefs), 0].astype(float)
                    ci_upper = ci_arr[: len(coefs), 1].astype(float)
                else:
                    ci_lower[:] = coefs - 1.96 * se_vals
                    ci_upper[:] = coefs + 1.96 * se_vals
        except Exception:
            ci_lower[:] = coefs - 1.96 * se_vals
            ci_upper[:] = coefs + 1.96 * se_vals
    else:
        ci_lower[:] = coefs - 1.96 * se_vals
        ci_upper[:] = coefs + 1.96 * se_vals

    # Percent change interpretation on original scale
    # model used log(Eff + 1e-6) so exp(beta)-1 ~ proportional change in Efficiency_per_min
    pct_change = (np.exp(coefs) - 1.0) * 100.0

    summary_df = pd.DataFrame({
        "coef": coefs,
        "se": se_vals,
        "z": z_stats,
        "p_value": p_vals,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pct_change_%": pct_change
    }, index=fe_names)

    # Round numeric columns for readability (but keep full precision in DataFrame)
    try:
        summary_display = summary_df.round({
            "coef": 4, "se": 4, "z": 3, "p_value": 4,
            "ci_lower": 4, "ci_upper": 4, "pct_change_%": 2
        })
    except Exception:
        summary_display = summary_df

    # Construct description explaining how to interpret the table
    desc_lines = [
        "This table shows the fixed-effects estimates from the mixed-effects model predicting",
        "log(Efficiency_per_min + 1e-6). Columns:",
        "- coef: estimated coefficient on the log scale.",
        "- se: standard error of the coefficient.",
        "- z: test statistic (coef / se).",
        "- p_value: two-sided p-value for H0: coef = 0 (may be computed if not provided by the model).",
        "- ci_lower / ci_upper: 95% confidence interval for the coefficient (on log scale).",
        "- pct_change_%: (exp(coef)-1)*100, approximate percent change in Efficiency_per_min",
        "  associated with a one-unit increase in the predictor (or for that category vs the reference),",
        "  because the model was fit on the log-transformed outcome.",
        "",
        "To answer the research question (How do age, sex, and receiving help influence nut-cracking efficiency?),",
        "examine the rows corresponding to 'age', the sex indicator(s) (e.g., 'C(sex)[T.M]' if male is compared to female),",
        "and the help indicator(s) (e.g., 'C(help)[T.yes]').",
        "- A statistically significant positive coef (p < 0.05) means the predictor is associated with higher",
        "  log-efficiency, i.e. higher efficiency; the pct_change_% column gives the approximate percent increase.",
        "- A statistically significant negative coef means lower efficiency (pct_change_% negative).",
        "",
        "Returned object: a pandas.DataFrame (index = fixed-effect names) summarising estimates.",
        "Note: categorical effects are relative to the reference level chosen by the model (check the parameter names)."
    ]
    description = "\n".join(desc_lines)

    # Return the detailed DataFrame and a user-friendly description
    return {"object": summary_df, "description": description}