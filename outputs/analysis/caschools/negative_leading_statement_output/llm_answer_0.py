def extract_final_answer(model_output):
    """
    Extracts statistics about the StudentTeacherRatio effect from a fitted statsmodels regression result.

    Returns a dictionary:
      - "object": dict with numeric results (coefficients, p-values, CIs, marginal effect at mean, etc.)
      - "description": text interpretation answering whether a lower student-teacher ratio is associated
                       with higher academic performance based on the estimated marginal effect.
    """
    import numpy as np
    from math import sqrt
    # scipy is used for p-value from t-distribution; fall back to normal if scipy not available
    try:
        from scipy import stats
        has_scipy = True
    except Exception:
        has_scipy = False

    res = model_output  # statsmodels RegressionResultsWrapper

    # Extract parameters, bse, pvalues, conf_int, covariance matrix
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    try:
        conf_int_df = res.conf_int()
    except Exception:
        # If conf_int fails, compute approximate 95% CI using normal approx
        conf_int_df = None

    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Identify names for linear and squared StudentTeacherRatio terms
    names = list(params.index)
    linear_names = [n for n in names if ('StudentTeacherRatio' in n) and ('I(' not in n)]
    squared_names = [n for n in names if ('StudentTeacherRatio' in n) and ('I(' in n)]

    # Fallback if identification failed
    linear_name = linear_names[0] if linear_names else ('StudentTeacherRatio' if 'StudentTeacherRatio' in params.index else None)
    sq_name = squared_names[0] if squared_names else None

    results = {}

    if linear_name is None:
        raise ValueError("Could not find a parameter name for StudentTeacherRatio in the model parameters.")

    # Linear term stats
    beta1 = float(params[linear_name])
    se1 = float(bse[linear_name]) if linear_name in bse.index else None
    p1 = float(pvalues[linear_name]) if linear_name in pvalues.index else None
    if conf_int_df is not None and linear_name in conf_int_df.index:
        ci1 = tuple(conf_int_df.loc[linear_name].astype(float).tolist())
    else:
        # approximate 95% CI
        if se1 is not None:
            ci1 = (beta1 - 1.96 * se1, beta1 + 1.96 * se1)
        else:
            ci1 = (None, None)

    results['linear_term'] = {
        'name': linear_name,
        'coef': beta1,
        'se': se1,
        'p_value': p1,
        '95%_ci': ci1
    }

    # Squared term stats (if present)
    if sq_name is not None:
        beta2 = float(params[sq_name])
        se2 = float(bse[sq_name]) if sq_name in bse.index else None
        p2 = float(pvalues[sq_name]) if sq_name in pvalues.index else None
        if conf_int_df is not None and sq_name in conf_int_df.index:
            ci2 = tuple(conf_int_df.loc[sq_name].astype(float).tolist())
        else:
            if se2 is not None:
                ci2 = (beta2 - 1.96 * se2, beta2 + 1.96 * se2)
            else:
                ci2 = (None, None)
        results['squared_term'] = {
            'name': sq_name,
            'coef': beta2,
            'se': se2,
            'p_value': p2,
            '95%_ci': ci2
        }
    else:
        beta2 = 0.0
        results['squared_term'] = None

    # Compute marginal effect of StudentTeacherRatio on AvgScore.
    # If quadratic included: marginal = beta1 + 2 * beta2 * x
    # We'll compute it at the sample mean of StudentTeacherRatio if the original dataframe is available.
    x_mean = None
    try:
        data_frame = getattr(res.model.data, 'frame', None)
        if data_frame is None:
            # try different attribute name that sometimes exists
            data_frame = getattr(res.model.data, 'orig_exog', None)
        if data_frame is not None:
            # data_frame might be a DataFrame or an array-like; try to get the column
            if hasattr(data_frame, 'columns') and 'StudentTeacherRatio' in data_frame.columns:
                x_mean = float(data_frame['StudentTeacherRatio'].mean())
            else:
                # attempt to access from model.data.x or exog; fallback to None
                try:
                    exog_df = res.model.data.frame  # try again
                    if exog_df is not None and 'StudentTeacherRatio' in exog_df.columns:
                        x_mean = float(exog_df['StudentTeacherRatio'].mean())
                except Exception:
                    x_mean = None
    except Exception:
        x_mean = None

    # If mean not found, use zero as fallback (marginal at 0) and indicate that in description
    used_x = x_mean if x_mean is not None else 0.0

    # Marginal effect and its standard error using Delta method if covariance available
    marg = beta1 + 2.0 * beta2 * used_x
    marg_se = None
    marg_p = None
    marg_ci = (None, None)
    if cov is not None and linear_name in cov.index:
        try:
            if sq_name is not None and sq_name in cov.index:
                # Var(marg) = Var(beta1) + (2x)^2 Var(beta2) + 2*(2x)*Cov(beta1,beta2)
                var_marg = cov.loc[linear_name, linear_name]
                var_marg = var_marg + (2.0 * used_x) ** 2 * cov.loc[sq_name, sq_name]
                var_marg = var_marg + 2.0 * (2.0 * used_x) * cov.loc[linear_name, sq_name]
            else:
                var_marg = cov.loc[linear_name, linear_name]
            marg_se = float(sqrt(max(var_marg, 0.0)))
            # t-stat for marginal
            t_stat = marg / marg_se if marg_se != 0 else np.nan
            # p-value: use t-distribution with resid df if scipy available, else normal approx
            if has_scipy:
                df_resid = getattr(res, 'df_resid', None)
                if df_resid is None:
                    marg_p = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
                else:
                    marg_p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_resid)))
            else:
                marg_p = float(2.0 * (1.0 - (0.5 * (1.0 + (1.0 if t_stat >= 0 else -1.0) * 0))))  # fallback nonsense handled below
                # fallback to normal if scipy not available:
                marg_p = float(2.0 * (1.0 - (0.5 * (1.0 + np.math.erf(abs(t_stat) / sqrt(2))))))

            marg_ci = (marg - 1.96 * marg_se, marg + 1.96 * marg_se)
        except Exception:
            marg_se = None
            marg_p = None
            marg_ci = (None, None)
    else:
        # No covariance: fall back to using linear term's se for marginal if no quadratic, else unknown
        if beta2 == 0.0 and results['linear_term']['se'] is not None:
            marg_se = results['linear_term']['se']
            t_stat = marg / marg_se if marg_se != 0 else np.nan
            if has_scipy:
                df_resid = getattr(res, 'df_resid', None)
                if df_resid is None:
                    marg_p = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
                else:
                    marg_p = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_resid)))
            else:
                marg_p = float(2.0 * (1.0 - (0.5 * (1.0 + np.math.erf(abs(t_stat) / sqrt(2))))))
            marg_ci = (marg - 1.96 * marg_se, marg + 1.96 * marg_se)

    results['marginal_at_mean'] = {
        'x_mean_used': used_x,
        'marginal_effect': marg,
        'marginal_se': marg_se,
        'marginal_p_value': marg_p,
        'marginal_95%_ci': marg_ci,
        'note': ("x_mean computed from model data" if x_mean is not None else
                 "x_mean not available from model object; marginal effect computed at 0.0 as fallback")
    }

    # Interpretation: lower student-teacher ratio => smaller numeric StudentTeacherRatio.
    # So a negative marginal effect means that reducing StudentTeacherRatio (fewer students per teacher)
    # is associated with higher AvgScore.
    interpret = ""
    sig_threshold = 0.05
    marg_effect = marg
    marg_pval = marg_p

    if marg_pval is None:
        interpret = ("Could not compute a reliable p-value for the marginal effect. "
                     "See the extracted coefficients and CIs for assessment.")
    else:
        if (marg_effect < 0) and (marg_pval < sig_threshold):
            interpret = ("Yes — the estimated marginal effect of StudentTeacherRatio on AvgScore at the sample mean "
                         "is negative and statistically significant (p < {:.3g}). This implies that a lower "
                         "student-teacher ratio (fewer students per teacher) is associated with higher average "
                         "academic performance.").format(sig_threshold)
        elif (marg_effect < 0) and (marg_pval >= sig_threshold):
            interpret = ("The estimated marginal effect is negative (suggesting that lower student-teacher ratio "
                         "is associated with higher AvgScore), but it is not statistically significant (p = {:.3g}). "
                         "This is weak evidence and not definitive.").format(marg_pval)
        elif (marg_effect > 0) and (marg_pval < sig_threshold):
            interpret = ("Contrary to the hypothesis, the estimated marginal effect of StudentTeacherRatio on AvgScore "
                         "at the sample mean is positive and statistically significant (p < {:.3g}). That would imply "
                         "higher StudentTeacherRatio (more students per teacher) is associated with higher AvgScore. "
                         "This is an unexpected result and should be investigated further for model misspecification.").format(sig_threshold)
        else:
            interpret = ("The estimated marginal effect at the sample mean is positive but not statistically significant "
                         "(p = {:.3g}), providing no evidence that lower student-teacher ratio is associated with higher performance.").format(marg_pval)

    description_lines = []
    description_lines.append("Extracted statistics for StudentTeacherRatio effect:")
    description_lines.append(f"- Linear term '{linear_name}': coef = {beta1:.6g}, se = {se1:.6g}, p = {p1:.6g}, 95% CI = {ci1}".replace("{ci1}", str(ci1)))
    if results['squared_term'] is not None:
        s = results['squared_term']
        description_lines.append(f"- Squared term '{sq_name}': coef = {s['coef']:.6g}, se = {s['se']:.6g}, p = {s['p_value']:.6g}, 95% CI = {s['95%_ci']}")
    description_lines.append(f"- Marginal effect at StudentTeacherRatio = {used_x:.4g}: marginal = {marg:.6g}, se = {marg_se}, p = {marg_p}, 95% CI = {marg_ci}")
    description_lines.append("")
    description_lines.append("Interpretation (direction and significance):")
    description_lines.append(interpret)

    description = "\n".join(description_lines)

    return {
        "object": results,
        "description": description
    }