def extract_final_answer(model_output):
    """
    Extracts statistics on the effect of the 'female' indicator from a fitted
    statsmodels Logit result and its marginal effects (if present).

    Returns:
      {
        "object": {
            "logit_coef": float,
            "logit_se": float,
            "logit_z": float,
            "logit_p": float,
            "logit_ci95": [lower, upper],
            "ame": float or None,             # average marginal effect (change in prob. of deny)
            "ame_se": float or None,
            "ame_p": float or None,
            "ame_ci95": [lower, upper] or [None, None],
            "conclusion": str                 # brief yes/no style conclusion re: gender effect
        },
        "description": str
      }
    """
    import math
    from scipy import stats

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'logit_result' and optionally 'marginal_effects'.")

    res = model_output.get('logit_result')
    me = model_output.get('marginal_effects', None)

    if res is None:
        raise ValueError("model_output does not contain 'logit_result'.")

    # Ensure 'female' is in the model
    try:
        param_index = list(res.params.index)
    except Exception as e:
        raise ValueError("Unable to read parameters from logit_result.") from e

    if 'female' not in param_index:
        raise ValueError("'female' not found among fitted model parameters.")

    # Extract logit coefficient statistics
    coef = float(res.params['female'])
    se = float(res.bse['female'])
    # tvalues / zvalues available as .tvalues (or .zvalues), fall back to coef/se
    try:
        z = float(res.tvalues['female'])
    except Exception:
        try:
            z = float(res.zvalues['female'])
        except Exception:
            z = coef / se if se != 0 else float('nan')
    pval = float(res.pvalues['female'])

    # Confidence interval (95%)
    ci = res.conf_int()
    try:
        ci_low, ci_high = ci.loc['female']
    except Exception:
        # If conf_int returns ndarray without index labels
        idx = param_index.index('female')
        ci_low, ci_high = ci[idx]

    # Extract average marginal effect (AME) if available
    ame = None
    ame_se = None
    ame_p = None
    ame_ci = (None, None)

    if me is not None:
        # Prefer structured summary_frame if available
        try:
            me_df = me.summary_frame()
            # Identify row for 'female' (index might be variable names)
            if 'female' in me_df.index:
                row = me_df.loc['female']
            else:
                # If index is numeric, try to find index position matching parameter order
                try:
                    idx = param_index.index('female')
                    row = me_df.iloc[idx]
                except Exception:
                    row = None

            if row is not None:
                # Common column names: 'dy/dx', 'Std. Err.', 'z', 'P>|z|', maybe CI columns
                cols = list(me_df.columns)
                # dy/dx is usually the first column
                ame = float(row[cols[0]])
                # try to get std err and p-value if present
                if len(cols) >= 2:
                    ame_se = float(row[cols[1]])
                # p-value often in a column named 'P>|z|' or the 3rd column
                pcol = None
                for cname in cols:
                    if cname.strip().lower().startswith('p'):
                        pcol = cname
                        break
                if pcol is not None:
                    try:
                        ame_p = float(row[pcol])
                    except Exception:
                        ame_p = None
                # 95% CI often the last two columns
                if len(cols) >= 2:
                    try:
                        ame_ci = (float(row[cols[-2]]), float(row[cols[-1]]))
                    except Exception:
                        ame_ci = (None, None)
        except Exception:
            # Fallback to attributes on the marginal effects object (if present)
            try:
                marg = getattr(me, 'margeff', None) or getattr(me, 'margeff_', None)
                marg_se_arr = getattr(me, 'margeff_se', None) or getattr(me, 'margeff_se_', None)
                if marg is not None:
                    idx = param_index.index('female')
                    ame = float(marg[idx])
                    if marg_se_arr is not None:
                        ame_se = float(marg_se_arr[idx])
                        # approximate p-value and CI
                        if ame_se != 0:
                            z_ame = ame / ame_se
                            ame_p = 2 * (1 - stats.norm.cdf(abs(z_ame)))
                            ame_ci = (ame - 1.96 * ame_se, ame + 1.96 * ame_se)
            except Exception:
                # give up on AME extraction
                ame = ame_se = ame_p = None
                ame_ci = (None, None)

    # Formulate a conclusion about whether gender affects denial
    # Use AME p-value if available, else use logit p-value
    significance_level = 0.05
    effect_stat_p = ame_p if (ame_p is not None) else pval
    if effect_stat_p is None or (isinstance(effect_stat_p, float) and math.isnan(effect_stat_p)):
        conclusion = "Could not determine statistical significance for the 'female' effect (p-value not available)."
    else:
        if effect_stat_p < significance_level:
            # direction: positive coef -> higher log-odds of denial for females
            if coef > 0:
                direction = "Female applicants are statistically significantly more likely to be denied (higher probability of denial)."
            elif coef < 0:
                direction = "Female applicants are statistically significantly less likely to be denied (lower probability of denial)."
            else:
                direction = "Coefficient is effectively zero though statistically significant (unusual)."
            conclusion = f"Yes — there is a statistically significant difference by gender (p = {effect_stat_p:.3g}). {direction}"
        else:
            conclusion = (
                f"No strong evidence that banks treat female applicants differently from male applicants "
                f"(p = {effect_stat_p:.3g} > {significance_level})."
            )

    # Prepare the object to return (numbers + conclusion)
    output_obj = {
        "logit_coef": coef,
        "logit_se": se,
        "logit_z": z,
        "logit_p": pval,
        "logit_ci95": [float(ci_low), float(ci_high)],
        "ame": (float(ame) if ame is not None else None),
        "ame_se": (float(ame_se) if ame_se is not None else None),
        "ame_p": (float(ame_p) if ame_p is not None else None),
        "ame_ci95": [float(ame_ci[0]) if ame_ci[0] is not None else None,
                     float(ame_ci[1]) if ame_ci[1] is not None else None],
        "conclusion": conclusion
    }

    description = (
        "Extracted statistics for the 'female' indicator from a fitted logistic regression "
        "predicting mortgage denial (deny=1). 'logit_coef' is the estimated log-odds coefficient "
        "for female (positive => higher log-odds of denial). 'logit_p' and 'logit_ci95' are the "
        "associated p-value and 95% confidence interval. 'ame' is the average marginal effect "
        "(interpretable as the approximate change in probability of denial associated with being female, "
        "in probability units, if available) with its standard error, p-value, and 95% CI. "
        "The 'conclusion' field gives a brief yes/no style answer about whether there is evidence of "
        "a gender difference, based on the AME p-value when available and otherwise the logit p-value."
    )

    return {"object": output_obj, "description": description}