def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs for the key predictors
    and computes marginal effects of MasFem_z at representative MaxWind_z values
    (mean = 0, +/- 1 SD since MaxWind_z is z-scored). Also constructs a short
    interpretation about whether results support the hypothesis that more feminine
    hurricane names are associated with higher fatalities.

    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "Plain-language summary and interpretation"
      }
    """
    import numpy as np
    from scipy import stats

    res = model_output  # statsmodels RegressionResultsWrapper

    # Helper to safely fetch parameter-related info
    params = res.params
    pvals = res.pvalues
    try:
        ci_df = res.conf_int()  # DataFrame-like with index param names
    except Exception:
        ci_df = None
    cov = res.cov_params()  # covariance matrix corresponding to cov_type used (HC3 here)
    df_resid = getattr(res, 'df_resid', None)

    def get_param(name):
        if name not in params.index:
            return None
        coef = float(params.loc[name])
        se = float(np.sqrt(cov.loc[name, name]))
        pval = float(pvals.loc[name]) if name in pvals.index else None
        if ci_df is not None and name in ci_df.index:
            ci_low, ci_high = float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])
        else:
            # fallback: compute 95% CI with t critical value (if df_resid available) else normal
            if df_resid is not None and df_resid > 0:
                crit = stats.t.ppf(0.975, df_resid)
            else:
                crit = stats.norm.ppf(0.975)
            ci_low = coef - crit * se
            ci_high = coef + crit * se
        return {"coef": coef, "se": se, "pval": pval, "95%_ci": (ci_low, ci_high)}

    # Key parameter names expected from the formula
    name_mas = 'MasFem_z'
    name_isfemale = 'IsFemaleName'
    name_inter = 'MasFem_z:MaxWind_z'  # patsy/statsmodels interaction naming

    mas = get_param(name_mas)
    isf = get_param(name_isfemale)
    inter = get_param(name_inter)

    # Compute marginal effect of MasFem_z at selected MaxWind_z values (0, +/-1)
    marginal_effects = {}
    if mas is not None:
        beta_m = mas['coef']
        var_m = cov.loc[name_mas, name_mas]
        for val in [ -1.0, 0.0, 1.0 ]:
            if (name_inter in params.index) and (name_inter in cov.index):
                beta_int = params.loc[name_inter]
                var_int = cov.loc[name_inter, name_inter]
                cov_m_int = cov.loc[name_mas, name_inter]
                # marginal effect = beta_m + beta_int * val
                me = float(beta_m + beta_int * val)
                # variance of linear combination
                var_me = float(var_m + (val**2) * var_int + 2 * val * cov_m_int)
                se_me = float(np.sqrt(var_me)) if var_me >= 0 else float(np.nan)
                # t-stat and p-value (two-sided) using t distribution if df_resid present else normal
                if (df_resid is not None) and (df_resid > 0):
                    tstat = me / se_me if se_me != 0 else float('nan')
                    pval = float(2 * stats.t.sf(abs(tstat), df_resid))
                    crit = stats.t.ppf(0.975, df_resid)
                else:
                    zstat = me / se_me if se_me != 0 else float('nan')
                    pval = float(2 * stats.norm.sf(abs(zstat)))
                    crit = stats.norm.ppf(0.975)
                ci_low = me - crit * se_me
                ci_high = me + crit * se_me
            else:
                # no interaction present: marginal effect is simply beta_m
                me = float(beta_m)
                se_me = float(np.sqrt(var_m))
                if (df_resid is not None) and (df_resid > 0):
                    tstat = me / se_me if se_me != 0 else float('nan')
                    pval = float(2 * stats.t.sf(abs(tstat), df_resid))
                    crit = stats.t.ppf(0.975, df_resid)
                else:
                    zstat = me / se_me if se_me != 0 else float('nan')
                    pval = float(2 * stats.norm.sf(abs(zstat)))
                    crit = stats.norm.ppf(0.975)
                ci_low = me - crit * se_me
                ci_high = me + crit * se_me

            marginal_effects[f"MaxWind_z={val}"] = {
                "marginal_effect": me,
                "se": se_me,
                "pval": pval,
                "95%_ci": (ci_low, ci_high)
            }
    else:
        marginal_effects = None

    # Construct brief interpretation of MasFem_z result (based on p-value)
    interpretation = "Could not find MasFem_z in model output."
    if mas is not None:
        p = mas['pval']
        coef = mas['coef']
        if p is None:
            interpretation = ("MasFem_z estimated coef = {:.4f} (SE = {:.4f}); p-value unavailable. "
                              "Cannot classify statistical support for the hypothesis.").format(coef, mas['se'])
        else:
            if p < 0.05:
                direction = "higher" if coef > 0 else "lower"
                interpretation = ("Statistically significant at p < 0.05: MasFem_z coef = {:.4f} (SE = {:.4f}, "
                                  "95% CI [{:.4f}, {:.4f}], p = {:.3g}). This implies that more feminine "
                                  "names are associated with {} log-fatalities, supporting the hypothesis that "
                                  "feminine names lead to greater human consequences (consistent with fewer precautions)."
                                 ).format(coef, mas['se'], mas['95%_ci'][0], mas['95%_ci'][1], direction)
            else:
                interpretation = ("No strong evidence: MasFem_z coef = {:.4f} (SE = {:.4f}, 95% CI [{:.4f}, {:.4f}], "
                                  "p = {:.3g}). This does not provide statistically significant support that "
                                  "more feminine names are associated with higher fatalities."
                                 ).format(coef, mas['se'], mas['95%_ci'][0], mas['95%_ci'][1], p)

    # Assemble output object
    output_object = {
        "MasFem_z": mas,
        "IsFemaleName": isf,
        "MasFem_z_by_MaxWind_z_interaction": inter,
        "Marginal_effects_of_MasFem_z_at_MaxWind_z": marginal_effects,
        "df_resid": df_resid
    }

    return {
        "object": output_object,
        "description": interpretation
    }