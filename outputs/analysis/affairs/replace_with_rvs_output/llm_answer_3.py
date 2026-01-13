def extract_final_answer(model_output):
    """
    Extracts the estimated effect of having children on number of extramarital affairs
    from a fitted statsmodels ZeroInflatedPoissonResultsWrapper.

    Returns a dictionary with:
      - "object": a dict with extracted numeric results (log-coefs, IRRs, p-values, 95% CIs)
                  for females (baseline gender_male=0) and males (gender_male=1),
                  computed from the count (Poisson) part of the ZIP model. Also returns
                  the coefficient and p-value from the zero-inflation part for children (if present).
      - "description": a short interpretation of what these numbers mean for the question:
                       "Does having children decrease engagement in extramarital affairs?"
    """
    import numpy as np
    import pandas as pd
    from math import exp, sqrt
    from scipy.stats import norm

    res = model_output  # statsmodels ZeroInflatedPoissonResultsWrapper

    # Obtain parameter names and parameter series in a robust way
    params = res.params
    if hasattr(params, "index"):
        params_ser = params.copy()
        param_names = list(params_ser.index)
    else:
        # fallback: build names from model exog and inflation exog
        exog_names = list(getattr(res.model, "exog_names", []))
        infl_names = ["inflate_" + n for n in exog_names]
        param_names = exog_names + infl_names
        params_ser = pd.Series(np.asarray(params), index=param_names)

    # Separate count (Poisson) part params from inflation (zero) part params.
    # Statsmodels typically prefixes inflation params with "inflate_".
    count_names = [n for n in param_names if not str(n).startswith("inflate")]
    infl_names = [n for n in param_names if str(n).startswith("inflate")]

    # Convert params_ser to DataFrame-like access
    params_ser = pd.Series(params_ser)

    # p-values and confidence intervals (may also be Series/DataFrame)
    pvals = getattr(res, "pvalues", None)
    if hasattr(pvals, "index"):
        pvals_ser = pd.Series(pvals)
    else:
        # fallback align with param_names
        pvals_ser = pd.Series(np.asarray(pvals), index=param_names)

    try:
        conf_int_df = res.conf_int()
        # conf_int_df indexed by param names
    except Exception:
        conf_int_df = None

    # covariance matrix for computing combined effects
    try:
        cov = res.cov_params()
        cov = pd.DataFrame(cov, index=param_names, columns=param_names)
    except Exception:
        cov = None

    # Helper to safely get a parameter (or None)
    def get_param(name):
        return params_ser.get(name, None)

    def get_pval(name):
        return pvals_ser.get(name, None)

    def get_confint(name):
        if conf_int_df is None:
            return None
        # conf_int_df may be DataFrame with two columns [0,1]
        if name in conf_int_df.index:
            row = conf_int_df.loc[name].values
            return (float(row[0]), float(row[1]))
        return None

    # Ensure the key variables exist in the count part
    if "children_yes" not in count_names:
        raise KeyError("children_yes not found among count-model parameter names: " + ", ".join(count_names))

    # Extract coefficients for children and (if present) the interaction children_gender, from the count model
    coef_children = get_param("children_yes")
    p_children = get_pval("children_yes")
    ci_children = get_confint("children_yes")

    coef_inter = get_param("children_gender")
    p_inter = get_pval("children_gender")
    ci_inter = get_confint("children_gender")

    # If interaction missing, treat as zero (no differential effect by gender)
    if coef_inter is None:
        coef_inter = 0.0
        # variance for interaction is zero if not present -> but cov may not have entry; handle later

    # Compute female (baseline) effect: children coefficient (gender_male=0)
    female_logcoef = float(coef_children)
    # Compute male effect as sum of children coefficient + interaction (gender_male=1)
    male_logcoef = float(coef_children + coef_inter)

    # Standard errors: if cov matrix available, use it to compute SE for sums
    def se_of_param(name):
        if cov is None or name not in cov.index:
            # fallback to bse if available
            bse = getattr(res, "bse", None)
            if hasattr(bse, "index"):
                return float(bse.get(name, np.nan))
            return np.nan
        return float(np.sqrt(cov.loc[name, name]))

    se_children = se_of_param("children_yes")
    se_inter = se_of_param("children_gender") if "children_gender" in cov.index else se_of_param("children_gender")

    # SE for female is se_children
    female_se = se_children

    # SE for male (sum) uses variance sum formula
    if cov is not None and ("children_yes" in cov.index) and ("children_gender" in cov.index):
        var_sum = cov.loc["children_yes", "children_yes"] + cov.loc["children_gender", "children_gender"] + 2.0 * cov.loc["children_yes", "children_gender"]
        male_se = float(np.sqrt(max(var_sum, 0.0)))
    else:
        # fallback: conservative approx by summing variances if available, else NaN
        if (not np.isnan(se_children)) and (not np.isnan(se_inter)):
            male_se = sqrt(se_children**2 + se_inter**2)
        else:
            male_se = np.nan

    # Compute Wald p-values for the combined male effect if cov present, else approximate using normal
    def p_from_coef_and_se(coef, se):
        if se is None or np.isnan(se) or se == 0:
            return None
        z = coef / se
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        return float(p)

    female_p = p_from_coef_and_se(female_logcoef, female_se) if p_children is None else float(p_children)
    male_p = p_from_coef_and_se(male_logcoef, male_se)

    # Compute 95% Wald CIs on log scale and exponentiate to get IRR CIs
    def logci_to_irr(ci_log):
        if ci_log is None:
            return None
        lower_log, upper_log = float(ci_log[0]), float(ci_log[1])
        return (exp(lower_log), exp(upper_log))

    female_ci_log = None
    male_ci_log = None
    if female_se is not None and not np.isnan(female_se):
        female_ci_log = (female_logcoef - 1.96 * female_se, female_logcoef + 1.96 * female_se)
    if male_se is not None and not np.isnan(male_se):
        male_ci_log = (male_logcoef - 1.96 * male_se, male_logcoef + 1.96 * male_se)

    female_irr = exp(female_logcoef)
    male_irr = exp(male_logcoef)
    female_ci_irr = logci_to_irr(female_ci_log) if female_ci_log is not None else get_confint("children_yes") and logci_to_irr(get_confint("children_yes"))
    male_ci_irr = logci_to_irr(male_ci_log)

    # Also extract inflation-part coefficient for children_yes if present (helps interpret zeros process)
    inflate_children_name = "inflate_children_yes" if "inflate_children_yes" in params_ser.index else None
    inflate_children_coef = None
    inflate_children_p = None
    inflate_children_ci = None
    if inflate_children_name is not None:
        inflate_children_coef = float(params_ser[inflate_children_name])
        inflate_children_p = float(pvals_ser.get(inflate_children_name, np.nan))
        if conf_int_df is not None and inflate_children_name in conf_int_df.index:
            rc = conf_int_df.loc[inflate_children_name].values
            inflate_children_ci = (float(rc[0]), float(rc[1]))

    # Build the "object" to be returned: numerical summaries
    result_object = {
        "count_part": {
            "children_coef_log_female": female_logcoef,
            "children_IRR_female": female_irr,
            "children_pvalue_female": female_p,
            "children_95CI_log_female": female_ci_log,
            "children_95CI_IRR_female": female_ci_irr,
            "children_coef_log_male": male_logcoef,
            "children_IRR_male": male_irr,
            "children_pvalue_male": male_p,
            "children_95CI_log_male": male_ci_log,
            "children_95CI_IRR_male": male_ci_irr,
            "notes": "Coefficients are from the count part (Poisson) of the ZIP model. They are log(IRR). IRR = exp(coef). Female corresponds to baseline (gender_male=0). Male = female + interaction."
        },
        "inflation_part_children_if_present": {
            "inflate_children_coef_logit": inflate_children_coef,
            "inflate_children_pvalue": inflate_children_p,
            "inflate_children_95CI_logit": inflate_children_ci
        }
    }

    # Short text description interpreting the direction and statistical evidence
    # (we include placeholders for numeric values computed above)
    def describe_effect(logcoef, irr, pval, gender_label):
        if pval is None:
            signif = "p-value unavailable"
        elif pval < 0.001:
            signif = "p < 0.001"
        else:
            signif = f"p = {pval:.3f}"
        pct_change = (irr - 1.0) * 100.0
        direction = "decrease" if irr < 1.0 else ("increase" if irr > 1.0 else "no change")
        return f"For {gender_label}, having children is associated with a {pct_change:.1f}% {direction} in the expected number of affairs (IRR = {irr:.3f}; {signif})."

    desc_lines = []
    desc_lines.append(describe_effect(female_logcoef, female_irr, female_p, "females (baseline)"))
    desc_lines.append(describe_effect(male_logcoef, male_irr, male_p, "males"))
    if inflate_children_coef is not None:
        desc_lines.append(
            "In the zero-inflation (logit) part, the children coefficient (if present) describes association with being in the 'always-zero' group; inspect inflate_children_* values for whether children make being zero-more/less likely."
        )
    full_description = " ".join(desc_lines)

    return {"object": result_object, "description": full_description}