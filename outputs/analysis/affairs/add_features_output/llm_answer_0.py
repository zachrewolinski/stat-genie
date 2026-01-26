def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of having children on the probability of any extramarital affair
    from the provided model_output dict (expected to contain a fitted statsmodels Logit result under
    'logit_any_affair' and optionally a GLM NegativeBinomial under 'negbin_pos_affairs').

    Returns a dict with keys:
      - "object": dict with extracted statistics (coefficients, SEs, p-values, odds-ratios or IRRs,
                    95% CIs) for:
            * main effect of children for males (gender_female=0),
            * interaction term,
            * combined effect for females (gender_female=1),
            * sample sizes.
      - "description": short plain-language interpretation of what the statistics mean for the research question.
    """
    import numpy as np
    from math import exp
    from scipy import stats

    out = {"object": {}, "description": ""}

    # Helper to find parameter name among possible variants
    def find_param_name(params_index, candidates):
        for c in candidates:
            if c in params_index:
                return c
        return None

    # Pull logistic model
    logit_res = model_output.get('logit_any_affair', None)
    n_total = model_output.get('n_total', None)
    n_pos = model_output.get('n_positive_affairs', None)
    n_children = model_output.get('n_children', None)

    out["object"]["n_total"] = n_total
    out["object"]["n_positive_affairs"] = n_pos
    out["object"]["n_children"] = n_children

    if logit_res is None:
        out["object"]["logistic"] = None
        out["description"] = "Logistic model result not available in model_output."
        return out

    # Extract parameter table and covariance
    params = logit_res.params
    pvalues = logit_res.pvalues
    cov = logit_res.cov_params()

    # Possible parameter names for main and interaction (cover ordering variants)
    main_candidates = ['children_yes']
    inter_candidates = ['children_yes:gender_female', 'gender_female:children_yes']

    main_name = find_param_name(params.index, main_candidates)
    inter_name = find_param_name(params.index, inter_candidates)

    # Prepare container for logistic stats
    log_stats = {}

    # Function to compute OR, CI, z, p
    def summarize_coef(name):
        if name is None or name not in params.index:
            return None
        coef = float(params[name])
        se = float(np.sqrt(cov.loc[name, name]))
        z = coef / se if se > 0 else np.nan
        p = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        or_point = exp(coef)
        or_ci = (exp(ci_lower), exp(ci_upper))
        return {
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "odds_ratio": or_point,
            "or_95ci": list(or_ci),
            "ci_coef": [ci_lower, ci_upper]
        }

    # Main effect (effect of children when gender_female == 0, i.e., males)
    main_summary = summarize_coef(main_name)
    log_stats['children_main_males'] = main_summary

    # Interaction term summary
    inter_summary = summarize_coef(inter_name)
    log_stats['children_x_female_interaction'] = inter_summary

    # Combined effect for females: beta_children + beta_interaction
    if main_summary is None:
        combined_summary = None
    else:
        beta_main = main_summary["coef"]
        # If interaction missing, female effect is same as main
        if inter_summary is None:
            beta_female = beta_main
            # se is same as main
            se_female = main_summary["se"]
            # p-value same
            p_female = main_summary["p_value"]
        else:
            beta_inter = inter_summary["coef"]
            beta_female = beta_main + beta_inter
            # variance = var(main) + var(inter) + 2*cov(main, inter)
            try:
                var_main = cov.loc[main_name, main_name]
                var_inter = cov.loc[inter_name, inter_name]
                covar = cov.loc[main_name, inter_name]
                var_sum = var_main + var_inter + 2.0 * covar
                se_female = float(np.sqrt(var_sum)) if var_sum >= 0 else float(np.nan)
            except Exception:
                se_female = float(np.nan)
            z_female = beta_female / se_female if se_female and not np.isnan(se_female) else np.nan
            p_female = float(2 * stats.norm.sf(abs(z_female))) if not np.isnan(z_female) else np.nan

        # Build female summary
        ci_lower = beta_female - 1.96 * se_female if se_female is not None else np.nan
        ci_upper = beta_female + 1.96 * se_female if se_female is not None else np.nan
        or_point = exp(beta_female)
        or_ci = (exp(ci_lower), exp(ci_upper)) if not np.isnan(ci_lower) and not np.isnan(ci_upper) else (np.nan, np.nan)

        combined_summary = {
            "coef": float(beta_female),
            "se": float(se_female),
            "z": float(beta_female / se_female) if se_female and not np.isnan(se_female) else np.nan,
            "p_value": p_female,
            "odds_ratio": or_point,
            "or_95ci": list(or_ci),
            "ci_coef": [ci_lower, ci_upper]
        }

    log_stats['children_effect_females'] = combined_summary

    out["object"]["logistic"] = log_stats

    # Now handle negative binomial (conditional frequency) if present
    nb_res = model_output.get('negbin_pos_affairs', None)
    if nb_res is None:
        out["object"]["negbin_pos_affairs"] = None
    else:
        # Extract params and cov
        nb_params = nb_res.params
        try:
            nb_cov = nb_res.cov_params()
        except Exception:
            nb_cov = None

        nb_main_name = find_param_name(nb_params.index, main_candidates)
        nb_inter_name = find_param_name(nb_params.index, inter_candidates)

        def summarize_nb_coef(name):
            if name is None or name not in nb_params.index:
                return None
            coef = float(nb_params[name])
            se = float(np.sqrt(nb_cov.loc[name, name])) if nb_cov is not None else float(np.nan)
            z = coef / se if se > 0 else np.nan
            p = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            irr = exp(coef)  # incidence rate ratio
            irr_ci = (exp(ci_lower), exp(ci_upper))
            return {
                "coef": coef,
                "se": se,
                "z": z,
                "p_value": p,
                "incidence_rate_ratio": irr,
                "irr_95ci": list(irr_ci),
                "ci_coef": [ci_lower, ci_upper]
            }

        nb_stats = {}
        nb_stats['children_main_males'] = summarize_nb_coef(nb_main_name)
        nb_stats['children_x_female_interaction'] = summarize_nb_coef(nb_inter_name)

        # Combined for females analogous to logistic
        if nb_stats['children_main_males'] is None:
            nb_combined = None
        else:
            beta_main = nb_stats['children_main_males']['coef']
            if nb_stats['children_x_female_interaction'] is None:
                beta_female = beta_main
                se_female = nb_stats['children_main_males']['se']
                p_female = nb_stats['children_main_males']['p_value']
            else:
                beta_inter = nb_stats['children_x_female_interaction']['coef']
                beta_female = beta_main + beta_inter
                try:
                    var_main = nb_cov.loc[nb_main_name, nb_main_name]
                    var_inter = nb_cov.loc[nb_inter_name, nb_inter_name]
                    covar = nb_cov.loc[nb_main_name, nb_inter_name]
                    var_sum = var_main + var_inter + 2.0 * covar
                    se_female = float(np.sqrt(var_sum)) if var_sum >= 0 else float(np.nan)
                except Exception:
                    se_female = float(np.nan)
                z_female = beta_female / se_female if se_female and not np.isnan(se_female) else np.nan
                p_female = float(2 * stats.norm.sf(abs(z_female))) if not np.isnan(z_female) else np.nan

            ci_lower = beta_female - 1.96 * se_female if se_female is not None else np.nan
            ci_upper = beta_female + 1.96 * se_female if se_female is not None else np.nan
            irr = exp(beta_female)
            irr_ci = (exp(ci_lower), exp(ci_upper)) if not np.isnan(ci_lower) and not np.isnan(ci_upper) else (np.nan, np.nan)
            nb_combined = {
                "coef": float(beta_female),
                "se": float(se_female),
                "z": float(beta_female / se_female) if se_female and not np.isnan(se_female) else np.nan,
                "p_value": p_female,
                "incidence_rate_ratio": irr,
                "irr_95ci": list(irr_ci),
                "ci_coef": [ci_lower, ci_upper]
            }

        nb_stats['children_effect_females'] = nb_combined
        out["object"]["negbin_pos_affairs"] = nb_stats

    # Construct a concise description that helps answer the yes/no question
    # We base the yes/no primarily on the logistic model:
    desc_lines = []
    desc_lines.append("Primary model: logistic regression predicting probability of any extramarital affair.")
    if log_stats['children_main_males'] is None:
        desc_lines.append("The main 'children_yes' coefficient is not present in the logistic model; cannot evaluate effect.")
    else:
        m = log_stats['children_main_males']
        desc_lines.append(
            "For males (gender_female=0): children_yes -> coef = {coef:.3f}, SE = {se:.3f}, p = {p:.3g}, OR = {or_: .3f} (95% CI = [{lo:.3f}, {hi:.3f}])"
            .format(coef=m['coef'], se=m['se'], p=m['p_value'], or_=m['odds_ratio'],
                    lo=m['or_95ci'][0], hi=m['or_95ci'][1])
        )
    if log_stats['children_x_female_interaction'] is not None:
        it = log_stats['children_x_female_interaction']
        desc_lines.append(
            "Interaction children_yes x female: coef = {coef:.3f}, SE = {se:.3f}, p = {p:.3g}."
            .format(coef=it['coef'], se=it['se'], p=it['p_value'])
        )
    if log_stats['children_effect_females'] is not None:
        f = log_stats['children_effect_females']
        desc_lines.append(
            "For females (gender_female=1): combined effect -> coef = {coef:.3f}, SE = {se:.3f}, p = {p:.3g}, OR = {or_: .3f} (95% CI = [{lo:.3f}, {hi:.3f}])."
            .format(coef=f['coef'], se=f['se'], p=f['p_value'], or_=f['odds_ratio'],
                    lo=(f['or_95ci'][0] if f['or_95ci'] is not None else np.nan),
                    hi=(f['or_95ci'][1] if f['or_95ci'] is not None else np.nan))
        )

    desc_lines.append("Interpretation guidance: An OR < 1 means children are associated with lower odds of any affair; "
                      "OR > 1 means higher odds. Use the p-value (commonly p < 0.05) to judge statistical significance.")
    desc_lines.append("Secondary model: negative binomial (frequency among those with any affairs) is provided under 'negbin_pos_affairs' in the 'object' output; "
                      "IRR < 1 indicates fewer affairs among those with children, conditional on having any affairs.")

    out["description"] = " ".join(desc_lines)

    return out