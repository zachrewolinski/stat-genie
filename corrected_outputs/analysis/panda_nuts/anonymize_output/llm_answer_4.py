def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels RegressionResultsWrapper
    for the predictors of interest: Age, Sex_Male, HelpBinary, and their interactions.

    Returns a dictionary with:
      - "object": a nested dict containing coefficients, robust SEs, p-values,
                  95% CIs for each term, plus marginal effects of Age and Sex
                  when HelpBinary = 0 and HelpBinary = 1 (with tests).
      - "description": a short explanation of what the numbers mean and how to interpret them.

    Notes:
      - Expects a statsmodels RegressionResultsWrapper with clustered covariances
        (so cov_params() and t_test() reflect those robust covariances).
      - If a term is missing from the model, it is omitted from results but noted.
    """
    import numpy as np

    res = model_output

    # Terms of substantive interest
    terms = ['Age', 'Sex_Male', 'HelpBinary', 'Age:HelpBinary', 'Sex_Male:HelpBinary']

    # Prepare container for per-term results
    coeffs = {}
    params_index = list(res.params.index)

    # Get standard outputs (params, bse, pvalues, conf_int)
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        ci_df = res.conf_int(alpha=0.05)
    except Exception:
        # fallback: compute using params +/- 1.96*bse (approximate)
        lower = params - 1.96 * bse
        upper = params + 1.96 * bse
        ci_df = None

    for t in terms:
        if t in params_index:
            ci = None
            if ci_df is not None:
                try:
                    row = ci_df.loc[t]
                    ci = [float(row.iloc[0]), float(row.iloc[1])]
                except Exception:
                    ci = [float(params[t] - 1.96 * bse[t]), float(params[t] + 1.96 * bse[t])]
            else:
                ci = [float(params[t] - 1.96 * bse[t]), float(params[t] + 1.96 * bse[t])]

            coeffs[t] = {
                'coef': float(params[t]),
                'robust_se': float(bse[t]),
                'p_value': float(pvals[t]),
                '95%_conf_int': ci,
                'interpretation': (
                    "See description for interpretation. (Main effect or interaction term; "
                    "main effects are for baseline levels of other variables.)"
                )
            }
        else:
            coeffs[t] = None  # missing term

    # Compute marginal effects using linear combinations and t_test where possible
    marginal = {}

    def safe_ttest(expr_label):
        # Use res.t_test with a simple expression like 'Age + Age:HelpBinary'
        try:
            tt = res.t_test(expr_label)
            eff = float(np.asarray(tt.effect).flatten()[0])
            sd = float(np.asarray(tt.sd).flatten()[0])
            tval = float(np.asarray(tt.tvalue).flatten()[0])
            pval = float(np.asarray(tt.pvalue).flatten()[0])
            ci_arr = tt.conf_int(alpha=0.05)
            ci_list = [float(ci_arr[0, 0]), float(ci_arr[0, 1])]
            return {'coef': eff, 'se': sd, 't': tval, 'p_value': pval, '95%_conf_int': ci_list}
        except Exception:
            return None

    # Effect of Age when HelpBinary = 0 is just Age coefficient (if present)
    if coeffs.get('Age') is not None:
        marginal['Age_when_NoHelp'] = {
            'coef': coeffs['Age']['coef'],
            'robust_se': coeffs['Age']['robust_se'],
            'p_value': coeffs['Age']['p_value'],
            '95%_conf_int': coeffs['Age']['95%_conf_int'],
            'meaning': "Change in nuts per minute per additional year of age when no help was received."
        }
    else:
        marginal['Age_when_NoHelp'] = None

    # Effect of Age when HelpBinary = 1  -> Age + Age:HelpBinary
    if (('Age' in params_index) and ('Age:HelpBinary' in params_index)):
        tt = safe_ttest('Age + Q("Age:HelpBinary")')  # Q(...) to be robust to colon name
        if tt is None:
            # try without quoting
            tt = safe_ttest('Age + Age:HelpBinary')
        marginal['Age_when_Help'] = tt
        if tt is not None:
            marginal['Age_when_Help']['meaning'] = "Change in nuts per minute per additional year of age when help was received."
    else:
        marginal['Age_when_Help'] = None

    # Effect of Sex_Male when HelpBinary = 0 is Sex_Male coefficient (if present)
    if coeffs.get('Sex_Male') is not None:
        marginal['SexMale_when_NoHelp'] = {
            'coef': coeffs['Sex_Male']['coef'],
            'robust_se': coeffs['Sex_Male']['robust_se'],
            'p_value': coeffs['Sex_Male']['p_value'],
            '95%_conf_int': coeffs['Sex_Male']['95%_conf_int'],
            'meaning': "Difference in nuts per minute between males and females when no help was received (male minus female)."
        }
    else:
        marginal['SexMale_when_NoHelp'] = None

    # Effect of Sex_Male when HelpBinary = 1 -> Sex_Male + Sex_Male:HelpBinary
    if (('Sex_Male' in params_index) and ('Sex_Male:HelpBinary' in params_index)):
        tt = safe_ttest('Sex_Male + Q("Sex_Male:HelpBinary")')
        if tt is None:
            tt = safe_ttest('Sex_Male + Sex_Male:HelpBinary')
        marginal['SexMale_when_Help'] = tt
        if tt is not None:
            marginal['SexMale_when_Help']['meaning'] = "Difference in nuts per minute between males and females when help was received (male minus female)."
    else:
        marginal['SexMale_when_Help'] = None

    # Effect of receiving help for baseline (e.g., female at Age=0) -- this is the main HelpBinary term
    if coeffs.get('HelpBinary') is not None:
        marginal['Help_effect_main'] = {
            'coef': coeffs['HelpBinary']['coef'],
            'robust_se': coeffs['HelpBinary']['robust_se'],
            'p_value': coeffs['HelpBinary']['p_value'],
            '95%_conf_int': coeffs['HelpBinary']['95%_conf_int'],
            'meaning': (
                "Main effect of receiving help on nuts per minute for the reference group "
                "(this typically corresponds to Sex_Male=0 (female) and Age=0 unless other coding)."
            )
        }
    else:
        marginal['Help_effect_main'] = None

    # Package final object
    results_dict = {
        'terms': coeffs,
        'marginal_effects': marginal,
        'notes': (
            "Term names and marginal-effects provided. Use p_value to judge statistical significance "
            "(conventionally p < 0.05). Interactions are handled: marginal effects for Age and Sex are "
            "given separately for sessions with and without help. HammerType coefficients (control) are "
            "not shown here but are present in the model output if needed."
        )
    }

    description = (
        "This output gives the estimated coefficients (with robust SEs, p-values, and 95% CIs) for Age, Sex, "
        "Help, and their interactions from the fitted regression. It also reports the estimated marginal effects:\n"
        "- Age_when_NoHelp: the effect of one additional year of age on nuts/min when no help was received.\n"
        "- Age_when_Help: the effect of one additional year of age on nuts/min when help was received (Age + Age:HelpBinary).\n"
        "- SexMale_when_NoHelp: male vs female difference in nuts/min when no help was received.\n"
        "- SexMale_when_Help: male vs female difference in nuts/min when help was received (Sex_Male + Sex_Male:HelpBinary).\n"
        "- Help_effect_main: main effect of receiving help for the model baseline.\n\n"
        "Interpretation guidance: positive coefficients mean higher nut-cracking efficiency (more nuts/min). "
        "If an interaction marginal effect (e.g., Age_when_Help) has a small p-value, it indicates the slope differs significantly "
        "when help is present vs absent. The numeric values are returned under 'object' and can be inspected to answer "
        "the question of how age, sex, and receiving help influence nut-cracking efficiency."
    )

    return {"object": results_dict, "description": description}