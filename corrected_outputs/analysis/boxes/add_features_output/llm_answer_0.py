def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels results object (with cluster-robust covariance).
    Specifically focused on testing how children's reliance on majority preference develops with age
    across cultures:
      - main linear age effect (Age_c)
      - quadratic age effect (Age2)
      - interaction terms between Age_c and culture (tests whether age slopes differ across cultures)
      - joint Wald test for all Age_c × culture interactions
      - odds ratios and 95% CIs for interpretable effect sizes

    Returns:
      dict with keys:
        - "object": dict of extracted numeric results
        - "description": plain-language interpretation of those results in context
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Safely extract parameter table pieces
    try:
        params = res.params.copy()          # pandas Series
        bse = res.bse.copy()
        pvalues = res.pvalues.copy()
        conf = res.conf_int()               # DataFrame with two columns (lower, upper)
    except Exception as e:
        raise ValueError("Unable to extract params/bse/pvalues/conf_int from model_output: " + str(e))

    names = list(params.index)

    # Find main Age_c term name (exact match preferred)
    main_age_name = None
    if 'Age_c' in names:
        main_age_name = 'Age_c'
    else:
        # fallback: any parameter that equals Age_c without ':' (i.e., not an interaction)
        for n in names:
            if 'Age_c' in n and ':' not in n:
                main_age_name = n
                break

    # Find Age2 term name
    age2_name = None
    if 'Age2' in names:
        age2_name = 'Age2'
    else:
        for n in names:
            if n == 'Age2' or n.endswith('Age2'):
                age2_name = n
                break

    # Identify interaction terms involving Age_c and culture
    interaction_names = []
    for n in names:
        # interaction likely contains ':' and 'Age_c' (or 'C(culture)' and 'Age_c')
        if 'Age_c' in n and ':' in n:
            interaction_names.append(n)
        # Some naming conventions have 'C(culture)[T.x]:Age_c' or 'Age_c:C(culture)[T.x]'
        elif 'Age_c' in n and 'C(culture)' in n:
            if n != main_age_name:
                interaction_names.append(n)

    # Build results dictionary
    out = {'main_age': None, 'age2': None, 'interactions': {}, 'interaction_joint_test': None}

    # Helper to compute odds ratio and CI
    def or_and_ci(param_name):
        coef = params[param_name]
        se = bse[param_name] if param_name in bse.index else np.nan
        p = pvalues[param_name] if param_name in pvalues.index else np.nan
        ci = conf.loc[param_name].values if param_name in conf.index else (np.nan, np.nan)
        or_ = float(np.exp(coef))
        ci_or = (float(np.exp(ci[0])), float(np.exp(ci[1])))
        return {'coef': float(coef), 'se': float(se), 'p': float(p),
                'ci_coef': (float(ci[0]), float(ci[1])),
                'odds_ratio': or_, 'odds_ratio_CI': ci_or}

    # Fill main age info
    if main_age_name is not None and main_age_name in params.index:
        out['main_age'] = or_and_ci(main_age_name)
    else:
        out['main_age'] = None

    # Fill age2 info
    if age2_name is not None and age2_name in params.index:
        out['age2'] = or_and_ci(age2_name)
    else:
        out['age2'] = None

    # Fill interactions info (per-interaction estimates)
    for iname in interaction_names:
        out['interactions'][iname] = or_and_ci(iname)

    # Joint Wald test for all Age_c x culture interaction coefficients (are all zero?)
    if len(interaction_names) > 0:
        # construct restriction string like "param1 = 0, param2 = 0, ..."
        restr = ', '.join([f"{n} = 0" for n in interaction_names])
        try:
            wres = res.wald_test(restr)
            # wres may have attributes .statistic and .pvalue or .pval; handle both
            stat = float(getattr(wres, 'statistic', np.nan))
            pval = float(getattr(wres, 'pvalue', np.nan) if hasattr(wres, 'pvalue') else getattr(wres, 'pval', np.nan))
            df_denom = getattr(wres, 'df_denom', None)
            df_num = getattr(wres, 'df_num', None)
            out['interaction_joint_test'] = {'statistic': stat, 'pvalue': pval, 'df_num': df_num, 'df_denom': df_denom,
                                             'tested_params': interaction_names}
        except Exception as e:
            out['interaction_joint_test'] = {'error': str(e), 'tested_params': interaction_names}
    else:
        out['interaction_joint_test'] = None  # no interactions present

    # Compose a concise description / interpretation
    desc_lines = []
    if out['main_age'] is not None:
        ma = out['main_age']
        desc_lines.append(
            f"Baseline (reference culture) linear age effect (Age_c): coef={ma['coef']:.3f}, p={ma['p']:.3g}, "
            f"OR={ma['odds_ratio']:.3f} (95% CI {ma['odds_ratio_CI'][0]:.3f}–{ma['odds_ratio_CI'][1]:.3f})."
        )
    else:
        desc_lines.append("No identifiable main Age_c term in the model output.")

    if out['age2'] is not None:
        a2 = out['age2']
        desc_lines.append(
            f"Quadratic age term (Age2): coef={a2['coef']:.3f}, p={a2['p']:.3g} (indicates nonlinearity if significant)."
        )

    if len(out['interactions']) > 0:
        desc_lines.append(f"Found {len(out['interactions'])} Age_c × culture interaction terms. Per-term estimates (coef, p):")
        for name, v in out['interactions'].items():
            desc_lines.append(f"  {name}: coef={v['coef']:.3f}, p={v['p']:.3g}, OR={v['odds_ratio']:.3f}")
        jt = out['interaction_joint_test']
        if jt is None:
            desc_lines.append("No joint test performed for interactions.")
        elif 'error' in jt:
            desc_lines.append("Attempted joint Wald test for interactions, but it failed: " + jt['error'])
        else:
            desc_lines.append(f"Joint Wald test for all Age_c×culture interactions: chi2={jt['statistic']:.3f}, p={jt['pvalue']:.3g}.")
            if jt['pvalue'] < 0.05:
                desc_lines.append("-> Conclusion: Age slopes differ across cultures (reject null of equal slopes).")
            else:
                desc_lines.append("-> Conclusion: No strong evidence that age slopes differ across cultures (fail to reject).")
    else:
        desc_lines.append("No Age_c × culture interaction terms found: model does not test slope differences across cultures.")

    description = " ".join(desc_lines)

    return {"object": out, "description": description}