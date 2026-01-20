def extract_final_answer(model_output):
    """
    Extract key statistics about the effect of age (Age_c) and its interaction with CultureID
    from a fitted statsmodels GLMResultsWrapper (logistic regression).

    Returns a dictionary with:
      - "object": dict of extracted numeric results (age coefficient, per-culture interactions,
                  joint test of interactions)
      - "description": plain-language interpretation of what those statistics mean for the
                       developmental trajectory of reliance on majority preference across cultures.
    """
    import numpy as np

    res = model_output

    # Basic parameter tables
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    conf = res.conf_int()

    # 1) Extract main effect of Age_c
    if 'Age_c' not in params.index:
        raise KeyError("The fitted model does not contain a parameter named 'Age_c'.")

    age_stat = {
        'coef': float(params['Age_c']),
        'se': float(bse['Age_c']),
        'p_value': float(pvals['Age_c']),
        'conf_int': (float(conf.loc['Age_c', 0]), float(conf.loc['Age_c', 1]))
    }

    # 2) Extract Age_c x CultureID interaction coefficients (if any)
    inter_names = [n for n in params.index if ('Age_c' in n) and ('C(CultureID)' in n)]
    interactions = {}
    for name in inter_names:
        # Example name: 'Age_c:C(CultureID)[T.3]' -> extract level '3' for clarity
        level = name
        # Try to parse the culture level inside brackets if present
        if '[' in name and ']' in name:
            inside = name.split('[')[1].split(']')[0]  # e.g., 'T.3'
            # Remove leading 'T.' if present
            if inside.startswith('T.'):
                level = inside.split('T.')[1]
            else:
                level = inside
        interactions[level] = {
            'param_name': name,
            'coef': float(params[name]),
            'se': float(bse[name]),
            'p_value': float(pvals[name]),
            'conf_int': (float(conf.loc[name, 0]), float(conf.loc[name, 1]))
        }

    # 3) Joint test: are all Age_c x CultureID interactions simultaneously zero?
    joint_test = None
    if len(inter_names) > 0:
        # Build R matrix to test each interaction coefficient = 0 jointly
        p = len(params)
        idxs = [list(params.index).index(n) for n in inter_names]
        R = np.zeros((len(idxs), p))
        for i, idx in enumerate(idxs):
            R[i, idx] = 1.0
        w = res.wald_test(R)
        # w.statistic can be array-like; coerce to float if scalar
        stat = w.statistic
        try:
            stat_val = float(stat)
        except Exception:
            stat_val = np.asarray(stat).item()
        joint_test = {
            'chi2_stat': stat_val,
            'df': int(w.df_denom) if hasattr(w, 'df_denom') and w.df_denom is not None else int(len(idxs)),
            'p_value': float(w.pvalue)
        }
    else:
        joint_test = {
            'chi2_stat': None,
            'df': 0,
            'p_value': None,
            'note': 'No interaction terms Age_c:C(CultureID) were found in the model (likely only a reference level).'
        }

    # 4) Short interpretation decisions
    alpha = 0.05
    age_effect_significant = age_stat['p_value'] < alpha
    interactions_joint_significant = (joint_test['p_value'] is not None) and (joint_test['p_value'] < alpha)

    if age_effect_significant:
        age_dir = 'increases' if age_stat['coef'] > 0 else 'decreases'
        age_interp = f"The main effect of age is statistically significant (coef = {age_stat['coef']:.3f}, p = {age_stat['p_value']:.3f}). This indicates that, averaged across cultures, older children are more likely to choose the majority option." if age_stat['coef'] > 0 else f"...older children are less likely to choose the majority option."
    else:
        age_interp = f"The main effect of age is not statistically significant (coef = {age_stat['coef']:.3f}, p = {age_stat['p_value']:.3f}), indicating no clear overall age-related change in choosing the majority option across cultures."

    if interactions_joint_significant:
        inter_interp = f"The Age × Culture interaction is significant (joint test p = {joint_test['p_value']:.3f}), meaning the developmental trajectory of majority-choice with age differs across cultures. Inspect per-culture interaction coefficients to see which cultures differ from the reference."
    else:
        if joint_test['p_value'] is None:
            inter_interp = "No Age × Culture interaction terms were present to test, so we cannot assess cross-cultural differences in age trajectories from this model."
        else:
            inter_interp = f"The Age × Culture interaction is not significant (joint test p = {joint_test['p_value']:.3f}), suggesting a similar age-related trajectory across the cultures represented in the model."

    description = "Summary of results regarding how reliance on majority preference develops with age across cultures:\n"
    description += f"- Main age effect: {age_interp}\n"
    description += f"- Interaction (Age × Culture): {inter_interp}\n"
    if interactions:
        description += "- Per-culture Age interaction coefficients (these represent deviations from the reference culture's age slope):\n"
        for lvl, info in interactions.items():
            description += f"    * Culture {lvl}: coef={info['coef']:.3f}, p={info['p_value']:.3f}, CI=[{info['conf_int'][0]:.3f}, {info['conf_int'][1]:.3f}]\n"

    # Final object to return (numerical results and flags)
    result_object = {
        'age_effect': age_stat,
        'interactions': interactions,
        'interactions_joint_test': joint_test,
        'age_effect_significant': age_effect_significant,
        'interactions_joint_significant': interactions_joint_significant
    }

    return {
        "object": result_object,
        "description": description
    }