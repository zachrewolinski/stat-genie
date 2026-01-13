def extract_final_answer(model_output):
    """
    Extracts age-related effects on choosing the majority option from the fitted model output.
    Returns a dictionary with:
      - "object": dict with numeric estimates (overall age coef, age-by-culture combined effects,
                  p-values, CIs, and a joint test of Age x Culture interactions).
      - "description": brief interpretation of those statistics in plain language.
    """
    import re
    import numpy as np
    import pandas as pd
    from math import sqrt
    from scipy import stats

    res = model_output.get('model_result')
    if res is None:
        raise ValueError("model_output must contain key 'model_result' with a fitted statsmodels result.")

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    cov = res.cov_params()

    # 1) Overall Age_c coefficient (from the model)
    overall_age_coef = params.get('Age_c', np.nan)
    overall_age_se = bse.get('Age_c', np.nan)
    overall_age_p = pvalues.get('Age_c', np.nan)
    overall_age_ci_low, overall_age_ci_high = np.nan, np.nan
    try:
        ci = res.conf_int().loc['Age_c']
        overall_age_ci_low, overall_age_ci_high = float(ci[0]), float(ci[1])
    except Exception:
        pass

    # 2) Determine culture levels (if original data available) or infer from parameter names
    levels = None
    base_level = None
    try:
        df = res.model.data.frame  # original dataframe used to fit the model
        if 'Culture' in df.columns:
            # use categorical ordering used by pandas/statsmodels
            cats = pd.Categorical(df['Culture'])
            # categories attribute gives sorted *observed* categories; first is the reference used by statsmodels
            levels = [str(x) for x in cats.categories]
            if len(levels) > 0:
                base_level = levels[0]
    except Exception:
        pass

    # If we couldn't get levels from data, infer from interaction parameter names:
    inter_pattern = re.compile(r'Age_c:C\(Culture\)\[T\.(.+)\]')
    inter_terms = []
    for name in params.index:
        m = inter_pattern.match(name)
        if m:
            inter_terms.append((m.group(1), name))
    if levels is None:
        # Use the interaction-derived levels (if any). If none, we'll still report overall Age_c.
        if inter_terms:
            levels = [lvl for (lvl, _) in inter_terms]
            # reference/base not directly known; name it "reference"
            base_level = 'reference'
        else:
            levels = [base_level if base_level is not None else 'reference']
            base_level = levels[0]

    # Ensure base_level is set
    if base_level is None and len(levels) > 0:
        base_level = levels[0]

    # 3) For each culture level, compute the combined Age effect:
    age_by_culture = {}
    for lvl in levels:
        # For base/reference level, there is no interaction term; combined = Age_c
        if str(lvl) == str(base_level) and base_level != 'reference':
            # explicit base present in levels
            inter_name = None
        else:
            # find the exact interaction parameter name for this level (if exists)
            inter_name = None
            for name in params.index:
                m = inter_pattern.match(name)
                if m and m.group(1) == str(lvl):
                    inter_name = name
                    break

        if inter_name is None:
            # Combined effect is simply Age_c
            coef = overall_age_coef
            # variance is var(Age_c)
            try:
                var = float(cov.loc['Age_c', 'Age_c'])
            except Exception:
                var = np.nan
        else:
            # Combined effect = Age_c + Age_c:C(Culture)[T.<lvl>]
            coef = float(params['Age_c'] + params[inter_name])
            # variance = Var(Age_c) + Var(inter) + 2*Cov(Age_c, inter)
            try:
                var_age = float(cov.loc['Age_c', 'Age_c'])
                var_inter = float(cov.loc[inter_name, inter_name])
                covar = float(cov.loc['Age_c', inter_name])
                var = var_age + var_inter + 2.0 * covar
            except Exception:
                var = np.nan

        se = float(sqrt(var)) if not np.isnan(var) else np.nan
        z = coef / se if (se is not None and se and not np.isnan(se)) else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low = coef - stats.norm.ppf(0.975) * se if not np.isnan(se) else np.nan
        ci_high = coef + stats.norm.ppf(0.975) * se if not np.isnan(se) else np.nan

        age_by_culture[str(lvl)] = {
            'combined_age_coef': coef,
            'se': se,
            'z': z,
            'p_value': p,
            '95ci': (ci_low, ci_high),
            'notes': f"combined Age effect for culture '{lvl}' (Age_c + interaction if present)"
        }

    # 4) Individual interaction term p-values (to show which interactions, if any, were significant)
    interaction_pvalues = {}
    inter_param_names = []
    for name in params.index:
        if name.startswith('Age_c:C(Culture)'):
            interaction_pvalues[name] = {
                'coef': float(params[name]),
                'se': float(bse[name]) if name in bse.index else np.nan,
                'p_value': float(pvalues[name]) if name in pvalues.index else np.nan,
                '95ci': tuple(res.conf_int().loc[name]) if name in res.conf_int().index else (np.nan, np.nan)
            }
            inter_param_names.append(name)

    # 5) Joint (Wald) test: Are all Age x Culture interactions simultaneously zero?
    joint_test = None
    if len(inter_param_names) > 0:
        # build constraint string like "Age_c:C(Culture)[T.2] = 0, Age_c:C(Culture)[T.3] = 0"
        constr = ", ".join([f"{n} = 0" for n in inter_param_names])
        try:
            wt = res.wald_test(constr)
            joint_test = {
                'statistic': float(wt.statistic) if hasattr(wt, 'statistic') else None,
                'p_value': float(wt.pvalue) if hasattr(wt, 'pvalue') else None,
                'df_denom': getattr(wt, 'df_denom', None),
                'df_num': getattr(wt, 'df_num', None),
                'constraint': constr
            }
        except Exception:
            joint_test = {'error': 'wald_test failed for joint test of interactions', 'constraint': constr}

    # 6) Create a compact human-readable description/interpretation.
    # Determine whether any culture-specific age slopes are significant at alpha=0.05
    sig_cultures = [lvl for lvl, info in age_by_culture.items() if (info['p_value'] is not None and not np.isnan(info['p_value']) and info['p_value'] < 0.05)]
    any_interaction_significant = any((v['p_value'] < 0.05) for v in interaction_pvalues.values()) if interaction_pvalues else False

    # Build the description
    desc_lines = []
    desc_lines.append("Overall age effect (Age_c): coef = {:.4f}, se = {:.4f}, p = {:.3f}, 95% CI = [{:.4f}, {:.4f}]"
                      .format(overall_age_coef if not np.isnan(overall_age_coef) else np.nan,
                              overall_age_se if not np.isnan(overall_age_se) else np.nan,
                              overall_age_p if not np.isnan(overall_age_p) else np.nan,
                              overall_age_ci_low if not np.isnan(overall_age_ci_low) else np.nan,
                              overall_age_ci_high if not np.isnan(overall_age_ci_high) else np.nan))
    if joint_test and 'p_value' in joint_test and joint_test.get('p_value') is not None:
        desc_lines.append("Joint test for Age x Culture interactions: chi2 = {:.3f}, p = {:.3f}".format(
            joint_test['statistic'] if joint_test.get('statistic') is not None else np.nan,
            joint_test['p_value'] if joint_test.get('p_value') is not None else np.nan
        ))
    elif joint_test and 'error' in joint_test:
        desc_lines.append("Joint test for interactions could not be computed: {}".format(joint_test.get('error')))

    # Summarize per-culture combined age effects
    desc_lines.append("Estimated effect of age on choosing majority, by culture (coef, se, p):")
    for lvl, info in age_by_culture.items():
        desc_lines.append(" - {}: coef = {:.4f}, se = {:.4f}, p = {:.3f}, 95% CI = [{:.4f}, {:.4f}]".format(
            lvl,
            info['combined_age_coef'] if not np.isnan(info['combined_age_coef']) else np.nan,
            info['se'] if not np.isnan(info['se']) else np.nan,
            info['p_value'] if not np.isnan(info['p_value']) else np.nan,
            info['95ci'][0] if not np.isnan(info['95ci'][0]) else np.nan,
            info['95ci'][1] if not np.isnan(info['95ci'][1]) else np.nan
        ))

    # Final interpretation sentence
    if (overall_age_p is not None and not np.isnan(overall_age_p) and overall_age_p < 0.05) or (len(sig_cultures) > 0) or any_interaction_significant:
        conclusion = ("There is evidence that age relates to reliance on the majority in at least some comparisons. "
                      "See per-culture estimates above for which cultures show significant age slopes.")
    else:
        conclusion = ("No strong evidence that reliance on the majority systematically changes with age overall or that "
                      "age-related slopes differ across cultures (no significant Age_c effect, and no significant Age x Culture interactions).")

    desc_lines.append(conclusion)
    description = " ".join(desc_lines)

    output = {
        'object': {
            'overall_age': {
                'coef': float(overall_age_coef) if not np.isnan(overall_age_coef) else None,
                'se': float(overall_age_se) if not np.isnan(overall_age_se) else None,
                'p_value': float(overall_age_p) if not np.isnan(overall_age_p) else None,
                '95ci': (overall_age_ci_low, overall_age_ci_high)
            },
            'age_by_culture': age_by_culture,
            'interaction_terms': interaction_pvalues,
            'joint_interaction_test': joint_test,
            'significant_cultures_by_age': sig_cultures
        },
        'description': description
    }

    return output