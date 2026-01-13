def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of age (and age-by-culture interactions)
    from the fitted logistic regression model stored in model_output.

    Returns:
      {
        "object": { ... }  # detailed numeric extraction
        "description": "..."  # brief interpretation in context
      }
    """
    import numpy as np
    import pandas as pd

    # Get fitted results object (prefer cluster-robust 'fit' if available)
    fit = model_output.get('fit') or model_output.get('model_raw')
    if fit is None:
        raise ValueError("No fitted model found in model_output (expected keys 'fit' or 'model_raw').")

    params = fit.params
    pvalues = fit.pvalues
    try:
        conf = fit.conf_int()
    except Exception:
        # fallback: if conf_int not available, create NaN placeholders
        conf = pd.DataFrame(index=params.index, columns=[0, 1], data=np.nan)

    # 1) Extract main age coefficient information
    age_param_name = None
    for name in params.index:
        # find the plain 'age_c' main effect parameter name (exact match or token)
        if name == 'age_c' or name.endswith('.age_c') or name.startswith('age_c'):
            # prefer exact 'age_c' name if present
            if name == 'age_c':
                age_param_name = name
                break
            else:
                # keep if no exact match found yet
                if age_param_name is None:
                    age_param_name = name
    if age_param_name is None:
        # try more general search
        matches = [n for n in params.index if 'age_c' in n and 'C(culture)' not in n]
        age_param_name = matches[0] if matches else None

    if age_param_name is None:
        raise ValueError("Could not find a main 'age_c' parameter in model parameters.")

    age_coef = float(params.loc[age_param_name])
    age_p = float(pvalues.loc[age_param_name]) if age_param_name in pvalues.index else np.nan
    age_ci = tuple(conf.loc[age_param_name]) if age_param_name in conf.index else (np.nan, np.nan)

    # 2) Identify interaction parameters (age_c x culture)
    interaction_names = [n for n in params.index if ('age_c' in n) and ('culture' in n)]
    # Normalize to unique set
    interaction_names = list(dict.fromkeys(interaction_names))

    interaction_info = {}
    for name in interaction_names:
        # Attempt to infer culture level label from the parameter name
        # Common forms: 'age_c:C(culture)[T.2]' or 'C(culture)[T.2]:age_c'
        # Extract the substring between 'T.' and ']' if present, else fallback to name
        label = None
        try:
            if 'C(culture)' in name:
                start = name.find('T.')
                end = name.find(']', start)
                if start != -1 and end != -1:
                    label = name[start+2:end]
        except Exception:
            label = None
        if label is None:
            label = name

        interaction_info[label] = {
            'param_name': name,
            'coef': float(params.loc[name]),
            'pvalue': float(pvalues.loc[name]) if name in pvalues.index else np.nan,
            'conf_int': tuple(conf.loc[name]) if name in conf.index else (np.nan, np.nan)
        }

    # 3) Compute per-culture age effect on the logit scale:
    # For reference culture: age_effect = age_coef
    # For other cultures: age_effect = age_coef + interaction_coef
    # We need the list of cultures; get from predicted_prob_grid if available
    pred_df = model_output.get('predicted_prob_grid')
    if pred_df is None:
        # fallback: infer culture levels from parameter names (less desirable)
        culture_levels = []
        for name in params.index:
            if 'C(culture)' in name and '[' in name:
                # extract label as above
                start = name.find('T.')
                end = name.find(']', start)
                if start != -1 and end != -1:
                    label = name[start+2:end]
                    if label not in culture_levels:
                        culture_levels.append(label)
        # We cannot determine reference level precisely; leave empty if unknown
        pred_df = None
    else:
        if 'culture' in pred_df.columns:
            culture_levels = list(pd.Categorical(pred_df['culture']).categories) if hasattr(pred_df['culture'], 'cat') else list(pd.unique(pred_df['culture']))
        else:
            culture_levels = []

    age_effects_logit = {}
    for c in culture_levels:
        # find matching interaction param for this culture if any
        # possible param strings include the culture label; search interaction_info keys
        match_key = None
        for k in interaction_info.keys():
            if str(k) == str(c) or (isinstance(k, str) and str(c) in k):
                match_key = k
                break
        if match_key is not None:
            inter_coef = interaction_info[match_key]['coef']
        else:
            inter_coef = 0.0
        age_effects_logit[c] = age_coef + inter_coef

    # 4) Compute per-culture slope on predicted probability scale using predicted_prob_grid
    prob_slopes = {}
    if pred_df is not None:
        # ensure age_c numeric
        pred_df = pred_df.copy()
        pred_df['age_c'] = pd.to_numeric(pred_df['age_c'])
        for c in culture_levels:
            subset = pred_df[pred_df['culture'] == c]
            if len(subset) >= 2:
                # linear fit pred_prob ~ age_c; slope is change in probability per 1 year
                slope, intercept = np.polyfit(subset['age_c'], subset['pred_prob'], 1)
                prob_slopes[c] = float(slope)
            else:
                prob_slopes[c] = float('nan')
    else:
        prob_slopes = {c: float('nan') for c in culture_levels}

    # 5) Joint test: are all age-by-culture interactions equal to zero?
    joint_p = None
    if interaction_names:
        try:
            # Construct Wald test string like "param1 = 0, param2 = 0, ..."
            restr = ", ".join([f"{name} = 0" for name in interaction_names])
            wtest = fit.wald_test(restr)
            # wtest.pvalue may be nested; attempt to extract robustly
            pval = None
            if hasattr(wtest, 'pvalue') and wtest.pvalue is not None:
                pval = wtest.pvalue
            elif hasattr(wtest, 'pvalue_raw') and wtest.pvalue_raw is not None:
                pval = wtest.pvalue_raw
            # Try to convert to float if possible
            if pval is not None:
                try:
                    joint_p = float(pval)
                except Exception:
                    # if it's array-like, take first element
                    try:
                        joint_p = float(getattr(pval, 'item', lambda: pval)())
                    except Exception:
                        joint_p = None
            else:
                joint_p = None
        except Exception:
            # fallback: if wald_test fails, set None
            joint_p = None

    # 6) Build a concise interpretation
    # Interpret overall age effect
    if np.isfinite(age_p):
        if age_p < 0.05:
            overall_trend = "There is a statistically significant overall effect of age on choosing the majority (age coefficient = {:.3f}, p = {:.3g}).".format(age_coef, age_p)
            if age_coef > 0:
                overall_trend += " On the logit scale, older children are more likely to choose the majority (positive coefficient)."
            else:
                overall_trend += " On the logit scale, older children are less likely to choose the majority (negative coefficient)."
        else:
            overall_trend = "No statistically significant overall age effect was detected (age coefficient = {:.3f}, p = {:.3g}).".format(age_coef, age_p)
    else:
        overall_trend = "Could not determine statistical significance for the overall age effect (p-value not available)."

    # Interpret interactions
    if interaction_names:
        if joint_p is not None:
            if joint_p < 0.05:
                interaction_trend = "The age-by-culture interaction terms collectively are statistically significant (joint test p = {:.3g}), indicating developmental trajectories differ across cultures.".format(joint_p)
            else:
                interaction_trend = "The age-by-culture interaction terms collectively are NOT statistically significant (joint test p = {:.3g}), indicating no strong evidence that developmental trajectories differ across cultures.".format(joint_p)
        else:
            interaction_trend = "Interaction parameters are present but a joint test could not be computed. Individual interaction coefficients are reported below."
    else:
        interaction_trend = "No age-by-culture interaction terms were found in the model."

    # Summarize magnitude range of predicted-probability slopes
    if prob_slopes:
        finite_slopes = [v for v in prob_slopes.values() if np.isfinite(v)]
        if finite_slopes:
            min_slope = float(np.min(finite_slopes))
            max_slope = float(np.max(finite_slopes))
            magnitude_summary = "Predicted-probability change per 1-year of age ranges from {:.3f} to {:.3f} across cultures (slope in probability points per year).".format(min_slope, max_slope)
        else:
            magnitude_summary = "Could not compute predicted-probability slopes across cultures (insufficient predicted data)."
    else:
        magnitude_summary = "No predicted probability grid available to compute slopes."

    # Safely check whether confidence interval values are available (not NaN)
    has_ci = (
        isinstance(age_ci, (list, tuple))
        and len(age_ci) == 2
        and not (pd.isna(age_ci[0]) or pd.isna(age_ci[1]))
    )

    description = "Overall age effect: {} {} Interaction summary: {} {}".format(
        overall_trend,
        ("95% CI (age coef): [{:.3f}, {:.3f}].".format(age_ci[0], age_ci[1]) if has_ci else ""),
        interaction_trend,
        magnitude_summary
    )

    # Construct the return object with numeric details
    result_object = {
        'age_param_name': age_param_name,
        'age_coef_logit': age_coef,
        'age_pvalue': age_p,
        'age_conf_int_logit': age_ci,
        'interaction_params': interaction_info,  # dict keyed by extracted label -> details
        'age_effects_logit_by_culture': age_effects_logit,  # per-culture logit-scale age slopes
        'pred_prob_slope_by_culture': prob_slopes,  # per-culture probability-scale slopes (per 1 yr)
        'joint_interaction_pvalue': joint_p
    }

    return {
        "object": result_object,
        "description": description
    }