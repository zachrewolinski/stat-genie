def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of 'Children' on 'AnyAffair' from the model output.
    Returns a dictionary with keys:
      - "object": a dict of extracted numeric results (marginal effects, coefficients, p-values, descriptives)
      - "description": a concise interpretation of those results in context.

    Expected model_output keys used: 'logit', 'logit_margeff' (optional), 'descriptives'
    """
    import numpy as np
    import pandas as pd

    out = {
        "object": None,
        "description": None
    }

    logit = model_output.get('logit')
    descriptives = model_output.get('descriptives')

    if logit is None:
        out["object"] = {"error": "Logistic model ('logit') not found in model_output."}
        out["description"] = "No logistic model available to answer the question."
        return out

    # 1) Try to obtain average marginal effects (preferred)
    ame = None
    ame_df = None
    try:
        marg = logit.get_margeff(at='overall', method='dydx')
        # summary_frame provides a DataFrame with columns: dy/dx, std err, z, P>|z|, [0.025,0.975]
        ame_df = marg.summary_frame()
        ame = ame_df.copy()
    except Exception:
        ame = None
        ame_df = None

    # 2) Always extract coefficient, se, pvalue, conf_int for Children and interaction from logit params
    params = logit.params
    bse = logit.bse
    pvals = logit.pvalues
    try:
        conf = logit.conf_int(alpha=0.05)
        conf.columns = ['ci_lower', 'ci_upper']
    except Exception:
        conf = None

    def safe_get(series, name):
        if name in series.index:
            return float(series[name])
        else:
            return None

    coeff_children = safe_get(params, 'Children')
    se_children = safe_get(bse, 'Children')
    p_children = safe_get(pvals, 'Children')
    ci_children = None
    if conf is not None and 'Children' in conf.index:
        ci_children = (float(conf.loc['Children', 'ci_lower']), float(conf.loc['Children', 'ci_upper']))

    coeff_inter = safe_get(params, 'Children_x_Gender')
    se_inter = safe_get(bse, 'Children_x_Gender')
    p_inter = safe_get(pvals, 'Children_x_Gender')
    ci_inter = None
    if conf is not None and 'Children_x_Gender' in conf.index:
        ci_inter = (float(conf.loc['Children_x_Gender', 'ci_lower']), float(conf.loc['Children_x_Gender', 'ci_upper']))

    # 3) Compute average predicted probability difference (Children=1 vs 0) overall and by gender using the original design matrix
    #    Using the model.exog as baseline, toggle Children and the interaction column accordingly.
    try:
        exog = np.asarray(logit.model.exog)  # shape (n, k)
        exog_names = list(logit.model.exog_names)
        # find column indices
        idx_children = exog_names.index('Children') if 'Children' in exog_names else None
        idx_gender = exog_names.index('Gender_Male') if 'Gender_Male' in exog_names else None
        idx_child_x_gender = exog_names.index('Children_x_Gender') if 'Children_x_Gender' in exog_names else None

        if idx_children is None or idx_gender is None or idx_child_x_gender is None:
            raise ValueError("Required columns not present in model.exog")

        exog0 = exog.copy()
        exog1 = exog.copy()

        # Set Children=0 and its interaction to 0
        exog0[:, idx_children] = 0
        exog0[:, idx_child_x_gender] = 0

        # Set Children=1 and interaction = Gender_Male * 1
        exog1[:, idx_children] = 1
        exog1[:, idx_child_x_gender] = exog1[:, idx_gender] * 1.0

        pred0 = logit.model.predict(exog0) if hasattr(logit.model, 'predict') else logit.predict(exog0)
        pred1 = logit.model.predict(exog1) if hasattr(logit.model, 'predict') else logit.predict(exog1)
        diff = pred1 - pred0
        avg_diff = float(np.mean(diff))
        se_avg_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))

        # By gender
        gender_vals = exog[:, idx_gender]
        if np.any(gender_vals == 1):
            pred0_men = pred0[gender_vals == 1]
            pred1_men = pred1[gender_vals == 1]
            diff_men = pred1_men - pred0_men
            avg_diff_men = float(np.mean(diff_men))
            se_avg_diff_men = float(np.std(diff_men, ddof=1) / np.sqrt(len(diff_men)))
        else:
            avg_diff_men = None
            se_avg_diff_men = None

        if np.any(gender_vals == 0):
            pred0_women = pred0[gender_vals == 0]
            pred1_women = pred1[gender_vals == 0]
            diff_women = pred1_women - pred0_women
            avg_diff_women = float(np.mean(diff_women))
            se_avg_diff_women = float(np.std(diff_women, ddof=1) / np.sqrt(len(diff_women)))
        else:
            avg_diff_women = None
            se_avg_diff_women = None

    except Exception as e:
        avg_diff = None
        se_avg_diff = None
        avg_diff_men = None
        se_avg_diff_men = None
        avg_diff_women = None
        se_avg_diff_women = None

    # 4) Prepare returned object
    result_object = {
        # Coefficients (log-odds) from logistic regression
        "coef_children_logodds": coeff_children,
        "se_children_logodds": se_children,
        "p_children": p_children,
        "ci_children_logodds": ci_children,
        "coef_children_x_gender_logodds": coeff_inter,
        "se_children_x_gender_logodds": se_inter,
        "p_children_x_gender": p_inter,
        "ci_children_x_gender_logodds": ci_inter,
        # Average marginal effects (probability differences)
        "ame_overall_prob_diff_children": avg_diff,
        "ame_overall_se": se_avg_diff,
        "ame_women_prob_diff_children": avg_diff_women,
        "ame_women_se": se_avg_diff_women,
        "ame_men_prob_diff_children": avg_diff_men,
        "ame_men_se": se_avg_diff_men,
        # If available, include the formal marginal effects table
        "marginal_effects_table": ame_df if ame_df is not None else None,
        # Descriptives
        "descriptives": None
    }

    # include descriptives if present (proportions of AnyAffair by Children)
    try:
        if descriptives is not None:
            # convert to plain dict for portability
            desc = descriptives.reset_index().rename(columns={'Children': 'Children_flag'})
            result_object["descriptives"] = desc.to_dict(orient='records')
        else:
            result_object["descriptives"] = None
    except Exception:
        result_object["descriptives"] = None

    # 5) Write concise interpretation
    # Use coefficient p-value and AME (if available) to form conclusion.
    interpretation_lines = []
    if avg_diff is not None:
        interpretation_lines.append(
            "Average marginal effect (Children=1 vs 0): {:.4f} (SE {:.4f}). "
            .format(avg_diff, se_avg_diff)
        )
        # direction
        if avg_diff < 0:
            interpretation_lines.append("On average, having children is associated with a LOWER probability of reporting any affair.")
        elif avg_diff > 0:
            interpretation_lines.append("On average, having children is associated with a HIGHER probability of reporting any affair.")
        else:
            interpretation_lines.append("No average change in probability.")
    else:
        interpretation_lines.append("Could not compute average marginal effect from the fitted model.")

    # Statistical significance notes from coefficient p-values
    if p_children is not None:
        interpretation_lines.append(
            "Logistic coef for 'Children' (log-odds) = {:.4f}, p = {:.3f}.".format(coeff_children, p_children)
        )
        # note typical alpha
        if p_children < 0.05:
            interpretation_lines.append("This effect is statistically significant at alpha=0.05.")
        elif p_children < 0.10:
            interpretation_lines.append("This effect is marginally significant (p<0.10).")
        else:
            interpretation_lines.append("This effect is not statistically significant at conventional levels.")
    if p_inter is not None:
        interpretation_lines.append(
            "Interaction 'Children_x_Gender' coefficient (log-odds) = {:.4f}, p = {:.3f}.".format(coeff_inter, p_inter)
        )
        if p_inter < 0.05:
            interpretation_lines.append("The interaction is statistically significant, suggesting the effect of children differs by gender.")
        elif p_inter < 0.10:
            interpretation_lines.append("The interaction is marginally significant (p<0.10), suggesting possible difference by gender.")
        else:
            interpretation_lines.append("The interaction is not statistically significant; evidence for gender differences is weak.")

    # Add gender-specific AMEs if computed
    if avg_diff_women is not None:
        interpretation_lines.append(
            "Estimated average effect for women (Children): {:.4f} (SE {:.4f}).".format(avg_diff_women, se_avg_diff_women)
        )
    if avg_diff_men is not None:
        interpretation_lines.append(
            "Estimated average effect for men (Children): {:.4f} (SE {:.4f}).".format(avg_diff_men, se_avg_diff_men)
        )

    # Short final verdict
    # Use AME and p-value for a concise yes/no:
    verdict = "Inconclusive"
    # If AME negative and p_children < 0.05 declare decrease; if p between .05 and .10 say weak evidence
    if avg_diff is not None and p_children is not None:
        if avg_diff < 0 and p_children < 0.05:
            verdict = "Yes — having children decreases probability of any affair (statistically significant)."
        elif avg_diff < 0 and p_children < 0.10:
            verdict = "Weak evidence that having children decreases probability of any affair (marginal significance)."
        elif avg_diff < 0:
            verdict = "No strong evidence that having children decreases affairs (effect estimated negative but not statistically significant)."
        elif avg_diff > 0 and p_children < 0.05:
            verdict = "Yes — having children increases probability of any affair (statistically significant)."
        elif avg_diff > 0:
            verdict = "No strong evidence that having children affects affairs (estimated effect positive but not statistically significant)."
        else:
            verdict = "No detectable effect of having children on affairs."

    interpretation_lines.append("Summary verdict: " + verdict)

    out["object"] = result_object
    out["description"] = " ".join(interpretation_lines)

    return out