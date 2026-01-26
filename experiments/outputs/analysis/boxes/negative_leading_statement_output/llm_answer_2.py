def extract_final_answer(model_output):
    """
    Extract key statistics relevant to whether developmental (age) effects
    on reliance on social information and majority preference vary across cultures.

    Returns a dict with:
      - "object": dict of extracted numeric results (LR-test p-values, multinomial age coefficients and p-values if available)
      - "description": concise interpretation of those results in context
    """
    import numpy as np
    out = {"object": {}, "description": ""}

    # 1) Extract LR-test results for interaction (age x culture) from the two logistic analyses
    try:
        cd = model_output.get('chose_demonstrated', {})
        lr_p_demo = cd.get('lr_pvalue', None)
        out["object"]['chose_demonstrated_lr_pvalue'] = float(lr_p_demo) if lr_p_demo is not None else None
    except Exception:
        out["object"]['chose_demonstrated_lr_pvalue'] = None

    try:
        cm = model_output.get('chose_majority', {})
        lr_p_major = cm.get('lr_pvalue', None)
        out["object"]['chose_majority_lr_pvalue'] = float(lr_p_major) if lr_p_major is not None else None
    except Exception:
        out["object"]['chose_majority_lr_pvalue'] = None

    # 2) Extract age coefficients (and p-values if available) from the multinomial model
    out["object"]['mnlogit_age_coef'] = {}
    out["object"]['mnlogit_age_pvalue'] = {}
    try:
        mn = model_output.get('mnlogit', None)
        if mn is not None:
            # statsmodels MNLogit results generally expose .params and .pvalues as DataFrames
            params = getattr(mn, 'params', None)
            pvals = getattr(mn, 'pvalues', None)

            if params is not None:
                # params is indexed by variable name; columns correspond to non-reference categories
                try:
                    if 'age_c' in params.index:
                        age_params = params.loc['age_c']
                    else:
                        # sometimes params is shaped differently (e.g., MultiIndex). Try alternative access:
                        age_params = params.loc['age_c', :]
                    # Convert to a simple dict mapping category idx -> coef
                    for col in age_params.index:
                        out["object"]['mnlogit_age_coef'][str(col)] = float(age_params[col])
                except Exception:
                    out["object"]['mnlogit_age_coef'] = None

            if pvals is not None:
                try:
                    if 'age_c' in pvals.index:
                        age_p = pvals.loc['age_c']
                    else:
                        age_p = pvals.loc['age_c', :]
                    for col in age_p.index:
                        out["object"]['mnlogit_age_pvalue'][str(col)] = float(age_p[col])
                except Exception:
                    # if p-values not available in the expected structure, leave None
                    if out["object"]['mnlogit_age_coef'] and not out["object"]['mnlogit_age_pvalue']:
                        out["object"]['mnlogit_age_pvalue'] = None
            else:
                if out["object"]['mnlogit_age_coef'] and not out["object"]['mnlogit_age_pvalue']:
                    out["object"]['mnlogit_age_pvalue'] = None
        else:
            out["object"]['mnlogit_age_coef'] = None
            out["object"]['mnlogit_age_pvalue'] = None
    except Exception:
        out["object"]['mnlogit_age_coef'] = None
        out["object"]['mnlogit_age_pvalue'] = None

    # 3) Build a succinct interpretation
    parts = []
    # Interpretation for chose_demonstrated interaction
    p_demo = out["object"].get('chose_demonstrated_lr_pvalue', None)
    if p_demo is None:
        parts.append("Could not find LR-test p-value for the interaction on 'chose_demonstrated'.")
    else:
        if p_demo < 0.05:
            parts.append(
                f"The likelihood-ratio test for the age × culture interaction on whether children chose a demonstrated option "
                f"is significant (p = {p_demo:.3f}). This indicates that developmental (age) effects on reliance on social information "
                f"(vs choosing an undemonstrated option) differ across cultural sites."
            )
        else:
            parts.append(
                f"The LR-test for age × culture on 'chose_demonstrated' is not significant (p = {p_demo:.3f}), "
                f"suggesting no evidence that age effects on reliance on social information differ by culture."
            )

    # Interpretation for chose_majority interaction
    p_major = out["object"].get('chose_majority_lr_pvalue', None)
    if p_major is None:
        parts.append("Could not find LR-test p-value for the interaction on 'chose_majority'.")
    else:
        if p_major < 0.05:
            parts.append(
                f"The LR-test for the age × culture interaction on preference for the majority among those who copied is significant (p = {p_major:.3f}), "
                f"indicating cross-cultural differences in how majority preference develops with age."
            )
        else:
            parts.append(
                f"The LR-test for age × culture on 'chose_majority' is not significant (p = {p_major:.3f}). "
                f"Thus there is no evidence that developmental effects on majority preference among demonstrators vary across cultures."
            )

    # Add note about overall age coefficients from multinomial (if available)
    age_coefs = out["object"].get('mnlogit_age_coef', None)
    age_pvals = out["object"].get('mnlogit_age_pvalue', None)
    if age_coefs:
        # Map of category column to coef. We should explain which column corresponds to which choice:
        # As in the original model, categories were coded so that reference is the undemonstrated option (original y==1),
        # and the two equations are for y==2 (majority) and y==3 (minority). The columns in params correspond to those two non-reference categories.
        coef_parts = []
        for col, coef in age_coefs.items():
            pval = None
            if age_pvals and col in age_pvals:
                pval = age_pvals[col]
            if pval is not None:
                coef_parts.append(f"category {col}: coef = {coef:.3f}, p = {pval:.3f}")
            else:
                coef_parts.append(f"category {col}: coef = {coef:.3f} (p-value unavailable)")
        parts.append(
            "Multinomial model age coefficients (each vs. undemonstrated): " + "; ".join(coef_parts) +
            ". Negative coefficients indicate older children were less likely to choose the demonstrated alternatives relative to the undemonstrated option."
        )
    else:
        parts.append("Multinomial-model age coefficients were not available or could not be extracted.")

    out["description"] = " ".join(parts)
    return out