def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of having children (ChildrenBinary) on
    the count of extramarital affairs from a fitted statsmodels GLM (NegativeBinomial).
    Also evaluates whether that effect differs by gender via the ChildrenBinary * gender_male interaction.

    Returns a dictionary with keys:
      - "object": dictionary of extracted statistics (coefficients, SEs, p-values,
                  confidence intervals, incidence rate ratios (IRR) and IRR CIs)
      - "description": textual interpretation of those statistics in context.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    params = res.params
    bse = res.bse
    pvals = res.pvalues
    cov = res.cov_params()
    ci_df = res.conf_int()  # DataFrame with [lower, upper] columns

    # Parameter names expected
    name_child = 'ChildrenBinary'
    name_inter = 'ChildrenBinary:gender_male'  # patsy/statsmodels naming for interaction

    if name_child not in params.index:
        raise KeyError(f"Expected parameter '{name_child}' not found in model parameters: {list(params.index)}")

    # Extract main effect (effect of children when gender_male = 0; i.e., females if gender_male coded 1=male)
    coef_child = float(params[name_child])
    se_child = float(bse[name_child])
    z_child = coef_child / se_child
    p_child = float(2 * stats.norm.sf(abs(z_child)))
    ci_child = tuple(ci_df.loc[name_child].values.tolist())
    irr_child = float(np.exp(coef_child))
    irr_child_ci = (float(np.exp(ci_child[0])), float(np.exp(ci_child[1])))

    # Interaction: may be absent if model didn't include it (but code included it)
    if name_inter in params.index:
        coef_inter = float(params[name_inter])
        se_inter = float(bse[name_inter])
        z_inter = coef_inter / se_inter
        p_inter = float(2 * stats.norm.sf(abs(z_inter)))
        ci_inter = tuple(ci_df.loc[name_inter].values.tolist())
    else:
        coef_inter = 0.0
        se_inter = 0.0
        z_inter = np.nan
        p_inter = np.nan
        ci_inter = (np.nan, np.nan)

    # Combined effect for males (gender_male = 1): sum of main + interaction
    coef_male = coef_child + coef_inter
    # variance of sum = var(child) + var(inter) + 2*cov(child,inter)
    if name_inter in cov.index:
        var_child = cov.loc[name_child, name_child]
        var_inter = cov.loc[name_inter, name_inter]
        cov_child_inter = cov.loc[name_child, name_inter]
        var_sum = var_child + var_inter + 2.0 * cov_child_inter
        se_male = float(np.sqrt(var_sum))
        z_male = coef_male / se_male
        p_male = float(2 * stats.norm.sf(abs(z_male)))
        # Confidence interval for coef_male (normal approx)
        ci_male = (coef_male - 1.96 * se_male, coef_male + 1.96 * se_male)
    else:
        # If interaction absent, male effect equals child effect
        se_male = se_child
        z_male = z_child
        p_male = p_child
        ci_male = ci_child

    irr_male = float(np.exp(coef_male))
    irr_male_ci = (float(np.exp(ci_male[0])), float(np.exp(ci_male[1])))

    # Build output object
    out = {
        "children_effect_female_scale": {
            "coef": coef_child,
            "se": se_child,
            "z": z_child,
            "p_value": p_child,
            "coef_ci_95": ci_child,
            "IRR": irr_child,
            "IRR_ci_95": irr_child_ci,
            "interpretation": "Effect of having children on log expected count of affairs for gender_male=0 (reference group)."
        },
        "interaction_children_by_male": {
            "coef": coef_inter,
            "se": se_inter,
            "z": z_inter,
            "p_value": p_inter,
            "coef_ci_95": ci_inter,
            "interpretation": "Additional change in the ChildrenBinary effect when gender_male=1 (i.e., difference in effect for males vs. reference)."
        },
        "children_effect_male_scale": {
            "coef": coef_male,
            "se": se_male,
            "z": z_male,
            "p_value": p_male,
            "coef_ci_95": ci_male,
            "IRR": irr_male,
            "IRR_ci_95": irr_male_ci,
            "interpretation": "Effect of having children on log expected count of affairs for gender_male=1 (males)."
        },
        # Summaries that are often most directly interpretable
        "summary": {
            "female_IRR": irr_child,
            "female_IRR_ci_95": irr_child_ci,
            "female_p_value": p_child,
            "male_IRR": irr_male,
            "male_IRR_ci_95": irr_male_ci,
            "male_p_value": p_male
        }
    }

    # Short textual description interpreting direction and significance
    def interpret(irr, p):
        if np.isnan(p):
            return "No p-value available; cannot assess statistical significance."
        if p < 0.05:
            if irr < 1.0:
                return "Statistically significant decrease in expected count (IRR < 1)."
            elif irr > 1.0:
                return "Statistically significant increase in expected count (IRR > 1)."
            else:
                return "No effect (IRR ~ 1) but statistically significant (rare)."
        else:
            if irr < 1.0:
                return "Point estimate suggests a decrease (IRR < 1) but not statistically significant."
            elif irr > 1.0:
                return "Point estimate suggests an increase (IRR > 1) but not statistically significant."
            else:
                return "No effect (IRR ~ 1) and not statistically significant."

    description_lines = []
    description_lines.append(
        "For gender_male = 0 (reference group): "
        f"coef = {coef_child:.4f}, IRR = {irr_child:.4f}, 95% CI for IRR = ({irr_child_ci[0]:.4f}, {irr_child_ci[1]:.4f}), p = {p_child:.4g}. "
        + interpret(irr_child, p_child)
    )
    description_lines.append(
        "Interaction (ChildrenBinary:gender_male): "
        f"coef = {coef_inter:.4f}, p = {p_inter:.4g}."
    )
    description_lines.append(
        "For gender_male = 1 (males): "
        f"coef = {coef_male:.4f}, IRR = {irr_male:.4f}, 95% CI for IRR = ({irr_male_ci[0]:.4f}, {irr_male_ci[1]:.4f}), p = {p_male:.4g}. "
        + interpret(irr_male, p_male)
    )

    description = " ".join(description_lines)

    return {"object": out, "description": description}