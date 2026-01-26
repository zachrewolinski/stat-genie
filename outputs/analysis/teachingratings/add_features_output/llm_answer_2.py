def extract_final_answer(model_output):
    """
    Extract relevant statistics about the effect of instructor beauty on course evaluations
    from a fitted statsmodels RegressionResultsWrapper (with clustered SEs).
    
    Returns a dictionary with:
      - "object": a dict containing coefficients, p-values, 95% CIs for the beauty terms
                  and estimated marginal effects (and SEs/p-values) for male/female
                  at beauty z = 0 (mean) and z = +1 SD.
      - "description": a brief, plain-language interpretation of those statistics.
    """
    import numpy as np
    from scipy.stats import norm

    # Parameter names expected in the fitted model
    name_b1 = 'Beauty_z'
    name_b2 = 'Beauty_sq'
    name_int = 'Beauty_z:Gender_Female'
    required_names = [name_b1, name_b2, name_int, 'Gender_Female']

    # Basic availability checks
    params = model_output.params
    pvalues = model_output.pvalues
    try:
        conf_int = model_output.conf_int()
    except Exception:
        # fallback: use params +/- 1.96*bse if conf_int not available
        bse = model_output.bse
        conf_int = np.vstack([params - 1.96 * bse, params + 1.96 * bse]).T
        conf_int = dict(zip(params.index, conf_int))

    for nm in required_names:
        if nm not in params.index:
            raise ValueError(f"Expected parameter '{nm}' not found in model results. "
                             f"Available params: {list(params.index)}")

    # Extract coefficients, p-values, and 95% CIs for beauty terms and interaction
    coef_b1 = float(params[name_b1])
    coef_b2 = float(params[name_b2])
    coef_int = float(params[name_int])
    p_b1 = float(pvalues[name_b1])
    p_b2 = float(pvalues[name_b2])
    p_int = float(pvalues[name_int])

    # Confidence intervals: statsmodels returns DataFrame-like; handle both types
    try:
        ci_b1 = tuple(conf_int.loc[name_b1].astype(float).tolist())
        ci_b2 = tuple(conf_int.loc[name_b2].astype(float).tolist())
        ci_int = tuple(conf_int.loc[name_int].astype(float).tolist())
    except Exception:
        # conf_int may be an ndarray-like or dict
        if isinstance(conf_int, dict):
            ci_b1 = tuple(conf_int[name_b1])
            ci_b2 = tuple(conf_int[name_b2])
            ci_int = tuple(conf_int[name_int])
        else:
            # assume numpy array with index alignment
            ci_b1 = tuple(conf_int[params.index.get_loc(name_b1)])
            ci_b2 = tuple(conf_int[params.index.get_loc(name_b2)])
            ci_int = tuple(conf_int[params.index.get_loc(name_int)])

    # Covariance matrix for delta method (DataFrame or ndarray)
    cov = model_output.cov_params()
    # Ensure cov is a DataFrame-like with .loc indexing; if ndarray convert to np.ndarray and map indices
    if hasattr(cov, 'loc'):
        cov_b1_b1 = float(cov.loc[name_b1, name_b1])
        cov_b2_b2 = float(cov.loc[name_b2, name_b2])
        cov_int_int = float(cov.loc[name_int, name_int])
        cov_b1_b2 = float(cov.loc[name_b1, name_b2])
        cov_b1_int = float(cov.loc[name_b1, name_int])
        cov_b2_int = float(cov.loc[name_b2, name_int])
    else:
        # cov is ndarray; find indices
        idx_map = {n: i for i, n in enumerate(params.index)}
        i1 = idx_map[name_b1]; i2 = idx_map[name_b2]; ii = idx_map[name_int]
        cov_b1_b1 = float(cov[i1, i1])
        cov_b2_b2 = float(cov[i2, i2])
        cov_int_int = float(cov[ii, ii])
        cov_b1_b2 = float(cov[i1, i2])
        cov_b1_int = float(cov[i1, ii])
        cov_b2_int = float(cov[i2, ii])

    # Function to compute marginal effect of beauty (derivative of Eval w.r.t. beauty_z)
    # effect = b1 + b_int * female + 2 * b2 * z
    # variance via delta method:
    # Var(effect) = Var(b1) + (female^2) Var(b_int) + (2z)^2 Var(b2)
    #              + 2*female*Cov(b1,b_int) + 2*(2z)*Cov(b1,b2) + 2*female*(2z)*Cov(b_int,b2)
    def marginal_effect(z, female_flag):
        f = 1.0 if female_flag else 0.0
        effect = coef_b1 + coef_int * f + 2.0 * coef_b2 * z
        var = (
            cov_b1_b1
            + (f ** 2) * cov_int_int
            + (2.0 * z) ** 2 * cov_b2_b2
            + 2.0 * f * cov_b1_int
            + 2.0 * (2.0 * z) * cov_b1_b2
            + 2.0 * f * (2.0 * z) * cov_b2_int
        )
        # numerical protection
        var = max(var, 0.0)
        se = float(np.sqrt(var))
        z_stat = effect / se if se > 0 else np.nan
        p_two = float(2.0 * norm.sf(abs(z_stat))) if se > 0 else np.nan
        return {"z": float(z), "female": bool(female_flag),
                "effect": float(effect), "se": se, "z_stat": float(z_stat), "p_value": p_two}

    # Compute marginal effects at mean (z=0) and +1 SD (z=1) for male and female
    me_male_z0 = marginal_effect(z=0.0, female_flag=False)
    me_female_z0 = marginal_effect(z=0.0, female_flag=True)
    me_male_z1 = marginal_effect(z=1.0, female_flag=False)
    me_female_z1 = marginal_effect(z=1.0, female_flag=True)

    # Assemble object to return
    output_object = {
        "coefficients": {
            "Beauty_z": {"coef": coef_b1, "p_value": p_b1, "95%CI": ci_b1},
            "Beauty_sq": {"coef": coef_b2, "p_value": p_b2, "95%CI": ci_b2},
            "Beauty_z:Gender_Female": {"coef": coef_int, "p_value": p_int, "95%CI": ci_int},
        },
        "marginal_effects": {
            "male_z0": me_male_z0,
            "female_z0": me_female_z0,
            "male_z1": me_male_z1,
            "female_z1": me_female_z1,
        },
        "notes": (
            "Marginal effect = derivative of Eval with respect to standardized beauty (Beauty_z). "
            "Effects and p-values for marginal effects are computed using the delta method and a normal "
            "approximation (two-sided). 'z' refers to the value of Beauty_z at which the marginal "
            "effect is evaluated (0 = mean, 1 = one SD above mean)."
        )
    }

    # Short interpretation string (concise)
    # It summarizes sign and statistical evidence for beauty's linear effect & interaction.
    desc_lines = []
    desc_lines.append(
        f"The linear beauty coefficient (Beauty_z) = {coef_b1:.3f} (p = {p_b1:.3g}), "
        f"the quadratic term (Beauty_sq) = {coef_b2:.3f} (p = {p_b2:.3g}), "
        f"and the interaction (Beauty_z:Gender_Female) = {coef_int:.3f} (p = {p_int:.3g})."
    )
    desc_lines.append(
        "At average beauty (z=0), the marginal effect on Eval is equal to the linear coefficient:"
        f" male = {me_male_z0['effect']:.3f} (SE={me_male_z0['se']:.3f}, p={me_male_z0['p_value']:.3g});"
        f" female = {me_female_z0['effect']:.3f} (SE={me_female_z0['se']:.3f}, p={me_female_z0['p_value']:.3g})."
    )
    desc_lines.append(
        "At +1 SD in beauty (z=1), the marginal effects are:"
        f" male = {me_male_z1['effect']:.3f} (p={me_male_z1['p_value']:.3g});"
        f" female = {me_female_z1['effect']:.3f} (p={me_female_z1['p_value']:.3g})."
    )
    desc_lines.append(
        "Interpretation: positive marginal effect means higher beauty is associated with higher evaluation scores. "
        "Statistical significance should be judged from the p-values and confidence intervals above."
    )
    description = " ".join(desc_lines)

    return {"object": output_object, "description": description}