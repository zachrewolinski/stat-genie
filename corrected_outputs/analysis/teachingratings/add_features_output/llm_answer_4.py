def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for:
      - main effect of beauty (beauty_std)
      - interaction beauty_std:IsFemale (if present)
      - implied marginal effects of beauty for male (IsFemale=0) and female (IsFemale=1) instructors
    
    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "Brief interpretation of results in context"
      }
    """
    import numpy as np

    # Basic presence checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels results object (missing .params)")

    params = model_output.params
    bse = model_output.bse
    pvals = model_output.pvalues
    try:
        cov = model_output.cov_params()
    except Exception:
        # fallback: attempt attribute
        cov = getattr(model_output, "cov_params_default", None)
        if cov is None:
            raise

    # possible names for the interaction term (statsmodels uses ":" between factor names)
    inter_name_candidates = ["beauty_std:IsFemale", "IsFemale:beauty_std"]
    inter_name = next((n for n in inter_name_candidates if n in params.index), None)

    beauty_name = "beauty_std"
    if beauty_name not in params.index:
        raise ValueError(f"Expected '{beauty_name}' in model parameters but it was not found. Found parameters: {list(params.index)}")

    # Extract main coefficients
    coef_beauty = float(params[beauty_name])
    se_beauty = float(bse[beauty_name]) if beauty_name in bse.index else float(np.nan)
    t_beauty = coef_beauty / se_beauty if se_beauty != 0 else float("nan")
    # p-value from model output for the main term
    p_beauty = float(pvals[beauty_name]) if beauty_name in pvals.index else float("nan")
    # 95% CI using covariance matrix (or using the reported conf_int if desired)
    z = 1.96  # normal approx for 95% CI
    ci_beauty = (coef_beauty - z * se_beauty, coef_beauty + z * se_beauty)

    results = {
        "term_beauty_name": beauty_name,
        "coef_beauty": coef_beauty,
        "se_beauty": se_beauty,
        "t_beauty": t_beauty,
        "p_beauty": p_beauty,
        "ci_beauty_95": ci_beauty
    }

    # Interaction term: if present, extract it and compute marginal effect for females
    if inter_name is not None:
        coef_inter = float(params[inter_name])
        se_inter = float(bse[inter_name]) if inter_name in bse.index else float("nan")
        t_inter = coef_inter / se_inter if se_inter != 0 else float("nan")
        p_inter = float(pvals[inter_name]) if inter_name in pvals.index else float("nan")
        ci_inter = (coef_inter - z * se_inter, coef_inter + z * se_inter)

        # Marginal effect for males (IsFemale = 0): it's the main beauty coefficient
        effect_male = coef_beauty
        se_male = se_beauty
        t_male = t_beauty
        p_male = p_beauty
        ci_male = ci_beauty

        # Marginal effect for females (IsFemale = 1): sum of main and interaction
        effect_female = coef_beauty + coef_inter
        # SE(effect_female) = sqrt(Var(b) + Var(b_inter) + 2*Cov(b, b_inter))
        try:
            var_b = float(cov.loc[beauty_name, beauty_name])
            var_inter = float(cov.loc[inter_name, inter_name])
            cov_b_inter = float(cov.loc[beauty_name, inter_name])
            var_female = var_b + var_inter + 2.0 * cov_b_inter
            se_female = float(np.sqrt(var_female)) if var_female >= 0 else float("nan")
        except Exception:
            # If covariance matrix lookup fails, fall back to naive sum of SEs (conservative/incorrect)
            se_female = float(np.nan)

        t_female = effect_female / se_female if (se_female is not None and not np.isnan(se_female) and se_female != 0) else float("nan")

        # p-value for female marginal effect: use t with residual df if available, else normal approx
        p_female = float("nan")
        try:
            # attempt to use scipy.stats for t cdf
            from scipy import stats
            df_resid = getattr(model_output, "df_resid", None)
            if se_female is not None and not np.isnan(se_female):
                if df_resid is not None and np.isfinite(df_resid) and df_resid > 0:
                    p_female = 2.0 * (1.0 - stats.t.cdf(abs(t_female), df=df_resid))
                else:
                    p_female = 2.0 * (1.0 - stats.norm.cdf(abs(t_female)))
        except Exception:
            # fallback to normal approx
            try:
                import math
                p_female = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_female) / math.sqrt(2.0))))
            except Exception:
                p_female = float("nan")

        # 95% CI for female effect
        ci_female = (effect_female - z * se_female, effect_female + z * se_female) if not np.isnan(se_female) else (float("nan"), float("nan"))

        results.update({
            "term_interaction_name": inter_name,
            "coef_interaction": coef_inter,
            "se_interaction": se_inter,
            "t_interaction": t_inter,
            "p_interaction": p_inter,
            "ci_interaction_95": ci_inter,
            "effect_male_beauty_coef": effect_male,
            "effect_male_se": se_male,
            "effect_male_t": t_male,
            "effect_male_p": p_male,
            "effect_male_ci95": ci_male,
            "effect_female_beauty_coef": effect_female,
            "effect_female_se": se_female,
            "effect_female_t": t_female,
            "effect_female_p": p_female,
            "effect_female_ci95": ci_female
        })
    else:
        # No interaction term: effect is same for both genders
        results.update({
            "term_interaction_name": None,
            "note": "No beauty_std:IsFemale interaction term found; reported beauty effect applies to both genders."
        })

    # A short text description explaining what the numbers mean in context
    if results.get("term_interaction_name") is not None:
        description = (
            "Summary of effect of standardized instructor attractiveness (beauty_std) on teaching evaluations (Eval). "
            "coef_beauty is the effect of a 1-SD increase in beauty for male instructors (IsFemale=0). "
            "coef_interaction is how much that effect differs for female instructors; the female marginal effect = coef_beauty + coef_interaction. "
            "For each reported effect we provide standard errors, t-statistics, p-values, and approximate 95% confidence intervals. "
            "If p < 0.05 the effect is commonly considered statistically significant (two-sided)."
        )
    else:
        description = (
            "Summary of effect of standardized instructor attractiveness (beauty_std) on teaching evaluations (Eval). "
            "No interaction with instructor gender was included in the fitted model, so the reported coef_beauty is the estimated change in Eval "
            "for a 1-SD increase in beauty for all instructors (holding controls constant). "
            "Standard errors, t-statistics, p-values, and approximate 95% confidence intervals are provided."
        )

    return {"object": results, "description": description}